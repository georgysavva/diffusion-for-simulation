import math
import os
import time
from functools import partial
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from diffusers import AutoencoderKL
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import wandb
from src.data import (
    BatchSampler,
    Dataset,
    TestDatasetTraverser,
    collate_segments_to_batch,
)
from src.data.episode import Episode
from src.diffusion import create_diffusion
from src.traj_eval import TrajectoryEvaluator, actions_to_captions
from src.utils import (
    count_parameters,
    get_warmup_lr_sched,
    keep_model_copies_every,
    prepare_image_obs,
    save_as_video,
    set_seed,
    to_concatenated_images_with_text,
    wandb_log,
)


class Trainer:

    def __init__(self, cfg: DictConfig, root_dir: Path) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        if cfg.debug:
            cfg.wandb.mode = "disabled"
            cfg.diffusion_model.training.train_batch_size = 1
            cfg.static_dataset.auto_regressive_steps = 2
            cfg.diffusion_model.training.eval_batch_size = 2
            cfg.diffusion_model.training.lr_warmup_steps = 2
            cfg.diffusion_model.training.lr_decay_every_epoch = 2
            cfg.diffusion_model.training.lr_decay_factor = 0.1
            cfg.training.epoch_size = 4
            cfg.inference.every = 1
            cfg.inference.num_generated_frames = 2
            cfg.inference.vae_batch_size = 2
            cfg.evaluation.sub_sample_rate = 20000

        OmegaConf.resolve(cfg)
        self._cfg = cfg
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Pick a random seed
        set_seed(torch.seed() % 10**9)

        # Device
        if torch.cuda.is_available():
            self._device = torch.device("cuda", self._rank)
        else:
            self._device = torch.device("cpu")
        print(f"Starting on {self._device}")
        self._use_cuda = self._device.type == "cuda"
        if self._use_cuda:
            torch.cuda.set_device(
                self._rank
            )  # fix compilation error on multi-gpu nodes

        # Init wandb
        if self._rank == 0:
            assert cfg.experiment_name, "experiment_name must be provided in hydra"
            wandb.init(
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=True,
                **cfg.wandb,
            )

        # Checkpointing
        self.run_dir = Path(cfg.common.run_dir)
        if self._rank == 0:
            print("Run dir:", self.run_dir)
        self._keep_model_copies = partial(
            keep_model_copies_every,
            every=cfg.checkpointing.save_diffusion_model_every,
            path_ckpt_dir=self.run_dir,
            num_to_keep=cfg.checkpointing.num_to_keep,
        )

        num_workers = cfg.training.num_workers_data_loaders
        p = Path(cfg.static_dataset.path)
        self.train_dataset = Dataset(
            p / "train",
            num_episodes=100,
        )
        self.test_dataset = Dataset(
            p / "test",
            num_episodes=10,
        )

        # Create models
        if self._rank == 0:
            print("Instantiating model")
        self.diffusion_model = instantiate(cfg.diffusion_model.model).to(self._device)
        if self._rank == 0:
            print(f"{count_parameters(self.diffusion_model)} parameters")
        self._diffusion_model = (
            DDP(self.diffusion_model, device_ids=[self._rank], output_device=self._rank)
            if dist.is_initialized()
            else self.diffusion_model
        )
        assert (
            cfg.initialization.pretrained_weights_path is None
            or cfg.initialization.path_to_ckpt is None
        ), "Only one of pretrained_weights_path or path_to_ckpt should be provided"
        if cfg.initialization.pretrained_weights_path is not None:
            weights = torch.load(
                Path(cfg.initialization.pretrained_weights_path),
                map_location=self._device,
                weights_only=True,
            )
            self.diffusion_model.load_pretrained_weights(weights)

        if cfg.initialization.path_to_ckpt is not None:
            sd = torch.load(
                Path(cfg.initialization.path_to_ckpt),
                map_location=self._device,
                weights_only=True,
            )
            self.diffusion_model.load_state_dict(sd)

        ######################################################
        self._train_batch_size = cfg.diffusion_model.training.train_batch_size
        self._eval_batch_size = cfg.diffusion_model.training.eval_batch_size  
        # Optimizers and LR schedulers
        optim_cfg = cfg.diffusion_model.training.optimizer
        self.opt = torch.optim.AdamW(
            self.diffusion_model.parameters(),
            lr=(
                optim_cfg.base_lr * self._train_batch_size * self._world_size
                if optim_cfg.scale_lr
                else optim_cfg.base_lr
            ),
        )

        self.warmup_lr_sched = get_warmup_lr_sched(
            self.opt, cfg.diffusion_model.training.lr_warmup_steps
        )
        self.lr_sched = torch.optim.lr_scheduler.StepLR(
            self.opt,
            cfg.diffusion_model.training.lr_decay_every_epoch,
            gamma=cfg.diffusion_model.training.lr_decay_factor,
        )
        # Data loaders

        batch_sampler = BatchSampler(
            self.train_dataset,
            self._rank,
            self._world_size,
            self._train_batch_size,
            cfg.static_dataset.seed_seq_length,
            cfg.diffusion_model.model.num_conditioning_steps,
            cfg.static_dataset.auto_regressive_steps,
        )

        self._data_loader_train = DataLoader(
            dataset=self.train_dataset,
            collate_fn=collate_segments_to_batch,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=self._use_cuda,
            pin_memory_device=str(self._device) if self._use_cuda else "",
            batch_sampler=batch_sampler,
        )

        self._data_loader_test = TestDatasetTraverser(
            self.test_dataset,
            self._eval_batch_size,
            cfg.evaluation.sub_sample_rate,
            self._rank,
            self._world_size,
            cfg.static_dataset.seed_seq_length,
            cfg.diffusion_model.model.num_conditioning_steps,
            cfg.static_dataset.auto_regressive_steps,
        )
        self.auto_regressive_length = cfg.static_dataset.auto_regressive_steps
        self.num_conditioning_steps = cfg.diffusion_model.model.num_conditioning_steps

        # Training state (things to be saved/restored)
        self.epoch = 0
        self.global_step = 0

        self.diffusion = create_diffusion(
            timestep_respacing=str(cfg.diffusion.num_sampling_steps),
            learn_sigma=cfg.diffusion.learn_sigma,
        )  # default: 1000 steps, linear noise schedule
        self._setup_inference(cfg)
        if cfg.diffusion.sampling_algorithm == "DDPM":
            self._sampling_function = self.diffusion.p_sample_loop
        elif cfg.diffusion.sampling_algorithm == "DDIM":
            self._sampling_function = self.diffusion.ddim_sample_loop
        else:
            raise ValueError(
                f"Unknown sampling algorithm: {cfg.diffusion.sampling_algorithm}"
            )

    def _setup_inference(self, cfg: DictConfig) -> None:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(
            self._device
        )
        if cfg.inference.vae_path:
            vae.decoder.load_state_dict(
                torch.load(
                    cfg.inference.vae_path, weights_only=True, map_location=self._device
                )
            )
        vae.eval()
        episode_path = Path(cfg.inference.episode_path)
        episode = Episode.load(episode_path)
        episode.obs = prepare_image_obs(
            episode.obs, cfg.static_dataset.image_resolution
        )
        episode = episode.slice(
            0,
            cfg.static_dataset.seed_seq_length + cfg.inference.num_generated_frames,
        )

        self.inference_episode = episode
        self.inference_episode_name = os.path.splitext(episode_path.name)[0]

        self.trajectory_evaluator = TrajectoryEvaluator(
            diffusion=self.diffusion,
            vae=vae,
            num_seed_steps=cfg.static_dataset.seed_seq_length,
            num_conditioning_steps=cfg.diffusion_model.model.num_conditioning_steps,
            sampling_algorithm=cfg.diffusion.sampling_algorithm,
            vae_batch_size=cfg.inference.vae_batch_size,
            device=self._device,
        )

        self.inference_action_captions = [""] + actions_to_captions(
            self.inference_episode.act[:-1], cfg.env.id
        )

    def run(self) -> None:

        num_epochs = self._cfg.training.num_epochs

        while self.epoch < num_epochs:
            self.epoch += 1
            start_time = time.time()

            if self._rank == 0:
                print(f"\nEpoch {self.epoch} / {num_epochs}\n")

            if self._cfg.training.should:
                self.train_diffusion_model()

            # Evaluation
            should_test = self._cfg.evaluation.should and (
                self.epoch % self._cfg.evaluation.every == 0
            )

            if should_test:
                self.test_diffusion_model()

            # Inference
            should_inference = self._cfg.inference.should and (
                self.epoch % self._cfg.inference.every == 0
            )

            if should_inference:
                self.inference_diffusion_model()

            # Logging
            if self._rank == 0:
                wandb_log(
                    {"duration": (time.time() - start_time) / 3600},
                    self.epoch,
                )
                wandb_log({"step": self.global_step}, self.epoch)
                wandb_log({"epoch": self.epoch}, self.epoch)

            # Checkpointing
            self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

            # if not training, no need to repeatedly eval or inference
            if not self._cfg.training.should:
                break

    def train_diffusion_model(self):
        self.diffusion_model.train()
        assert (
            self._cfg.training.epoch_size % (self._train_batch_size * self._world_size)
            == 0
        ), "epoch_size should be divisible by train_batch_size * world_size * auto_regressive_length"
        num_steps = self._cfg.training.epoch_size // (
            self._train_batch_size * self._world_size * self.auto_regressive_length
        )
        model = self._diffusion_model
        opt = self.opt
        data_loader = self._data_loader_train

        opt.zero_grad()
        data_iterator = iter(data_loader)
        train_loss = 0.0
        train_loss_auto_regressive = 0.0
        for _ in trange(num_steps, desc=f"Training", disable=self._rank > 0):
            self.global_step = self.global_step + 1
            batch = next(data_iterator).to(self._device)
            losses = self.call_model_autoregressively(model, batch)
            train_loss += losses[0]
            train_loss_auto_regressive += sum(losses) / len(losses)

            if self.global_step <= self._cfg.diffusion_model.training.lr_warmup_steps:
                self.warmup_lr_sched.step()
        train_loss = train_loss / num_steps
        train_loss_auto_regressive = train_loss_auto_regressive / num_steps
        if self._world_size > 1:
            train_loss = self.average_across_processes(train_loss)
            train_loss_auto_regressive = self.average_across_processes(
                train_loss_auto_regressive
            )
        self.lr_sched.step()
        to_log = {
            "loss": train_loss,
            "loss_auto_regressive": train_loss_auto_regressive,
            "lr": opt.param_groups[0]["lr"],
        }
        to_log = {f"train/{k}": v for k, v in to_log.items()}
        if self._rank == 0:
            wandb_log(to_log, self.epoch)

    @torch.no_grad()
    def test_diffusion_model(self):
        self.diffusion_model.eval()
        model = self.diffusion_model
        data_loader = self._data_loader_test
        eval_loss = 0.0
        eval_loss_auto_regressive = 0.0
        for batch in tqdm(data_loader, desc="Evaluating", disable=self._rank > 0):
            batch = batch.to(self._device)
            losses = self.call_model_autoregressively(model, batch, evaluate=True)
            eval_loss += losses[0]
            eval_loss_auto_regressive += sum(losses) / len(losses)

        eval_loss = eval_loss / len(data_loader)
        eval_loss_auto_regressive = eval_loss_auto_regressive / len(data_loader)
        if self._world_size > 1:
            eval_loss = self.average_across_processes(eval_loss)
            eval_loss_auto_regressive = self.average_across_processes(
                eval_loss_auto_regressive
            )
        to_log = {"loss": eval_loss, "loss_auto_regressive": eval_loss_auto_regressive}

        to_log = {f"test/{k}": v for k, v in to_log.items()}

        if self._rank == 0:
            wandb_log(to_log, self.epoch)

    @torch.no_grad()
    def inference_diffusion_model(self):
        self.diffusion_model.eval()
        output_dir = (
            self.run_dir
            / "trajectory_evaluation"
            / f"diffusion_model_epoch_{self.epoch:05d}"
            / self.inference_episode_name
        )
        if self._rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        for generation_mode in self._cfg.inference.generation_mode:
            generated_trajectory, psnr = self.trajectory_evaluator.evaluate_episode(
                self.diffusion_model,
                self.inference_episode,
                generation_mode,
                disable_progress=self._rank > 0,
            )
            if self._rank == 0:
                wandb_log({f"inference/PSNR_{generation_mode}": psnr}, self.epoch)
                save_as_video(
                    generated_trajectory,
                    output_dir
                    / f"generated_{generation_mode}_{self._cfg.diffusion.sampling_algorithm}.mp4",
                    fps=self._cfg.inference.video_fps,
                )
                generated_img = to_concatenated_images_with_text(
                    generated_trajectory,
                    self.inference_action_captions,
                )
                generated_img.save(
                    output_dir
                    / f"generated_{generation_mode}_{self._cfg.diffusion.sampling_algorithm}.png"
                )
                wandb_log(
                    {
                        f"inference/generated_{generation_mode}": wandb.Image(
                            generated_img
                        )
                    },
                    self.epoch,
                )
        if self._rank == 0:
            ground_truth_trajectory = self.inference_episode.obs
            save_as_video(
                ground_truth_trajectory,
                output_dir / "ground_truth.mp4",
                fps=self._cfg.inference.video_fps,
            )
            ground_truth_img = to_concatenated_images_with_text(
                ground_truth_trajectory,
                self.inference_action_captions,
            )
            ground_truth_img.save(output_dir / "ground_truth.png")
            wandb_log(
                {"inference/ground_truth": wandb.Image(ground_truth_img)}, self.epoch
            )

    def save_checkpoint(self) -> None:
        if self._rank == 0:
            self._keep_model_copies(self.diffusion_model.state_dict(), self.epoch)

    def call_model_autoregressively(self, model, batch, evaluate=False):
        obs, act = batch.obs, batch.act
        prev_obs = obs[:, : self.num_conditioning_steps]
        prev_act = act[:, : self.num_conditioning_steps]
        losses = []
        for i in range(self.auto_regressive_length):
            n = obs.shape[0]
            t = torch.randint(
                0, self.diffusion.num_timesteps, (n,), device=self._device
            )
            model_kwargs = dict(prev_obs=prev_obs, prev_act=prev_act)
            current_obs = obs[:, self.num_conditioning_steps + i]
            if evaluate:
                self.diffusion_model.eval()
                with torch.no_grad():
                    loss_dict = self.diffusion.training_losses(
                        model, current_obs, t, model_kwargs
                    )
            else:
                self.diffusion_model.train()
                loss_dict = self.diffusion.training_losses(
                    model, current_obs, t, model_kwargs
                )

            loss = loss_dict["loss"].mean()
            if not evaluate:
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
            losses.append(loss.item())
            self.diffusion_model.eval()
            with torch.no_grad():
                z = torch.randn(*current_obs.shape, device=self._device)
                model_kwargs = dict(prev_obs=prev_obs, prev_act=prev_act)
                generated_obs = self._sampling_function(
                    model.forward,
                    z.shape,
                    z,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=False,
                    device=self._device,
                )
                prev_obs = torch.roll(prev_obs, -1, 1)
                prev_obs[:, -1] = generated_obs
                prev_act = torch.roll(prev_act, -1, 1)
                prev_act[:, -1] = act[:, self.num_conditioning_steps + i]

        return losses

    def average_across_processes(self, value):
        """
        Utility to get the mean of `value` across all processes (GPUs).
        Assumes `value` is a Python float or torch scalar on rank's CPU.

        Steps:
          - Convert to a tensor
          - All-reduce (sum)
          - Divide by world_size
          - Return the average (as float)
        """
        tensor_value = torch.tensor(value, dtype=torch.float, device=self._device)
        dist.all_reduce(tensor_value, op=dist.ReduceOp.SUM)
        return (tensor_value / self._world_size).item()
