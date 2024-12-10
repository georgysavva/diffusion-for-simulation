import os
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers import AutoencoderKL
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
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
from src.traj_eval import TrajectoryEvaluator
from src.utils import (
    build_ddp_wrapper,
    count_parameters,
    download_model_weights,
    get_lr_sched,
    keep_model_copies_every,
    prepare_image_obs,
    set_seed,
    to_numpy_video,
    wandb_log,
)


class Trainer:
    def __init__(self, cfg: DictConfig, root_dir: Path) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
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
        if self._rank == 0 and self._cfg.wandb.do_log:
            assert cfg.experiment_name, "experiment_name must be provided in hydra"
            wandb.init(
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=True,
                **cfg.wandb,
            )

        # Checkpointing

        self._keep_model_copies = partial(
            keep_model_copies_every,
            every=cfg.checkpointing.save_diffusion_model_every,
            path_ckpt_dir=Path(cfg.common.run_dir),
            num_to_keep=cfg.checkpointing.num_to_keep,
        )

        num_workers = cfg.training.num_workers_data_loaders
        p = Path(cfg.static_dataset.path)
        self.train_dataset = Dataset(
            p / "train",
            guarantee_full_seqs=cfg.static_dataset.guarantee_full_seqs,
            cache_in_ram=False,
        )
        self.test_dataset = Dataset(
            p / "test",
            cfg.static_dataset.guarantee_full_seqs,
            cache_in_ram=True,
        )

        # Create models
        if self._rank == 0:
            print("Instantiating model")
        self.diffusion_model = instantiate(cfg.diffusion_model.model).to(self._device)
        if self._rank == 0:
            print(f"{count_parameters(self.diffusion_model)} parameters")
        self._diffusion_model = (
            build_ddp_wrapper(**self.diffusion_model._modules)
            if dist.is_initialized()
            else self.diffusion_model
        )
        assert (
            cfg.pretrained_weights_url is None
            or cfg.initialization.path_to_ckpt is None
        ), "Only one of pretrained_weights_url or path_to_ckpt should be provided"
        if cfg.pretrained_weights_url is not None:
            weights = download_model_weights(
                cfg.pretrained_weights_url,
                os.path.join(
                    cfg.common.project_storage_base_path, "pretrained_weights"
                ),
                self._device,
            )
            self.diffusion_model.load_pretrained_weights(weights)

        if cfg.initialization.path_to_ckpt is not None:
            sd = torch.load(
                Path(cfg.initialization.path_to_ckp),
                map_location=self._device,
                weights_only=True,
            )
            self.diffusion_model.load_state_dict(sd)

        ######################################################

        # Optimizers and LR schedulers

        optim_cfg = cfg.diffusion_model.training.optimizer
        self.opt = torch.optim.AdamW(
            self.diffusion_model.parameters(),
            lr=(
                optim_cfg.base_lr * cfg.diffusion_model.training.train_batch_size
                if optim_cfg.scale_lr
                else optim_cfg.base_lr
            ),
        )

        self.lr_sched = get_lr_sched(
            self.opt, cfg.diffusion_model.training.lr_warmup_steps
        )

        # Data loaders

        c = cfg.diffusion_model.training
        seq_length = cfg.diffusion_model.model.num_conditioning_steps + 1

        batch_sampler = BatchSampler(
            self.train_dataset,
            self._rank,
            self._world_size,
            c.train_batch_size,
            seq_length,
            guarantee_full_seqs=cfg.static_dataset.guarantee_full_seqs,
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
            c.eval_batch_size,
            seq_length,
            cfg.evaluation.subsample_rate,
        )

        # Training state (things to be saved/restored)
        self.epoch = 0
        self.global_step = 0

        self.diffusion = create_diffusion(
            timestep_respacing=str(cfg.diffusion.num_sampling_steps),
            learn_sigma=cfg.diffusion.learn_sigma,
        )  # default: 1000 steps, linear noise schedule
        self._setup_inference(cfg)

    def _setup_inference(self, cfg: DictConfig) -> None:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(
            self._device
        )
        vae.decoder.load_state_dict(
            torch.load(
                cfg.inference.vae_path, weights_only=True, map_location=self._device
            )
        )
        vae.eval()
        episode_path = Path(cfg.inference.episode_path)
        episode = Episode.load(episode_path)
        episode.obs = prepare_image_obs(episode.obs, cfg.static_dataset.img_resolution)
        self.inference_episode = episode

        self.trajectory_evaluator = TrajectoryEvaluator(
            diffusion=self.diffusion,
            vae=vae,
            num_seed_steps=(
                cfg.diffusion_model.model.num_conditioning_steps
                if cfg.static_dataset.guarantee_full_seqs
                else cfg.inference.num_seed_steps
            ),
            num_conditioning_steps=cfg.diffusion_model.model.num_conditioning_steps,
            sampling_algorithm=cfg.inference.sampling_algorithm,
            vae_batch_size=cfg.inference.vae_batch_size,
            device=self._device,
        )

    def run(self) -> None:

        num_epochs = self._cfg.diffusion_model.training.num_epochs

        while self.epoch < num_epochs:
            self.epoch += 1
            start_time = time.time()

            if self._rank == 0:
                print(f"\nEpoch {self.epoch} / {num_epochs}\n")

            if self._cfg.training.should:
                self.train_diffusion_model()

            # Evaluation
            should_test = (
                self._rank == 0
                and self._cfg.evaluation.should
                and (self.epoch % self._cfg.evaluation.every == 0)
            )

            if should_test:
                self.test_diffusion_model()

            # Inference
            should_inference = (
                self._rank == 0
                and self._cfg.inference.should
                and (self.epoch % self._cfg.inference.every == 0)
            )

            if should_inference:
                self.inference_diffusion_model()

            # Logging
            if self._rank == 0:
                wandb_log(
                    {"duration": (time.time() - start_time) / 3600},
                    self.epoch,
                    self.global_step,
                )

            # Checkpointing
            self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

            # if not training, no need to repeatedly eval or inference
            if not self._cfg.training.should:
                break

    def train_diffusion_model(self):
        self.diffusion_model.train()
        self.diffusion_model.zero_grad()
        cfg = self._cfg.diffusion_model.training
        num_steps = cfg.steps_per_epoch
        model = self._diffusion_model
        opt = self.opt
        lr_sched = self.lr_sched
        data_loader = self._data_loader_train

        opt.zero_grad()
        data_iterator = iter(data_loader)

        for _ in trange(num_steps, desc=f"Training", disable=self._rank > 0):
            self.global_step = self.global_step + 1
            batch = next(data_iterator).to(self._device)
            loss = self.call_model(model, batch)
            loss.backward()
            to_log = {"loss": loss.item()}

            opt.step()
            opt.zero_grad()

            if lr_sched is not None:
                to_log["lr"] = lr_sched.get_last_lr()[0]
                lr_sched.step()

            to_log = {f"train/{k}": v for k, v in to_log.items()}
            wandb_log(to_log, self.epoch, self.global_step)

    @torch.no_grad()
    def test_diffusion_model(self):
        self.diffusion_model.eval()
        model = self.diffusion_model
        data_loader = self._data_loader_test
        eval_loss = 0.0
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = batch.to(self._device)
            loss = self.call_model(model, batch)
            eval_loss += loss.item()

        eval_loss = eval_loss / len(data_loader)
        to_log = {"loss": eval_loss}

        to_log = {f"test/{k}": v for k, v in to_log.items()}
        wandb_log(to_log, self.epoch, self.global_step)

    @torch.no_grad()
    def inference_diffusion_model(self):
        self.diffusion_model.eval()
        for generation_mode in self._cfg.inference.generation_mode:
            generated_trajectory = self.trajectory_evaluator.evaluate_episode(
                self.diffusion_model, self.inference_episode, generation_mode
            )
            video = wandb.Video(generated_trajectory, fps=self._cfg.env.fps)
            wandb_log(
                {f"inference/{generation_mode}": video},
                self.epoch,
                self.global_step,
            )
        ground_truth_trajectory = to_numpy_video(self.inference_episode.obs)
        video = wandb.Video(ground_truth_trajectory, fps=self._cfg.env.fps)
        wandb_log(
            {f"inference/ground_truth": video},
            self.epoch,
            self.global_step,
        )

    def save_checkpoint(self) -> None:
        if self._rank == 0:
            self._keep_model_copies(self.diffusion_model.state_dict(), self.epoch)

    def call_model(self, model, batch):
        obs, act = batch.obs, batch.act
        n = obs.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, (n,), device=self._device)
        prev_obs = obs[:, :-1]
        prev_act = act[:, :-1]
        model_kwargs = dict(prev_obs=prev_obs, prev_act=prev_act)
        current_obs = obs[:, -1]
        loss_dict = self.diffusion.training_losses(model, current_obs, t, model_kwargs)
        loss = loss_dict["loss"].mean()
        return loss
