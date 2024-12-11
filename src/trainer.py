import os
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import wandb
from src.data import (
    BatchSampler,
    Dataset,
    TestDatasetTraverser,
    TestDatasetTraverserNew,
    collate_segments_to_batch,
)
from src.diffusion import create_diffusion
from src.utils import (
    build_ddp_wrapper,
    count_parameters,
    get_lr_sched,
    keep_model_copies_every,
    set_seed,
    wandb_log,
)
from src.models.DiT import DiT
from src.models.VDA import VDA

from diffusers import AutoencoderKL
from torchvision.utils import make_grid
from PIL import Image


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
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu", self._rank
        )
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
                project=self._cfg.wandb.project,
                entity=self._cfg.wandb.entity,
                name=self._cfg.wandb.name,
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
            cache_in_ram=False,
        )
        self.test_dataset = Dataset(p / "test", cache_in_ram=True)

        # Create models
        if self._rank == 0:
            print("Instantiating model")
        self.diffusion_model = instantiate(
            cfg.diffusion_model.model, num_actions=cfg.env.num_actions
        ).to(self._device)
        if self._rank == 0:
            print(f"{count_parameters(self.diffusion_model)} parameters")
        self._diffusion_model = (
            build_ddp_wrapper(**self.diffusion_model._modules)
            if dist.is_initialized()
            else self.diffusion_model
        )
        assert (
            cfg.pretrained_weights is None or cfg.initialization.path_to_ckpt is None
        ), "Only one of pretrained_weights or path_to_ckpt should be provided"
        if cfg.pretrained_weights is not None:
            self.diffusion_model.load_pretrained_weights(cfg.pretrained_weights)

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
            lr=optim_cfg.base_lr * cfg.diffusion_model.training.train_batch_size if optim_cfg.scale_lr else optim_cfg.base_lr,
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

        if cfg.evaluation.dataloader_name == 'TestDatasetTraverser':
            self._data_loader_test = TestDatasetTraverser(
                self.test_dataset, c.train_batch_size, seq_length
            )
        elif cfg.evaluation.dataloader_name == 'TestDatasetTraverserNew':
            self._data_loader_test = TestDatasetTraverserNew(
                self.test_dataset, c.eval_batch_size, seq_length, cfg.evaluation.subsample_rate, cfg.evaluation.max_num_episodes,
            )
        else:
            raise ValueError(f'{cfg.evaluation.dataloader_name} is not recognized')

        if cfg.inference.dataloader_name == 'TestDatasetTraverser':
            self._data_loader_inference = TestDatasetTraverser(
                self.test_dataset, c.train_batch_size, seq_length
            )
        elif cfg.inference.dataloader_name == 'TestDatasetTraverserNew':
            self._data_loader_inference = TestDatasetTraverserNew(
                self.test_dataset, c.eval_batch_size, seq_length, cfg.inference.subsample_rate, cfg.inference.max_num_episodes
            )
        else:
            raise ValueError(f'{cfg.inference.dataloader_name} is not recognized')

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self._device)
        vae.decoder.load_state_dict(torch.load(cfg.inference.vae_path, weights_only=True, map_location=self._device))
        vae.eval()
        self.vae = vae

        # Training state (things to be saved/restored)
        self.epoch = 0
        self.global_step = 0

        self.diffusion = create_diffusion(timestep_respacing="", learn_sigma=self._cfg.diffusion.learn_sigma)  # default: 1000 steps, linear noise schedule

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
            if self._rank == 0 and self._cfg.wandb.do_log:
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
            loss, _, _, _ = self.call_model(model, batch)
            loss.backward()
            to_log = {"loss": loss.item()}

            opt.step()
            opt.zero_grad()

            if lr_sched is not None:
                to_log["lr"] = lr_sched.get_last_lr()[0]
                lr_sched.step()

            to_log = {f"train/{k}": v for k, v in to_log.items()}
            if self._cfg.wandb.do_log:
                wandb_log(to_log, self.epoch, self.global_step)

    @torch.no_grad()
    def test_diffusion_model(self):
        self.diffusion_model.eval()
        model = self.diffusion_model
        data_loader = self._data_loader_test
        eval_loss, eval_mse, eval_mse_last_frame, eval_vb = 0.0, 0.0, 0.0, 0.0
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = batch.to(self._device)
            loss, mse, mse_last_frame, vb = self.call_model(model, batch)
            eval_loss += loss.item()
            eval_mse += mse.item()
            eval_mse_last_frame += mse_last_frame.item()
            eval_vb += vb.item() if vb is not None else 0.0

        to_log = {
            "loss": eval_loss / len(data_loader),
            'mse': eval_mse / len(data_loader),
            'mse_last_frame': eval_mse_last_frame / len(data_loader),
            'vb': eval_vb / len(data_loader),
        }
        to_log = {f"test/{k}": v for k, v in to_log.items()}
        if self._cfg.wandb.do_log:
            wandb_log(to_log, self.epoch, self.global_step)

    @torch.no_grad()
    def inference_diffusion_model(self):
        self.diffusion_model.eval()
        model = self.diffusion_model
        data_loader = self._data_loader_inference
        diffusion = create_diffusion(str(self._cfg.inference.num_sampling_steps), learn_sigma=self._cfg.diffusion.learn_sigma)

        all_decoded_samples = []
        all_prev_acts = []
        for batch in tqdm(data_loader, desc="Inferencing"):
            prev_act = batch.act[:, :-1].to(self._device)
            z = torch.randn_like(batch.obs, device=self._device)
            samples = diffusion.ddim_sample_loop(model.forward, batch.obs.shape, z, clip_denoised=False, model_kwargs={'prev_act': prev_act}, progress=True, device=self._device)
            # vae decode
            decoded_samples = []
            for frame_i in range(samples.shape[1]):
                decoded_sample = self.vae.decode(samples[:, frame_i] / 0.18215).sample.clamp(-1, 1)
                decoded_samples.append(decoded_sample)
            decoded_samples = torch.stack(decoded_samples, dim=1) # (N, seqlen, 3, 256, 256) in [-1, 1]
            decoded_samples = ((decoded_samples * 0.5 + 0.5) * 255.0).clamp(0, 255).byte()
            all_decoded_samples.append(decoded_samples.cpu())
            all_prev_acts.append(prev_act)
        all_decoded_samples = torch.cat(all_decoded_samples, dim=0)
        all_prev_acts = torch.cat(all_prev_acts, dim=0)

        sample_dir = os.path.join(self._cfg.common.run_dir, f'inference_epoch_{self.epoch}')
        os.makedirs(sample_dir, exist_ok=True)
        print('saving inference results to', sample_dir)
        for sample_i, (sample, acts) in enumerate(zip(all_decoded_samples, all_prev_acts)):
            grid = make_grid(sample, nrow=sample.shape[0], padding=2)
            image = Image.fromarray(grid.permute(1, 2, 0).cpu().numpy())
            image.save(os.path.join(sample_dir, f'{sample_i}.jpg'))

            if self._cfg.wandb.do_log and sample_i < self._cfg.inference.num_log_wandb:
                act_names = [["TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD", "MOVE_LEFT", "MOVE_RIGHT", "NOOP"][act] for act in acts]
                try:
                    wandb.log({f'inference_img_{sample_i}': wandb.Image(image, caption=', '.join(act_names)), "epoch": self.epoch}, step=self.global_step)
                except:
                    pass

        del all_decoded_samples

    def save_checkpoint(self) -> None:
        if self._rank == 0:
            self._keep_model_copies(self.diffusion_model.state_dict(), self.epoch)

    def call_model(self, model, batch):
        obs, act = batch.obs, batch.act
        n = obs.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, (n,), device=self._device)
        if isinstance(model, DiT):
            prev_obs = obs[:, :-1]
            prev_act = act[:, :-1]
            model_kwargs = dict(prev_obs=prev_obs, prev_act=prev_act)
            current_obs = obs[:, -1]
            loss_dict = self.diffusion.training_losses(model, current_obs, t, model_kwargs)
        elif isinstance(model, VDA):
            prev_act = act[:, :-1]
            model_kwargs = dict(prev_act=prev_act)
            loss_dict = self.diffusion.training_losses(model, obs, t, model_kwargs)
        else:
            raise ValueError(f'{type(model)} is not recognized')

        loss = loss_dict["loss"].mean()
        mse = loss_dict['mse'].mean()
        mse_last_frame = loss_dict['mse_last_frame'].mean()
        vb = loss_dict['vb'].mean() if 'vb' in loss_dict else None
        return loss, mse, mse_last_frame, vb
