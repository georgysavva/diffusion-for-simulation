import shutil
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from data import BatchSampler, Dataset, TestDatasetTraverser, collate_segments_to_batch
from diffusion import create_diffusion
from utils import (
    build_ddp_wrapper,
    count_parameters,
    download_model_weights,
    get_lr_sched,
    keep_model_copies_every,
    set_seed,
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
        if self._rank == 0:

            wandb.init(
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=True,
                **cfg.wandb,
            )

        # Checkpointing
        self._path_ckpt_dir = Path(cfg.checkpointing.save_path) / wandb.run.name
        self._keep_model_copies = partial(
            keep_model_copies_every,
            every=cfg.checkpointing.save_diffusion_model_every,
            path_ckpt_dir=self._path_ckpt_dir,
            num_to_keep=cfg.checkpointing.num_to_keep,
        )

        # First time, init files hierarchy
        if self._rank == 0:
            self._path_ckpt_dir.mkdir(exist_ok=False, parents=False)

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
            weights = download_model_weights(
                cfg.pretrained_weights.url,
                cfg.pretrained_weights.save_path,
                self._device,
            )
            self.diffusion_model.load_pretrained_weights(weights)

        if cfg.initialization.path_to_ckpt is not None:
            sd = torch.load(
                Path(cfg.initialization.path_to_ckp),
                map_location=self._device,
            )
            self.diffusion_model.load_state_dict(sd)

        ######################################################

        # Optimizers and LR schedulers

        self.opt = torch.optim.AdamW(
            self.diffusion_model.parameters(), **cfg.diffusion_model.training.optimizer
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
            c.batch_size,
            seq_length,
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
            self.test_dataset, c.batch_size, seq_length
        )

        # Training state (things to be saved/restored)
        self.epoch = 0
        self.global_step = 0

        self.diffusion = create_diffusion(
            timestep_respacing="",
            learn_sigma=False,
        )  # default: 1000 steps, linear noise schedule

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
        model.eval()
        eval_loss = 0.0
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = batch.to(self._device)
            loss = self.call_model(model, batch)
            eval_loss += loss.item()

        eval_loss = eval_loss / len(data_loader)
        to_log = {"loss": eval_loss}

        to_log = {f"test/{k}": v for k, v in to_log.items()}
        wandb_log(to_log, self.epoch, self.global_step)

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
