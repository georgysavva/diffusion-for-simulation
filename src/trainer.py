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

from data import (
    BatchSampler,
    CSGOHdf5Dataset,
    Dataset,
    DatasetTraverser,
    collate_segments_to_batch,
)
from utils import (
    Logs,
    StateDictMixin,
    broadcast_if_needed,
    build_ddp_wrapper,
    configure_opt,
    count_parameters,
    get_lr_sched,
    keep_model_copies_every,
    process_confusion_matrices_if_any_and_compute_classification_metrics,
    save_info_for_import_script,
    save_with_backup,
    set_seed,
    try_until_no_except,
    wandb_log,
)


class Trainer(StateDictMixin):
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
            try_until_no_except(
                partial(
                    wandb.init,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    reinit=True,
                    resume=True,
                    **cfg.wandb,
                )
            )

        # Checkpointing
        self._path_ckpt_dir = Path("checkpoints")
        self._path_state_ckpt = self._path_ckpt_dir / "state.pt"
        self._keep_model_copies = partial(
            keep_model_copies_every,
            every=cfg.checkpointing.save_diffusion_model_every,
            path_ckpt_dir=self._path_ckpt_dir,
            num_to_keep=cfg.checkpointing.num_to_keep,
        )
        self._save_info_for_import_script = partial(
            save_info_for_import_script,
            run_name=cfg.wandb.name,
            path_ckpt_dir=self._path_ckpt_dir,
        )

        # First time, init files hierarchy
        if not cfg.common.resume and self._rank == 0:
            self._path_ckpt_dir.mkdir(exist_ok=False, parents=False)
            path_config = Path("config") / "trainer.yaml"
            path_config.parent.mkdir(exist_ok=False, parents=False)
            shutil.move(".hydra/config.yaml", path_config)
            wandb.save(str(path_config))
            shutil.copytree(src=root_dir / "src", dst="./src")
            shutil.copytree(src=root_dir / "scripts", dst="./scripts")

        assert (
            cfg.env.path_data_low_res is not None
            and cfg.env.path_data_full_res is not None
        ), "Make sure to download CSGO data and set the relevant paths in cfg.env"
        dataset_full_res = CSGOHdf5Dataset(Path(cfg.env.path_data_full_res))

        num_workers = cfg.training.num_workers_data_loaders
        use_manager = cfg.training.cache_in_ram and (num_workers > 0)
        p = Path(cfg.static_dataset.path)
        self.train_dataset = Dataset(
            p / "train",
            dataset_full_res,
            "train_dataset",
            cfg.training.cache_in_ram,
            use_manager,
        )
        self.test_dataset = Dataset(
            p / "test", dataset_full_res, "test_dataset", cache_in_ram=True
        )
        self.train_dataset.load_from_default_path()
        self.test_dataset.load_from_default_path()

        # Create models
        self.diffusion_model = instantiate(cfg.diffusion_model.model).to(self._device)
        self._diffusion_model = (
            build_ddp_wrapper(**self.diffusion_model._modules)
            if dist.is_initialized()
            else self.diffusion_model
        )

        if cfg.initialization.path_to_ckpt is not None:
            sd = torch.load(
                Path(cfg.initialization.path_to_ckp),
                map_location=self.diffusion_model.device,
            )
            self.diffusion_model.load_state_dict(sd)

        ######################################################

        # Optimizers and LR schedulers

        self.opt = configure_opt(
            self.diffusion_model, **cfg.diffusion_model.training.optimizer
        )

        self.lr_sched = get_lr_sched(
            self.opt, cfg.diffusion_model.training.lr_warmup_steps
        )

        # Data loaders

        make_data_loader = partial(
            DataLoader,
            dataset=self.train_dataset,
            collate_fn=collate_segments_to_batch,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=self._use_cuda,
            pin_memory_device=str(self._device) if self._use_cuda else "",
        )

        make_batch_sampler = partial(
            BatchSampler, self.train_dataset, self._rank, self._world_size
        )

        c = cfg.diffusion_model.training
        seq_length = (
            cfg.diffusion_model.model_cfg.inner_model.num_steps_conditioning
            + 1
            + c.num_autoregressive_steps
        )
        bs = make_batch_sampler(c.batch_size, seq_length, sample_weights=None)

        self._data_loader_train = make_data_loader(batch_sampler=bs)
        self._data_loader_test = DatasetTraverser(
            self.test_dataset, c.batch_size, seq_length
        )

        # Training state (things to be saved/restored)
        self.epoch = 0
        self.num_episodes_test = 0
        self.num_batch_train = 0
        self.num_batch_test = 0

        if cfg.common.resume:
            self.load_state_checkpoint()
        else:
            self.save_checkpoint()

        if self._rank == 0:
            print(f"{count_parameters(self.diffusion_model)} parameters")
            print(self.train_dataset)
            print(self.test_dataset)

    def run(self) -> None:
        to_log = []

        num_epochs = self._cfg.training.num_final_epochs

        while self.epoch < num_epochs:
            self.epoch += 1
            start_time = time.time()

            if self._rank == 0:
                print(f"\nEpoch {self.epoch} / {num_epochs}\n")

            (sd_train_dataset,) = broadcast_if_needed(
                self.train_dataset.state_dict()
            )  # update dataset for ranks > 0
            self.train_dataset.load_state_dict(sd_train_dataset)

            if self._cfg.training.should:
                to_log += self.train_diffusion_model()

            # Evaluation
            should_test = (
                self._rank == 0
                and self._cfg.evaluation.should
                and (self.epoch % self._cfg.evaluation.every == 0)
            )

            if should_test:
                to_log += self.test_diffusion_model()

            # Logging
            to_log.append({"duration": (time.time() - start_time) / 3600})
            if self._rank == 0:
                wandb_log(to_log, self.epoch)
            to_log = []

            # Checkpointing
            self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

    def train_diffusion_model(self) -> Logs:
        self.diffusion_model.train()
        self.diffusion_model.zero_grad()
        to_log = []
        cfg = self._cfg.diffusion_model.training
        if self.epoch > cfg.start_after_epochs:
            steps = cfg.steps_first_epoch if self.epoch == 1 else cfg.steps_per_epoch
            to_log += self.train_component(steps)
        return to_log

    @torch.no_grad()
    def test_diffusion_model(self) -> Logs:
        self.diffusion_model.eval()
        to_log = []
        cfg = self._cfg.diffusion_model.training
        if self.epoch > cfg.start_after_epochs:
            to_log += self.test_component()
        return to_log

    def train_component(self, steps: int) -> Logs:
        cfg = self._cfg.diffusion_model.training
        model = self._diffusion_model
        opt = self.opt
        lr_sched = self.lr_sched
        data_loader = self._data_loader_train

        model.train()
        opt.zero_grad()
        data_iterator = iter(data_loader)
        to_log = []

        num_steps = cfg.grad_acc_steps * steps

        for i in trange(num_steps, desc=f"Training", disable=self._rank > 0):
            batch = next(data_iterator).to(self._device)
            loss, metrics = model(batch) if batch is not None else model()
            loss.backward()

            num_batch = self.num_batch_train
            metrics[f"num_batch_train"] = num_batch
            self.num_batch_train = num_batch + 1

            if (i + 1) % cfg.grad_acc_steps == 0:
                if cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.max_grad_norm
                    )
                    metrics["grad_norm_before_clip"] = grad_norm

                opt.step()
                opt.zero_grad()

                if lr_sched is not None:
                    metrics["lr"] = lr_sched.get_last_lr()[0]
                    lr_sched.step()

            to_log.append(metrics)

        process_confusion_matrices_if_any_and_compute_classification_metrics(to_log)
        to_log = [{f"train/{k}": v for k, v in d.items()} for d in to_log]
        return to_log

    @torch.no_grad()
    def test_component(self) -> Logs:
        model = self.diffusion_model
        data_loader = self._data_loader_test
        model.eval()
        to_log = []
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = batch.to(self._device)
            _, metrics = model(batch)
            num_batch = self.num_batch_test
            metrics["num_batch_test"] = num_batch
            self.num_batch_test = num_batch + 1
            to_log.append(metrics)

        process_confusion_matrices_if_any_and_compute_classification_metrics(to_log)
        to_log = [{f"test/{k}": v for k, v in d.items()} for d in to_log]
        return to_log

    def load_state_checkpoint(self) -> None:
        self.load_state_dict(
            torch.load(self._path_state_ckpt, map_location=self._device)
        )

    def save_checkpoint(self) -> None:
        if self._rank == 0:
            save_with_backup(self.state_dict(), self._path_state_ckpt)
            self.train_dataset.save_to_default_path()
            self.test_dataset.save_to_default_path()
            self._keep_model_copies(self.diffusion_model.state_dict(), self.epoch)
            self._save_info_for_import_script(self.epoch)
