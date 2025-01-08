import os
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group, init_process_group

from src.trainer import Trainer

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="../config", config_name="trainer", version_base="1.3")
def main(cfg: DictConfig) -> None:
    world_size = torch.cuda.device_count()
    root_dir = Path(hydra.utils.get_original_cwd())
    if world_size < 2:
        run(cfg, root_dir)
    else:

        hydra_cfg = HydraConfig.get()
        mp.spawn(
            main_ddp, args=(world_size, cfg, hydra_cfg, root_dir), nprocs=world_size
        )


def main_ddp(
    rank: int, world_size: int, cfg: DictConfig, hydra_cfg: HydraConf, root_dir: Path
) -> None:
    setup_ddp(rank, world_size)
    hydra.initialize(version_base=None)
    HydraConfig.instance().set_config(OmegaConf.create({"hydra": hydra_cfg}))
    run(cfg, root_dir)
    destroy_process_group()


def run(cfg: DictConfig, root_dir: Path) -> None:
    trainer = Trainer(cfg, root_dir)
    trainer.run()


def setup_ddp(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6006"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


if __name__ == "__main__":
    main()
