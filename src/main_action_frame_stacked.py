import os
from pathlib import Path
from typing import List, Union

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group, init_process_group

from src.trainer import Trainer

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="../config", config_name="trainer_VDA_action_frame_stacked", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_visible_cuda_devices(cfg.common.devices)
    world_size = torch.cuda.device_count()
    root_dir = Path(hydra.utils.get_original_cwd())
    if world_size < 2:
        run(cfg, root_dir)
    else:
        mp.spawn(main_ddp, args=(world_size, cfg, root_dir), nprocs=world_size)


def main_ddp(rank: int, world_size: int, cfg: DictConfig, root_dir: Path) -> None:
    setup_ddp(rank, world_size)
    run(cfg, root_dir)
    destroy_process_group()


def run(cfg: DictConfig, root_dir: Path) -> None:
    trainer = Trainer(cfg, root_dir)
    trainer.run()


def setup_ddp(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6006"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def setup_visible_cuda_devices(devices: Union[str, int, List[int]]) -> None:
    if isinstance(devices, str):
        if devices == "cpu":
            devices = []
        else:
            assert devices == "all"
            return
    elif isinstance(devices, int):
        devices = [devices]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))


if __name__ == "__main__":
    main()
