import json
import os
import random
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from torchvision.datasets.utils import download_url

import wandb


def prepare_image_obs(obs, resolution):
    obs = rearrange(obs, "n h w c-> n c h w")
    size = min(obs.shape[-2], obs.shape[-2])  # Crop to square
    obs = transforms.functional.center_crop(obs, size)
    transform = transforms.Resize(
        resolution, interpolation=transforms.InterpolationMode.BILINEAR
    )

    obs = transform(obs)
    return obs


def normalize_img(img: torch.Tensor) -> torch.Tensor:
    img = img.float() / 255.0
    transform = transforms.Normalize([0.5], [0.5])
    img = transform(img)
    return img


def to_numpy_video(imgs: torch.Tensor) -> np.ndarray:
    imgs = rearrange(imgs, "n c h w-> n h w c")
    return imgs.cpu().numpy()


def denormalize_img(img: torch.Tensor) -> torch.Tensor:
    img = img * 0.5 + 0.5
    img = (img * 255.0).clamp(0, 255).byte()
    return img


def build_ddp_wrapper(**modules_dict: Dict[str, nn.Module]) -> Namespace:
    return Namespace(**{name: DDP(module) for name, module in modules_dict.items()})


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_lr_sched(opt: torch.optim.Optimizer, num_warmup_steps: int) -> LambdaLR:
    def lr_lambda(current_step: int):
        return (
            1
            if current_step >= num_warmup_steps
            else current_step / max(1, num_warmup_steps)
        )

    return LambdaLR(opt, lr_lambda, last_epoch=-1)


def get_path_diffusion_model_ckpt(
    path_ckpt_dir: Union[str, Path], epoch: int, num_zeros: int = 5
) -> Path:
    d = Path(path_ckpt_dir) / "diffusion_model_versions"
    if epoch >= 0:
        return d / f"diffusion_model_epoch_{epoch:0{num_zeros}d}.pt"
    else:
        all_ = sorted(list(d.iterdir()))
        assert len(all_) >= -epoch
        return all_[epoch]


def keep_model_copies_every(
    model_sd: Dict[str, Any],
    epoch: int,
    path_ckpt_dir: Path,
    every: int,
    num_to_keep: Optional[int],
) -> None:
    assert every > 0
    assert num_to_keep is None or num_to_keep > 0
    get_path = partial(get_path_diffusion_model_ckpt, path_ckpt_dir)
    get_path(0).parent.mkdir(parents=False, exist_ok=True)

    # Save diffusion_model
    save_with_backup(model_sd, get_path(epoch))

    # Clean oldest
    if (num_to_keep is not None) and (epoch % every == 0):
        get_path(max(0, epoch - num_to_keep * every)).unlink(missing_ok=True)

    # Clean previous
    if (epoch - 1) % every != 0:
        get_path(max(0, epoch - 1)).unlink(missing_ok=True)


def save_with_backup(obj: Any, path: Path):
    bk = path.with_suffix(".bk")
    if path.is_file():
        path.rename(bk)
    torch.save(obj, path)
    bk.unlink(missing_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def wandb_log(log: dict[str, float], epoch: int, global_step: int) -> None:
    wandb.log({"epoch": epoch, **log}, step=global_step)


def download_model_weights(url: str, save_path: str, device: torch.device):
    """
    Downloads a pre-trained model from the web.
    """
    model_name = os.path.basename(url)
    local_path = f"{save_path}/{model_name}"
    if not os.path.isfile(local_path):
        os.makedirs(save_path, exist_ok=True)
        download_url(url, save_path, filename=model_name)
    model = torch.load(local_path, map_location=device, weights_only=True)
    return model


def save_np_video(frames: np.ndarray, path: str, fps: int) -> None:
    """
    Saves a numpy array of frames to disk as a playable video.

    Args:
        frames (np.ndarray): Array of frames with shape (num_frames, height, width, channels).
        path (str): Path to save the video file.
        fps (int): Frames per second for the video.
    """
    height, width = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
    video_writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video_writer.release()
