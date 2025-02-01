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
import torchvision.transforms as T
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
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


def denormalize_img(img: torch.Tensor) -> torch.Tensor:
    img = img * 0.5 + 0.5
    img = (img * 255.0).clamp(0, 255).byte()
    return img


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_warmup_lr_sched(opt: torch.optim.Optimizer, num_warmup_steps: int) -> LambdaLR:
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


def wandb_log(log: dict[str, Any], epoch: int) -> None:
    wandb.log(log, step=epoch)


def save_as_video(frames, path: str | Path, fps: int) -> None:
    """
    Saves a numpy array of frames to disk as a playable video.

    Args:
        frames (torch.Tensor): Array of frames with shape (num_frames, height, width, channels).
        path (str): Path to save the video file.
        fps (int): Frames per second for the video.
    """
    frames = rearrange(frames, "n c h w-> n h w c")
    frames = frames.cpu().numpy()
    height, width = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
    video_writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video_writer.release()


def to_concatenated_images_with_text(images, words, margin=10, font_size=24):
    """
    Concatenate images in a row with a margin and add words below each image except the last one.

    Args:
    - images (torch.Tensor): Tensor of shape (N, 3, 256, 256) in range [0, 255], dtype uint8.
    - words (list of str): List of N words for labeling the images.
    - margin (int): Margin between images in pixels.
    - output_path (str): File path to save the concatenated image.
    - font_size (int): Font size for the text.
    """
    # Convert torch tensor to PIL images
    pil_images = [T.ToPILImage()(img) for img in images]

    # Load a default font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate the maximum text height for words
    temp_image = Image.new("RGB", (1, 1), "white")
    temp_draw = ImageDraw.Draw(temp_image)
    text_heights = [
        temp_draw.textbbox((0, 0), word, font=font)[3]
        - temp_draw.textbbox((0, 0), word, font=font)[1]
        for word in words
    ]
    max_text_height = max(text_heights) if text_heights else 0

    # Determine dimensions
    N, _, height, width = images.shape
    total_width = N * width + (N - 1) * margin
    output_height = height + max_text_height + margin
    output_image = Image.new("RGB", (total_width, output_height), "white")
    draw = ImageDraw.Draw(output_image)

    # Paste images and add text
    x_offset = 0
    for i, img in enumerate(pil_images):
        output_image.paste(img, (x_offset, 0))
        text_bbox = draw.textbbox((0, 0), words[i], font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x_offset + (width - text_width) // 2
        text_y = height + (margin // 2)
        draw.text((text_x, text_y), words[i], fill="black", font=font)
        x_offset += width + margin
    return output_image
