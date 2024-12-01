import argparse
import json
import os
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from einops import rearrange
from torchvision import transforms
from tqdm import tqdm

from src.data.dataset import Dataset


# Main function
def preprocess_data_with_vae(dataset, save_path, vae, resolution, batch_size):
    os.makedirs(save_path, exist_ok=True)

    for episode_id in tqdm(range(dataset.num_episodes)):
        episode = dataset.load_episode(episode_id)
        obs = episode.obs

        latents = []
        for i in range(0, len(obs), batch_size):
            batch_obs = obs[i : i + batch_size].to(vae.device)
            batch_obs = rearrange(batch_obs, "n h w c-> n c h w")
            batch_obs = batch_obs.float() / 255.0
            size = min(batch_obs.shape[-2], batch_obs.shape[-2])  # Crop to square
            batch_obs = transforms.functional.center_crop(batch_obs, size)
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        resolution, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

            batch_obs = transform(batch_obs)
            latent = vae.encode(batch_obs).latent_dist.sample()
            latents.append(latent)
        latents = torch.cat(latents, dim=0)
        episode.obs = latents
        episode.save(save_path / f"episode_{episode_id}.pt")
    with open(save_path / "episodes_info.json", "w") as f:
        json.dump(dataset.episodes_info, f)


if __name__ == "__main__":

    # Configuration
    parser = argparse.ArgumentParser(description="Preprocess episodes with VAE")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/scratch/gs4288/shared/diffusion_for_simulation/data/doom/original",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/scratch/gs4288/shared/diffusion_for_simulation/data/doom/latent",
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for computation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for inference"
    )
    args = parser.parse_args()
    data_path, save_path = Path(args.data_path), Path(args.save_path)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    vae.eval()
    vae.to(args.device)
    for dataset_type in ["train", "test"]:

        dataset = Dataset(
            data_path / "train",
        )
        print(f"Preprocessing {dataset_type} data...")
        with torch.no_grad():
            preprocess_data_with_vae(
                dataset, save_path / dataset_type, vae, args.resolution, args.batch_size
            )