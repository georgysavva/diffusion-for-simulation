import argparse
import json
import os
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from tqdm import tqdm

from src.data.episode import Episode
from src.utils import normalize_img, prepare_image_obs


# Main function
def preprocess_data_with_vae(load_path, save_path, vae, resolution, batch_size):
    os.makedirs(save_path, exist_ok=True)

    episode_files = sorted(load_path.glob("episode_*.pt"))
    for episode_file in tqdm(episode_files):
        episode = Episode.load(episode_file)
        obs = episode.obs

        latents = []
        for i in range(0, len(obs), batch_size):
            batch_obs = obs[i : i + batch_size].to(vae.device)
            batch_obs = prepare_image_obs(batch_obs, resolution)
            batch_obs = normalize_img(batch_obs)
            latent = vae.encode(batch_obs).latent_dist.sample()
            latents.append(latent)
        latents = torch.cat(latents, dim=0)
        episode.obs = latents
        episode.save(save_path / episode_file.name)
    with open(save_path / "episodes_info.json", "w") as f:
        with open (load_path / "episodes_info.json", "r") as load_f:
            info = json.load(load_f)
            json.dump(info, f, indent=4)


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
    parser.add_argument("--device", type=str, help="Device to use for computation")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for inference"
    )

    args = parser.parse_args()
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    vae.eval()
    vae.to(device)
    for dataset_type in ["test", "train"]:
        print(f"Preprocessing {dataset_type} data...")
        data_path, save_path = Path(args.data_path), Path(args.save_path)

        with torch.no_grad():
            preprocess_data_with_vae(
                data_path / dataset_type, save_path / dataset_type, vae,args.resolution, args.batch_size
            )
