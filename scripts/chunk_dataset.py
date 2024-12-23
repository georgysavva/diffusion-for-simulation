import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from src.data.episode import Episode


def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    load_path = Path(args.dataset_path)
    save_path = Path(args.save_path)
    episode_files = sorted(load_path.glob("episode_*.pt"))
    for episode_file in tqdm(episode_files):
        episode = Episode.load(episode_file)
        episode_path = save_path / f"episode_{episode.episode_id}"
        os.makedirs(episode_path, exist_ok=True)
        obs = episode.obs
        act = episode.act
        for chunk_start in range(0, len(obs), args.chunk_size):
            chunk_obs = obs[chunk_start : chunk_start + args.chunk_size]
            chunk_act = act[chunk_start : chunk_start + args.chunk_size]
            chunk = {
                "observations": chunk_obs,
                "actions": chunk_act,
            }
            chunk_id = chunk_start // args.chunk_size
            torch.save(
                chunk,
                episode_path / f"chunk_{chunk_id:05d}.pt",
            )
    with open(save_path / "episodes_info.json", "w") as f:
        with open(load_path / "episodes_info.json", "r") as load_f:
            info = json.load(load_f)
            info["chunk_size"] = args.chunk_size
            json.dump(info, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset episodes into chunks")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/scratch/gs4288/shared/diffusion_for_simulation/data/doom/latent/train",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/scratch/gs4288/shared/diffusion_for_simulation/data/doom/latent_chunked/train",
        help="Path where to save the chunked dataset",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="Chunk size",
    )
    args = parser.parse_args()
    main(args)
