#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import tqdm
from PIL import Image
from tqdm import tqdm

FRAME_SIZE = 256


def main():
    parser = argparse.ArgumentParser(
        description="Process episodes to create PyTorch data files."
    )
    parser.add_argument(
        "--source_dir",
        required=True,
        help="Path to the source directory containing episode subdirectories.",
    )
    parser.add_argument(
        "--save_dir", required=True, help="Path to the destination directory."
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    save_dir = args.save_dir

    # Create the destination directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episode_files = sorted(source_dir.glob("episode_*.pt"))
    episodes_info = []

    for episode_path in tqdm(episode_files):
        # Paths to files
        episode_id = int(episode_path.stem.split("_")[-1])
        episode_data = torch.load(episode_path, weights_only=True, map_location=device)
        obs = episode_data["observations"]
        act = episode_data["actions"]
        obs = obs[:-1]
        act = act[1:]
        episode_data = {
            "observations": obs,
            "actions": act,
        }

        save_path = os.path.join(save_dir, f"episode_{episode_id}.pt")
        torch.save(episode_data, save_path)
        print(f"Saved episode data to: {save_path}")
        assert len(obs) == len(act)
        length = len(act)
        episodes_info.append(
            {
                "episode_id": episode_id,
                "length": length,
                "source": "exploration_agent",
            }
        )
    # Sort episodes_info by episode_id
    episodes_info.sort(key=lambda x: x["episode_id"])
    info_path = os.path.join(save_dir, "episodes_info.json")
    with open(info_path, "w") as f:
        data = {"episodes_num": len(episodes_info), "episodes": episodes_info}
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
