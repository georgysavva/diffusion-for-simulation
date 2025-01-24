#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

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
        "--dest_dir", required=True, help="Path to the destination directory."
    )
    parser.add_argument(
        "--frame_skip",
        default=0,
        type=int,
        help="How many frames to skip between each frame.",
    )
    parser.add_argument(
        "--observations_only",
        default=0,
        type=bool,
        help="Whether to only save observations.",
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    dest_dir = args.dest_dir
    frame_skip = args.frame_skip
    observations_only = args.observations_only

    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Get list of subdirectories in the source directory
    # We assume each subdirectory is one episode
    subdirectories = sorted(
        [
            os.path.join(source_dir, d)
            for d in os.listdir(source_dir)
            if os.path.isdir(os.path.join(source_dir, d))
        ]
    )

    episodes_info = []

    for episode_path in subdirectories:
        # Paths to files
        episode_id = int(os.path.basename(episode_path).split("_")[-1])
        frames_path = os.path.join(episode_path, "frames.png")
        actions_path = os.path.join(episode_path, "actions.txt")

        # 1) Read and split frames.png into 256x256 frames
        with Image.open(frames_path) as img:
            # Convert image to RGB if not already
            img = img.convert("RGB")
            width, height = img.size

            # Expect height = 256, width = 256 * number_of_frames

            assert height == FRAME_SIZE and (width % FRAME_SIZE) == 0, (
                f"frames.png in '{episode_path}' has unexpected dimensions: "
                f"{width}x{height}"
            )

            num_frames = width // FRAME_SIZE

            # Split the image horizontally into frames
            frame_list = []
            for i in range(0, num_frames, frame_skip + 1):
                left = i * FRAME_SIZE
                upper = 0
                right = left + FRAME_SIZE
                lower = FRAME_SIZE

                frame = img.crop((left, upper, right, lower))  # 256x256
                # Convert frame to a torch tensor
                # PIL image => (H x W x C)
                np_image = np.array(frame)
                frame_tensor = torch.from_numpy(np_image)
                frame_list.append(frame_tensor)

            # Stack along dimension 0 to get a shape of (num_frames, 256, 256, 3)
            frames_tensor = torch.stack(frame_list, dim=0)
        if not observations_only:
            # 2) Read actions.txt
            with open(actions_path, "r") as f:
                lines = f.readlines()

            # Each line has 5 numbers separated by spaces
            actions = []
            for line in lines:
                values = line.strip().split()
                assert (
                    len(values) == 5
                ), f"Malformed action line in {actions_path}: {line.strip()}"
                # Convert each to float (or int if you prefer)
                action_values = list(map(int, values))
                actions.append(action_values)

            # Convert actions to a Tensor: shape (num_frames, 5)
            actions_tensor = torch.tensor(actions, dtype=torch.int32)

            # Ensure the number of frames matches the number of actions
            assert frames_tensor.shape[0] == actions_tensor.shape[0], (
                f"Mismatch in frames ({frames_tensor.shape[0]}) and actions "
                f"({actions_tensor.shape[0]}) in '{episode_path}'."
            )

            # 3) Combine data into a single dict
            episode_data = {
                "observations": frames_tensor,  # (num_frames, 256, 256, 3)
                "actions": actions_tensor,  # (num_frames, 5)
            }
        else:
            episode_data = frames_tensor

        # 4) Save
        save_path = os.path.join(dest_dir, f"episode_{episode_id}.pt")
        torch.save(episode_data, save_path)
        print(f"Saved episode data to: {save_path}")
        if not observations_only:
            length = frames_tensor.shape[0]
            episodes_info.append(
                {
                    "episode_id": episode_id,
                    "length": length,
                    "source": "exploration_agent",
                }
            )
    if not observations_only:
        # Sort episodes_info by episode_id
        episodes_info.sort(key=lambda x: x["episode_id"])
        info_path = os.path.join(dest_dir, "episodes_info.json")
        with open(info_path, "w") as f:
            data = {"episodes_num": len(episodes_info), "episodes": episodes_info}
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
