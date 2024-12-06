import argparse
import os

import torch


def main(args):
    num_frames = args.num_frames
    episode_path = args.episode_path

    # Load the episode
    episode = torch.load(episode_path, weights_only=True, map_location="cpu")

    # Slice the tensors
    sliced_episode = {key: tensor[:num_frames] for key, tensor in episode.items()}

    # Create the new file name
    file_name, file_extension = os.path.splitext(os.path.basename(episode_path))
    dir_path = os.path.dirname(episode_path)
    new_file_path = os.path.join(
        dir_path, f"{file_name}_{num_frames}_frames{file_extension}"
    )

    torch.save(sliced_episode, new_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut episode script")
    parser.add_argument(
        "--episode_path", type=str, required=True, help="Path to the episode"
    )
    parser.add_argument(
        "--num_frames", type=int, required=True, help="Number of frames"
    )
    args = parser.parse_args()
    main(args)
