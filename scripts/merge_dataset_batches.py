import argparse
import json
from pathlib import Path


def main(args):
    merged_episodes_info = {"episodes": []}
    current_merged_episode_id = 0
    dest_dir = Path(args.dest_dir)
    for source_dir in args.source_dirs:
        source_dir = Path(source_dir)
        with open(source_dir / "episodes_info.json", "r") as f:
            episodes_info = json.load(f)
        for episode in episodes_info["episodes"]:
            episode_id = episode["episode_id"]
            new_episode_id = current_merged_episode_id
            source_file = source_dir / f"episode_{episode_id}.pt"
            dest_file = dest_dir / f"episode_{new_episode_id}.pt"
            print(f"Moving episode {source_file} to {dest_file}")
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            source_file.rename(dest_file)
            episode["episode_id"] = new_episode_id
            merged_episodes_info["episodes"].append(episode)
            current_merged_episode_id += 1
    merged_episodes_info["episodes_num"] = len(merged_episodes_info["episodes"])
    with open(dest_dir / "episodes_info.json", "w") as f:
        json.dump(merged_episodes_info, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process episodes to create PyTorch data files."
    )

    parser.add_argument(
        "--dest_dir", required=True, help="Path to the destination directory."
    )
    parser.add_argument(
        "--source_dirs", required=True, nargs="+", help="List of source directories."
    )
    args = parser.parse_args()
    main(args)
