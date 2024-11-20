import argparse  # Add argparse import
import json
import os

import cv2
import gymnasium as gym
import torch
from tqdm import tqdm
from vizdoom import gymnasium_wrapper  # to register envs


# Collect episodes using a random policy
def collect_episodes(env, num_episodes, save_path):
    os.makedirs(save_path, exist_ok=True)
    video_path = os.path.join(save_path, "videos")
    os.makedirs(video_path, exist_ok=True)
    episodes_info = {"episodes_num": 0, "episodes": []}
    for episode in tqdm(range(num_episodes), desc="Sampling episodes"):
        episode_data = {"observations": [], "actions": [], "rewards": []}
        observation, _ = env.reset()
        done = False
        episode_reward = 0
        video_filename = os.path.join(video_path, f"episode_{episode}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'XVID' for .avi files
        out = cv2.VideoWriter(
            video_filename,
            fourcc,
            60,
            (observation["screen"].shape[1], observation["screen"].shape[0]),
        )
        while not done:
            # Random action
            action = env.action_space.sample()

            out.write(observation["screen"])

            # Step environment
            (
                next_observation,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            # Save episode data

            episode_data["observations"].append(torch.from_numpy(observation["screen"]))
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_reward += reward

            observation = next_observation
            done = terminated or truncated

        episode_data["observations"] = torch.stack(episode_data["observations"])
        episode_data["actions"] = torch.tensor(episode_data["actions"])
        episode_data["rewards"] = torch.tensor(episode_data["rewards"])
        # Save episode data to disk
        episode_file = os.path.join(save_path, f"episode_{episode}.pt")
        torch.save(episode_data, episode_file)
        print(f"Episode {episode} saved to {episode_file}")
        # Close the video writer
        out.release()
        episodes_info["episodes_num"] += 1
        episodes_info["episodes"].append(
            {
                "episode_id": episode,
                "length": len(episode_data["actions"]),
                "return": episode_reward,
                "source": "random_policy",
            }
        )
        print(f"Episode {episode} video saved to {video_filename}")

    # Save episodes info to disk
    episodes_info_file = os.path.join(save_path, "episodes_info.json")
    with open(episodes_info_file, "w") as json_file:
        json.dump(episodes_info, json_file, indent=4)


# Main function
if __name__ == "__main__":
    # Configuration
    parser = argparse.ArgumentParser(description="Collect Doom data")
    parser.add_argument(
        "--save_path",
        type=str,
        default="/scratch/gs4288/shared/diffusion_for_simulation/data/doom/test",
        help="Path to save the collected data",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1, help="Number of episodes to sample"
    )
    args = parser.parse_args()

    env_id = (
        "VizdoomMyWayHome-v0"  # Ensure the ViZDoom Gymnasium wrapper supports this ID
    )
    save_path = args.save_path
    num_episodes = args.num_episodes

    # Create environment
    env = gym.make(env_id)
    try:
        collect_episodes(env, num_episodes, save_path)
    finally:
        env.close()
