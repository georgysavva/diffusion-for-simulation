import argparse  # Add argparse import
import os

import gymnasium as gym
import torch
from tqdm import tqdm
from vizdoom import gymnasium_wrapper  # to register envs


# Collect episodes using a random policy
def collect_episodes(env, num_episodes, save_path):
    os.makedirs(save_path, exist_ok=True)

    for episode in tqdm(range(num_episodes), desc="Sampling episodes"):
        episode_data = {"observations": [], "actions": [], "rewards": []}
        observation, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # Random action
            action = env.action_space.sample()

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
        episode_file = os.path.join(
            save_path, f"episode_{episode}_r{episode_reward:.2f}.pt"
        )
        torch.save(episode_data, episode_file)
        print(f"Episode {episode} saved to {episode_file}")


# Main function
if __name__ == "__main__":
    # Configuration
    parser = argparse.ArgumentParser(description="Collect Doom data")
    parser.add_argument(
        "--save_path",
        type=str,
        default="../shared/datasets/doom/random_policy",
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
