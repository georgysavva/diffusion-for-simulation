import argparse
import os
from pathlib import Path  # Add argparse for CLI arguments

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from vizdoom import gymnasium_wrapper  # to register envs

# Ensure the vizdoom environment is installed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PPO agent on Vizdoom")
    parser.add_argument(
        "--timesteps", type=int, default=1_000_000, help="Total timesteps for training"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/scratch/gs4288/shared/diffusion_for_simulation/rl_agent/doom/",
        help="Path to save the model checkpoints",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10_000,
        help="Frequency of evaluation during training",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the environment during evaluation"
    )
    return parser.parse_args()


def main(args):
    # Environment parameters
    env_id = "VizdoomMyWayHome-v0"
    timesteps = args.timesteps  # Use CLI argument for timesteps

    # Create and wrap the environment
    def make_env():
        return gym.make(env_id)

    # Vectorize the environment for stable training
    env = make_vec_env(make_env, n_envs=4)

    # Create evaluation environment
    eval_env = make_env()

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.eval_freq,
        save_path=os.path.join(args.save_path, "models"),
        name_prefix="vizdoom_ppo",  # Use CLI argument for save_path
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.save_path, "best_model"),
        log_path="./logs/",
        eval_freq=args.eval_freq,  # Use CLI argument for eval_freq
        deterministic=True,
        render=args.render,  # Use CLI argument for render
    )

    # Initialize the PPO model
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")

    # Train the model
    model.learn(
        total_timesteps=timesteps, callback=[checkpoint_callback, eval_callback]
    )

    # Save the final model
    model.save("vizdoom_ppo_final")


# Test the trained model
def test_agent(model_path, env_id):
    model = PPO.load(model_path)
    env = gym.make(env_id)
    obs = env.reset()
    total_reward = 0

    for _ in range(1000):  # Run the environment for a fixed number of steps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        if done:
            obs = env.reset()

    print(f"Total reward during test: {total_reward}")
    env.close()


if __name__ == "__main__":
    args = parse_args()  # Parse CLI arguments
    main(args)
    # Uncomment to test the agent after training
    # test_agent("vizdoom_ppo_final", env_id)
