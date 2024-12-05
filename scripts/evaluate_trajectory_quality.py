import argparse

import torch
from diffusers import AutoencoderKL

from data.episode import Episode
from diffusion.respace import SpacedDiffusion
from src.traj_eval import TrajectoryEvaluator


def main(args):
    device = torch.device(args.device)
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    model = load_model(args.model_path, device)
    episode = Episode.load(args.episode_path)

    evaluator = TrajectoryEvaluator(
        diffusion=diffusion,
        vae=vae,
        num_seed_steps=args.num_seed_steps,
        num_conditioning_steps=args.num_conditioning_steps,
        device=device,
    )

    generated_trajectory = evaluator.evaluate_episode(
        model, episode, args.auto_regressive
    )
    # Save or process the generated trajectory as needed
    # For example, save to a file
    np.save(args.output_path, generated_trajectory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trajectory quality.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the diffusion model."
    )
    parser.add_argument(
        "--vae_path", type=str, required=True, help="Path to the VAE model."
    )
    parser.add_argument(
        "--episode_path", type=str, required=True, help="Path to the episode data."
    )
    parser.add_argument(
        "--num_seed_steps", type=int, help="Number of seed steps."
        default = 10
    )
    parser.add_argument(
        "--num_diffusion_steps",
        type=int,
        help="Number of diffusion sampling steps.",
        default=250
    )
    parser.add_argument(
        "--num_conditioning_steps",
        type=int,
        help="Number of conditioning steps.",
        default = 300
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the evaluation on."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the generated trajectory.",
    )
    args = parser.parse_args()
    main(args)
