import argparse
import os
from pathlib import Path

import hydra
import torch
from diffusers import AutoencoderKL
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.data.episode import Episode
from src.diffusion import create_diffusion
from src.traj_eval import TrajectoryEvaluator
from src.utils import prepare_image_obs, save_np_video, to_numpy_video


def main(args):
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.decoder.load_state_dict(
        torch.load(args.vae_decoder_path, weights_only=True, map_location=device)
    )
    run_dir = Path(args.run_dir)
    run_config_path = run_dir / ".hydra" / "config.yaml"
    run_config = OmegaConf.load(run_config_path)
    diffusion = create_diffusion(
        str(args.num_sampling_steps), learn_sigma=run_config.diffusion.learn_sigma
    )
    diffusion_model = instantiate(run_config.diffusion_model.model).to(device)
    if args.model_version == "latest":
        model_versions = sorted(
            os.listdir(run_dir / "diffusion_model_versions"), reverse=True
        )
        if len(model_versions) == 0:
            raise ValueError("No model versions found.")
        elif len(model_versions) == 1:
            model_version = model_versions[0]
        else:
            model_version = model_versions[
                1
            ]  # Prefer the second latest model because the most latest if a temp one
    else:
        model_version = args.model_version
    print("Using model version:", model_version)
    diffusion_model.load_state_dict(
        torch.load(
            run_dir / "diffusion_model_versions" / model_version,
            map_location=device,
            weights_only=True,
        )
    )

    episode = Episode.load(args.episode_path)
    episode_name = os.path.splitext(os.path.basename(args.episode_path))[0]
    episode.obs = prepare_image_obs(
        episode.obs, run_config.static_dataset.image_resolution
    )
    if args.take_first_n_steps is not None:
        episode = episode.slice(0, args.take_first_n_steps)
    evaluator = TrajectoryEvaluator(
        diffusion=diffusion,
        vae=vae,
        num_seed_steps=(
            run_config.diffusion_model.model.num_conditioning_steps
            if run_config.static_dataset.guarantee_full_seqs
            else args.num_seed_steps
        ),
        num_conditioning_steps=run_config.diffusion_model.model.num_conditioning_steps,
        sampling_algorithm=args.sampling_algorithm,
        vae_batch_size=args.vae_batch_size,
        device=device,
    )
    output_dir = (
        run_dir / "trajectory_evaluation" / model_version.split(".")[0] / episode_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for generation_mode in args.generation_modes:
        generated_trajectory = evaluator.evaluate_episode(
            diffusion_model, episode, generation_mode
        )
        save_np_video(
            generated_trajectory,
            output_dir / f"generated_{generation_mode}_{args.sampling_algorithm}.mp4",
            run_config.env.fps,
        )

    ground_truth_trajectory = to_numpy_video(episode.obs)
    save_np_video(
        ground_truth_trajectory,
        output_dir / f"ground_truth.mp4",
        run_config.env.fps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trajectory quality.")
    parser.add_argument("--run_dir", type=str, help="Path to the directory of the run.")
    parser.add_argument(
        "--model_version",
        type=str,
        default="latest",
        help="Name of the checkpoint file.",
    )
    parser.add_argument(
        "--generation_modes",
        type=str,
        nargs='+',
        default=["teacher_forcing"],
        help="List of generation modes to evaluate.",
    )
    parser.add_argument(
        "--vae_decoder_path",
        type=str,
        help="Path to the VAE model.",
        default="/scratch/gs4288/shared/diffusion_for_simulation/vae/trained_vae_decoder.pth",
    )

    parser.add_argument(
        "--episode_path",
        type=str,
        help="Path to the episode data.",
        default="/scratch/gs4288/shared/diffusion_for_simulation/data/doom/original/test/episode_0.pt",
    )
    parser.add_argument(
        "--num_seed_steps", type=int, help="Number of seed steps.", default=8
    )
    parser.add_argument(
        "--num_sampling_steps",
        type=int,
        help="Number of diffusion sampling steps.",
        default=8,
    )
    parser.add_argument(
        "--take_first_n_steps",
        type=int,
        help="Number of first frames in the episode to generate the trajectory for.",
        default=100,
    )
    parser.add_argument("--device", type=str, help="Device to run the evaluation on.")
    parser.add_argument(
        "--vae_batch_size",
        type=int,
        default=128,
        help="Batch size for VAE encode and decode",
    )
    parser.add_argument(
        "--sampling_algorithm",
        type=str,
        choices=["DDIM", "DDPM"],
        default="DDIM",
        help="Sampling algorithm to use for diffusion.",
    )
    args = parser.parse_args()
    main(args)
