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
    diffusion_model.load_state_dict(
        torch.load(
            run_dir / "diffusion_model_versions" / args.model_version,
            map_location=device,
            weights_only=True,
        )
    )

    episode = Episode.load(args.episode_path)
    episode_name = os.path.splitext(os.path.basename(args.episode_path))[0]
    episode.obs = prepare_image_obs(
        episode.obs, run_config.static_dataset.image_resolution
    )

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
        run_dir
        / "trajectory_evaluation"
        / args.model_version.split(".")[0]
        / episode_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for generation_mode in ["teacher_forcing"]:
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
        required=True,
        help="Name of the checkpoint file.",
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
        default="/scratch/gs4288/shared/diffusion_for_simulation/data/doom/original/test/episode_11.pt",
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
        default="DDPM",
        help="Sampling algorithm to use for diffusion.",
    )
    args = parser.parse_args()
    main(args)
