# python scripts/evaluate_trajectory_vda.py \
#   --run_dir /scratch/gs4288/shared/diffusion_for_simulation/runs/2024.12.11T22-54_main_action_frame_xl_pretrained_short \
#   --model_version diffusion_model_epoch_00012.pt \
#   --clip_denoised \
#   --num_sampling_steps 250 \
#   --sampling_algorithm DDPM \
#   --teacherforcing

import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from diffusers import AutoencoderKL
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.data.episode import Episode
from src.diffusion import create_diffusion
from src.traj_eval import TrajectoryEvaluator
from src.utils import prepare_image_obs, save_np_video, to_numpy_video

from PIL import Image, ImageDraw, ImageFont


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_concatenated_images_with_text(images, words, margin=10, font_size=24):
    """
    Concatenate images in a row with a margin and add words below each image except the last one.

    Args:
    - images (torch.Tensor): Tensor of shape (N, 3, 256, 256) in range [0, 255], dtype uint8.
    - words (list of str): List of N-1 words for labeling the images.
    - margin (int): Margin between images in pixels.
    - output_path (str): File path to save the concatenated image.
    - font_size (int): Font size for the text.
    """
    # Convert torch tensor to PIL images
    pil_images = [T.ToPILImage()(img) for img in images]

    # Load a default font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate the maximum text height for words
    temp_image = Image.new("RGB", (1, 1), "white")
    temp_draw = ImageDraw.Draw(temp_image)
    text_heights = [
        temp_draw.textbbox((0, 0), word, font=font)[3] - temp_draw.textbbox((0, 0), word, font=font)[1]
        for word in words
    ]
    max_text_height = max(text_heights) if text_heights else 0

    # Determine dimensions
    N, _, height, width = images.shape
    total_width = N * width + (N - 1) * margin
    output_height = height + max_text_height + margin
    output_image = Image.new("RGB", (total_width, output_height), "white")
    draw = ImageDraw.Draw(output_image)

    # Paste images and add text
    x_offset = 0
    for i, img in enumerate(pil_images):
        output_image.paste(img, (x_offset, 0))
        if i < len(words):  # Add text below the image (except the last one)
            text_bbox = draw.textbbox((0, 0), words[i], font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = x_offset + (width - text_width) // 2
            text_y = height + (margin // 2)
            draw.text((text_x, text_y), words[i], fill="black", font=font)
        x_offset += width + margin

    return output_image


def main(args):
    run_dir = Path(args.run_dir)
    run_config_path = run_dir / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(run_config_path)

    # vae
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.decoder.load_state_dict(torch.load(cfg.inference.vae_path, weights_only=True, map_location=device))
    print('loaded diffusion model')

    # diffusion
    diffusion = create_diffusion(str(args.num_sampling_steps), learn_sigma=cfg.diffusion.learn_sigma)
    diffusion_model = instantiate(cfg.diffusion_model.model).to(device)
    diffusion_model.load_state_dict(torch.load(run_dir / "diffusion_model_versions" / args.model_version, weights_only=True, map_location=device))
    print('loaded vae decoder')

    # data
    episode_paths = [l.strip() for l in open(args.episodes_path, 'r').readlines() if l.strip()]
    episode_starts = [int(l.strip()) for l in open(args.episode_starts_path, 'r').readlines() if l.strip()]
    episodes = [Episode.load(p) for p in episode_paths]
    assert len(episodes) == len(episode_starts)
    for start, episode in zip(episode_starts, episodes):
        start = min(start, len(episode) - cfg.diffusion_model.model.num_conditioning_steps - args.num_gen)
        # assert start + cfg.diffusion_model.model.num_conditioning_steps + args.num_gen <= len(episode)
        episode.obs = episode.obs[start: start + cfg.diffusion_model.model.num_conditioning_steps + args.num_gen]
        episode.act = episode.act[start: start + cfg.diffusion_model.model.num_conditioning_steps + args.num_gen]
        episode.rew = episode.rew[start: start + cfg.diffusion_model.model.num_conditioning_steps + args.num_gen]
        episode.obs = prepare_image_obs(episode.obs, args.img_resolution)

    evaluator = TrajectoryEvaluator(
        diffusion=diffusion,
        vae=vae,
        diffusion_model=diffusion_model,
        num_conditioning_steps=cfg.diffusion_model.model.num_conditioning_steps,
        sampling_algorithm=args.sampling_algorithm,
        vae_batch_size=args.vae_batch_size,
        device=device,
    )
    generated_trajectories = evaluator.evaluate_episodes(episodes, args.teacherforcing, args.clip_denoised) # (N, nframe, 3, 256, 256)
    assert generated_trajectories.shape[0] == len(episodes)
    assert generated_trajectories.shape[1] == cfg.diffusion_model.model.num_conditioning_steps + args.num_gen

    tag = "teacherforcing" if args.teacherforcing else "autoregressive"
    for gen, episode, episode_path in zip(generated_trajectories, episodes, episode_paths):
        assert gen.shape == episode.obs.shape # (nframe, 3, 256, 256)
        output_dir = run_dir / "trajectory_evaluation" / args.model_version / Path(episode_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        act_names = [["NOOP", "TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD", "MOVE_LEFT", "MOVE_RIGHT"][act] for act in episode.act]
        assert len(act_names) == len(gen)
        gt_img_path = output_dir / 'groundtruth.jpg'
        pred_img_path = output_dir / f'{tag}.jpg'
        save_concatenated_images_with_text(episode.obs, act_names[:-1]).save(gt_img_path)
        save_concatenated_images_with_text(gen, act_names[:-1]).save(pred_img_path)
        gt_vid_path = output_dir / 'groundtruth.mp4'
        pred_vid_path = output_dir / f'{tag}.mp4'
        save_np_video(to_numpy_video(episode.obs), gt_vid_path, fps=15)
        save_np_video(to_numpy_video(gen), pred_vid_path, fps=15)

        print("saved groundtruth img to", gt_img_path)
        print("saved groundtruth vid to", gt_vid_path)
        print("saved pred img to", pred_img_path)
        print("saved pred vid to", pred_vid_path)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trajectory quality.")
    parser.add_argument("--run_dir", type=str, help="Path to the directory of the run.")
    parser.add_argument("--model_version", type=str, required=True, help="Name of the checkpoint file.")
    parser.add_argument("--fps", type=int, help="Trajectory frames per second.", default=2)
    parser.add_argument("--img_resolution", type=int, help="Resolution of the images.", default=256)
    parser.add_argument("--vae_batch_size", type=int, default=32, help="Batch size for VAE encode and decode",)
    parser.add_argument("--sampling_algorithm", type=str, choices=["DDIM", "DDPM"], default="DDIM", help="Sampling algorithm to use for diffusion.")
    parser.add_argument("--teacherforcing", action='store_true')
    parser.add_argument("--clip_denoised", action='store_true')
    parser.add_argument("--num_sampling_steps", type=int, help="Number of diffusion sampling steps.", default=25)
    # data
    parser.add_argument("--episodes_path", type=str, default="trajectory_test_bed/episode_paths.txt")
    parser.add_argument("--episode_starts_path", type=str, default="trajectory_test_bed/episode_starts.txt")
    parser.add_argument("--episode_start", type=int, default=0)
    parser.add_argument("--num_gen", type=int, default=10)
    args = parser.parse_args()
    main(args)
