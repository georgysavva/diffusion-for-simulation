import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.episode import Episode
from src.diffusion.respace import SpacedDiffusion
from src.utils import denormalize_img, normalize_img, prepare_image_obs, to_numpy_video


class TrajectoryEvaluator:

    def __init__(
        self,
        diffusion: SpacedDiffusion,
        vae,
        num_seed_steps: int,
        num_conditioning_steps: int,
        sampling_algorithm: str,
        vae_batch_size: int,
        device: torch.device,
    ):
        self._diffusion = diffusion
        self._vae = vae
        self._num_seed_steps = num_seed_steps
        self._num_conditioning_steps = num_conditioning_steps
        if sampling_algorithm == "DDPM":
            self._sampling_function = diffusion.p_sample_loop
        elif sampling_algorithm == "DDIM":
            self._sampling_function = diffusion.ddim_sample_loop
        else:
            raise ValueError(f"Unknown sampling algorithm: {sampling_algorithm}")
        assert (
            num_conditioning_steps >= num_seed_steps
        ), "Number of conditioning steps must be greater than or equal to the number of seed steps."
        self._vae_batch_size = vae_batch_size
        self._device = device

    @torch.no_grad()
    def evaluate_episode(
        self, model, episode: Episode, generation_mode: str
    ) -> tuple[np.ndarray, float]:
        model.eval()
        self._vae.eval()
        obs_img = episode.obs.to(self._device)
        act = episode.act.to(self._device)
        obs_img_norm = normalize_img(obs_img)
        obs_latent = self._run_encode_on_episode(obs_img_norm)
        latent_shape = obs_latent.shape[-3:]
        prev_obs = torch.zeros(
            self._num_conditioning_steps, *latent_shape, device=self._device
        )
        prev_act = torch.zeros(
            self._num_conditioning_steps, device=self._device, dtype=torch.int32
        )
        prev_obs[-self._num_seed_steps :] = obs_latent[: self._num_seed_steps]
        prev_act[-self._num_seed_steps :] = act[: self._num_seed_steps]

        generated_trajectory_latent = torch.empty(
            obs_latent.size(0) - self._num_seed_steps,
            *latent_shape,
            device=self._device,
        )
        for step in tqdm(
            range(self._num_seed_steps, len(episode)),
            desc="Inference/Generating trajectory",
        ):
            z = torch.randn(1, *latent_shape, device=self._device)
            model_kwargs = dict(
                prev_obs=prev_obs.unsqueeze(0), prev_act=prev_act.unsqueeze(0)
            )
            generated_obs = self._sampling_function(
                model.forward,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=self._device,
            ).squeeze(0)
            if self._num_conditioning_steps > 0:
                prev_act = torch.roll(prev_act, -1, 0)
                prev_act[-1] = act[step]
                prev_obs = torch.roll(prev_obs, -1, 0)
                if generation_mode == "auto_regressive":
                    prev_obs[-1] = generated_obs
                elif generation_mode == "teacher_forcing":
                    prev_obs[-1] = obs_latent[step]
                else:
                    raise ValueError(
                        f"Unknown generation mode: {generation_mode}. "
                        "Choose from ['auto_regressive', 'teacher_forcing']."
                    )
            generated_trajectory_latent[step - self._num_seed_steps] = generated_obs
        generated_trajectory_img_norm = self._run_decode_on_episode(
            generated_trajectory_latent
        )
        generated_trajectory_img = denormalize_img(generated_trajectory_img_norm)
        psnr = compute_psnr(obs_img[self._num_seed_steps:], generated_trajectory_img)
        full_trajectory_img = torch.cat(
            [obs_img[: self._num_seed_steps], generated_trajectory_img], dim=0
        )

        generated_trajectory_img_np = to_numpy_video(full_trajectory_img)
        return generated_trajectory_img_np, psnr

    @torch.no_grad()
    def run_vae_on_episode(
        self,
        episode: Episode,
    ) -> np.ndarray:
        self._vae.eval()
        obs_img = episode.obs.to(self._device)
        obs_img_norm = normalize_img(obs_img)
        obs_latent = self._run_encode_on_episode(obs_img_norm)
        obs_img_norm = self._run_decode_on_episode(obs_latent)
        obs_img = denormalize_img(obs_img_norm)
        return to_numpy_video(obs_img)

    def _run_encode_on_episode(self, obs_img_norm) -> torch.Tensor:
        obs_latent = []
        for i in tqdm(
            range(0, len(obs_img_norm), self._vae_batch_size),
            desc="Inference/Vae encoding",
        ):
            batch = obs_img_norm[i : i + self._vae_batch_size]
            obs_latent.append(
                self._vae.encode(batch).latent_dist.sample().mul_(0.18215)
            )
        obs_latent = torch.cat(obs_latent, dim=0)
        return obs_latent

    def _run_decode_on_episode(self, obs_latent) -> torch.Tensor:
        obs_img_norm = []
        for i in tqdm(
            range(0, len(obs_latent), self._vae_batch_size),
            desc="Inference/Vae decoding",
        ):
            batch = obs_latent[i : i + self._vae_batch_size]
            obs_img_norm.append(self._vae.decode(batch / 0.18215).sample.clamp(-1, 1))
        obs_img_norm = torch.cat(obs_img_norm, dim=0)
        return obs_img_norm

def compute_psnr(frames1: torch.Tensor, frames2: torch.Tensor, max_pixel_value: int = 255) -> float:
    """
    Compute the average PSNR between two batches of image frames.

    Args:
        frames1 (torch.Tensor): Tensor of shape (N, C, H, W) representing the first set of frames.
        frames2 (torch.Tensor): Tensor of shape (N, C, H, W) representing the second set of frames.
        max_pixel_value (float): The maximum pixel value (default is 1.0 for normalized images).

    Returns:
        float: The average PSNR across all frames.
    """
    frames1 = frames1.float()
    frames2 = frames2.float()
    # Check that tensors have the same shape
    if frames1.shape != frames2.shape:
        raise ValueError("Input tensors must have the same shape")

    # Compute Mean Squared Error (MSE) for each frame
    mse = F.mse_loss(frames1, frames2, reduction='none')  # Shape: (N, C, H, W)
    mse_per_frame = mse.mean(dim=(1, 2, 3))  # Shape: (N,) - Mean MSE per frame

    # Avoid division by zero by clamping MSE values
    mse_per_frame = torch.clamp(mse_per_frame, min=1e-10)
    max_pixel_value = torch.tensor(max_pixel_value, device=frames1.device, dtype=frames1.dtype)
    # Compute PSNR per frame
    psnr_per_frame = 10 * torch.log10((max_pixel_value ** 2) / mse_per_frame)

    # Average PSNR across all frames
    avg_psnr = psnr_per_frame.mean().item()

    return avg_psnr


def to_strip_of_images(frames, num_seed_frames, stride, num_frames):
    """
    Convert a batch of video frames into a strip of images for visualization.
    """
    # Select frames from the video
    frames = frames[
        num_seed_frames - 1 : (num_seed_frames - 1) + num_frames * stride : stride
    ]
    horizontal_strip = np.concatenate(frames, axis=1)
    return horizontal_strip
