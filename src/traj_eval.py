import numpy as np
import torch
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
        self._device = device

    @torch.no_grad()
    def evaluate_episode(
        self, model, episode: Episode, auto_regressive: bool = False
    ) -> np.ndarray:
        model.eval()
        self._vae.eval()
        obs_img = episode.obs.cpu()
        act = episode.act.to(self._device)
        obs_img_norm = normalize_img(obs_img)
        seed_obs = (
            self._vae.encode(obs_img_norm[: self._num_seed_steps].to(self._device))
            .latent_dist.sample()
            .mul_(0.18215)
        )
        latent_shape = seed_obs.shape[-3:]
        prev_obs = torch.zeros(
            self._num_conditioning_steps, *latent_shape, device=self._device
        )
        prev_act = torch.zeros(
            self._num_conditioning_steps, device=self._device, dtype=torch.int32
        )
        prev_obs[-self._num_seed_steps :] = seed_obs
        prev_act[-self._num_seed_steps :] = act[: self._num_seed_steps]

        generated_trajectory_img = torch.empty_like(obs_img)
        generated_trajectory_img[: self._num_seed_steps] = obs_img[
            : self._num_seed_steps
        ]
        for step in tqdm(range(self._num_seed_steps, len(episode)), desc="Generating"):
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
            prev_act = torch.roll(prev_act, -1, 0)
            prev_act[-1] = act[step]
            prev_obs = torch.roll(prev_obs, -1, 0)
            if auto_regressive:
                prev_obs[-1] = generated_obs
            else:
                prev_obs[-1] = (
                    self._vae.encode(obs_img_norm[step].unsqueeze(0).to(self._device))
                    .latent_dist.sample()
                    .mul_(0.18215)
                    .squeeze(0)
                )
            generated_trajectory_img[step] = (
                denormalize_img(
                    self._vae.decode(generated_obs.unsqueeze(0) / 0.18215).sample.clamp(
                        -1, 1
                    )
                )
                .squeeze(0)
                .cpu()
            )

        generated_trajectory_img_np = to_numpy_video(generated_trajectory_img)
        return generated_trajectory_img_np
