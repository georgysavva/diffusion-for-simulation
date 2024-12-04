import torch
from tqdm import tqdm

from data.episode import Episode
from diffusion.respace import SpacedDiffusion
from src.utils import transform_image_obs


class TrajectoryEvaluator:
    def __init__(
        self,
        diffusion: SpacedDiffusion,
        vae,
        num_seed_steps: int,
        image_resolution: int,
        num_conditioning_steps: int,
        device: torch.device,
    ):
        self._diffusion = diffusion
        self._vae = vae
        self._num_seed_steps = num_seed_steps
        self._image_resolution = image_resolution
        self._num_conditioning_steps = num_conditioning_steps
        self._device = device

    def evaluate_episode(self, model, episode: Episode, auto_regressive: bool = False):
        obs_img = episode.obs.to(self._device)
        act = episode.act.to(self._device)
        obs_img = transform_image_obs(obs_img, self._image_resolution)
        obs = self._vae.encode(obs).latent_dist.sample()
        latent_shape = obs.shape[-3:]
        prev_obs = torch.zeros(
            self._num_conditioning_steps, *latent_shape, device=self._device
        )
        prev_act = torch.zeros(self._num_conditioning_steps, device=self._device)
        prev_obs[-self._num_seed_steps :] = obs[: self._num_seed_steps]
        prev_act[-self._num_seed_steps :] = act[: self._num_seed_steps]

        generated_trajectory_obs = torch.empty_like(obs, device=self._device)
        generated_trajectory_obs[: self._num_seed_steps] = obs[: self._num_seed_steps]
        for step in tqdm(range(self._num_seed_steps, len(episode)), desc="Generating"):
            z = torch.randn(1, *latent_shape, device=self._device)
            model_kwargs = dict(
                prev_obs=prev_obs.unsqueeze(0), prev_act=prev_act.unsqueeze(0)
            )
            generated_obs = self._diffusion.p_sample_loop(
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
                prev_obs[-1] = obs[step]
            generated_trajectory_obs[step] = generated_obs
        generated_trajectory_img = torch.cat(
            obs_img[: self._num_seed_steps],
            self._vae.decode(generated_trajectory_obs).sample.clamp(-1, 1),
        )
        return generated_trajectory_img
