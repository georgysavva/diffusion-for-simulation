import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy

from src.data.episode import Episode
from src.diffusion.respace import SpacedDiffusion
from src.utils import denormalize_img, normalize_img, to_numpy_video


class TrajectoryEvaluator:

    def __init__(
        self,
        diffusion: SpacedDiffusion,
        vae,
        diffusion_model,
        num_conditioning_steps: int,
        sampling_algorithm: str,
        vae_batch_size: int,
        device: torch.device,
    ):
        self._diffusion = diffusion
        self._vae = vae
        self._diffusion_model = diffusion_model
        self._num_conditioning_steps = num_conditioning_steps
        self._vae_batch_size = vae_batch_size
        self._device = device

        if sampling_algorithm == "DDPM":
            self._sampling_function = diffusion.p_sample_loop
        elif sampling_algorithm == "DDIM":
            self._sampling_function = diffusion.ddim_sample_loop
        else:
            raise ValueError(f"Unknown sampling algorithm: {sampling_algorithm}")

    @torch.no_grad()
    def evaluate_episodes(self, episodes, teacherforcing: bool, clip_denoised: bool) -> np.ndarray:
        self._diffusion_model.eval()
        self._vae.eval()

        batch_size = len(episodes)
        episode_lens = [len(e) for e in episodes]
        assert len(set(episode_lens)) == 1

        obs_imgs = torch.stack([episode.obs.to(self._device) for episode in episodes])
        acts = torch.stack([episode.act.to(self._device) for episode in episodes]) # (N, nframe,)
        obs_img_norms = [normalize_img(obs_img) for obs_img in obs_imgs]
        obs_latents = torch.stack([self._run_encode_on_episode(obs_img_norm) for obs_img_norm in obs_img_norms]) # (N, nframe, 4, 32, 32)

        prev_act = deepcopy(acts[:, :self._num_conditioning_steps])
        prev_obs = deepcopy(obs_latents[:, :self._num_conditioning_steps])
        generated_trajectory_latent = torch.empty(batch_size, obs_latents.shape[1] - self._num_conditioning_steps, *obs_latents.shape[-3:], device=self._device)

        for step in tqdm(range(self._num_conditioning_steps, len(episodes[0])), desc="Generating"):
            # generate next frame
            z = torch.randn(batch_size, self._num_conditioning_steps + 1, *obs_latents.shape[-3:], device=self._device)
            generated_obs = self._sampling_function(
                self._diffusion_model.forward,
                z.shape,
                z,
                clip_denoised=clip_denoised,
                model_kwargs={'prev_act': prev_act},
                progress=False,
                device=self._device,
                prev_obs=prev_obs,
            )
            # print((generated_obs[:self._num_conditioning_steps] - prev_obs).abs().mean())
            generated_obs = generated_obs[:, -1]

            # update conditioning
            prev_act = torch.roll(prev_act, -1, 1)
            prev_act[:, -1] = acts[:, step]
            prev_obs = torch.roll(prev_obs, -1, 1)
            if teacherforcing:
                prev_obs[:, -1] = obs_latents[:, step]
            else:
                prev_obs[:, -1] = generated_obs
            # update output
            generated_trajectory_latent[:, step - self._num_conditioning_steps] = generated_obs

        generated_trajectory_img_norm = torch.stack([self._run_decode_on_episode(gen) for gen in generated_trajectory_latent])
        generated_trajectory_img = denormalize_img(generated_trajectory_img_norm)
        return torch.cat([obs_imgs[:, :self._num_conditioning_steps], generated_trajectory_img], dim=1)

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
        for i in range(0, len(obs_img_norm), self._vae_batch_size):
            batch = obs_img_norm[i : i + self._vae_batch_size]
            obs_latent.append(
                self._vae.encode(batch).latent_dist.sample().mul_(0.18215)
            )
        obs_latent = torch.cat(obs_latent, dim=0)
        return obs_latent

    def _run_decode_on_episode(self, obs_latent) -> torch.Tensor:
        obs_img_norm = []
        for i in range(0, len(obs_latent), self._vae_batch_size):
            batch = obs_latent[i : i + self._vae_batch_size]
            obs_img_norm.append(self._vae.decode(batch / 0.18215).sample.clamp(-1, 1))
        obs_img_norm = torch.cat(obs_img_norm, dim=0)
        return obs_img_norm
