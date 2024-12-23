from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class Episode:
    obs: torch.Tensor
    act: torch.IntTensor
    episode_id: int

    def __len__(self) -> int:
        return self.obs.size(0)

    def to(self, device) -> Episode:
        return Episode(
            **{
                k: v.to(device) if k in ("obs", "act") else v
                for k, v in self.__dict__.items()
            }
        )

    def slice(self, start, stop) -> Episode:
        return Episode(
            obs=self.obs[start:stop],
            act=self.act[start:stop],
            episode_id=self.episode_id,
        )

    @classmethod
    def load(cls, path: Path, map_location: Optional[torch.device] = None) -> Episode:
        episode_id = int(path.stem.split(".")[0].split("_")[-1])
        data = torch.load(
            path, map_location=map_location or torch.device("cpu"), weights_only=True
        )
        obs = data["observations"]
        act = data["actions"].to(torch.int32)

        return cls(obs=obs, act=act, episode_id=episode_id)

    def save(self, path: Path) -> None:
        data = {
            "observations": self.obs,
            "actions": self.act,
        }
        torch.save(data, path)
