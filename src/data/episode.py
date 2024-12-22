from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class Episode:
    obs: torch.Tensor
    act: torch.IntTensor
    rew: torch.FloatTensor

    def __len__(self) -> int:
        return self.obs.size(0)

    def to(self, device) -> Episode:
        return Episode(**{k: v.to(device) for k, v in self.__dict__.items()})

    def slice(self, start, stop) -> Episode:
        return Episode(
            obs=self.obs[start:stop],
            act=self.act[start:stop],
            rew=self.rew[start:stop],
        )

    @classmethod
    def load(cls, path: Path, map_location: Optional[torch.device] = None) -> Episode:
        data = torch.load(path, map_location=map_location, weights_only=True)
        obs = data["observations"]
        act = data["actions"].to(torch.int32)
        rew = data["rewards"]

        return cls(obs=obs, act=act, rew=rew)

    def save(self, path: Path) -> None:
        data = {
            "observations": self.obs,
            "actions": self.act,
            "rewards": self.rew,
        }
        torch.save(data, path)
