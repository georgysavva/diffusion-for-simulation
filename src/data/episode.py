from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class Episode:
    obs: torch.Tensor
    act: torch.IntTensor

    def __len__(self) -> int:
        return self.obs.size(0)

    def to(self, device) -> Episode:
        return Episode(**{k: v.to(device) for k, v in self.__dict__.items()})

    @classmethod
    def load(cls, path: Path, map_location: Optional[torch.device] = None) -> Episode:
        data = torch.load(path, map_location=map_location)
        obs = data["observations"]
        act = data["actions"].to(torch.int32)

        return cls(obs=obs, act=act)
