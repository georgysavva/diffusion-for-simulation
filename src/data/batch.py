from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Batch:
    obs: torch.Tensor
    act: torch.IntTensor

    def pin_memory(self) -> Batch:
        return Batch(**{k: v.pin_memory() for k, v in self.__dict__.items()})

    def to(self, device: torch.device) -> Batch:
        return Batch(**{k: v.to(device) for k, v in self.__dict__.items()})
