from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from .segment import SegmentId


@dataclass
class Batch:
    obs: torch.Tensor
    act: torch.IntTensor
    segment_ids: List[SegmentId]

    def pin_memory(self) -> Batch:
        return Batch(
            **{
                k: v if k in ("segment_ids") else v.pin_memory()
                for k, v in self.__dict__.items()
            }
        )

    def to(self, device: torch.device) -> Batch:
        return Batch(
            **{
                k: v if k in ("segment_ids") else v.to(device)
                for k, v in self.__dict__.items()
            }
        )
