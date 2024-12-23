from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Union

import torch


@dataclass
class SegmentId:
    episode_id: int
    start: int
    stop: int


@dataclass
class Segment:
    obs: torch.Tensor
    act: torch.IntTensor

    def __len__(self) -> int:
        return self.obs.size(0)
