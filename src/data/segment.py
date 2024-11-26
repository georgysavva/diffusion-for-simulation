from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Union

import torch


@dataclass
class SegmentId:
    episode_id: Union[int, str]
    start: int
    stop: int


@dataclass
class Segment:
    obs: torch.ByteTensor
    act: torch.IntTensor
    id: SegmentId
