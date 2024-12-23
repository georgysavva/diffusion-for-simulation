from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from src.data.segment import SegmentId


@dataclass
class EpisodePartition:
    obs: torch.Tensor
    act: torch.IntTensor
    partition_start: int
    episode_length: int
    episode_id: int

    def __len__(self) -> int:
        return self.obs.size(0)

    def to(self, device) -> EpisodePartition:
        return EpisodePartition(
            **{
                k: v.to(device) if k in ["obs", "act"] else v
                for k, v in self.__dict__.items()
            }
        )

    @classmethod
    def load_for_segment(
        cls,
        path: Path,
        segment_id: SegmentId,
        episode_length: int,
        chunk_size: int,
        map_location: Optional[torch.device] = None,
    ):
        start_chunk = max(segment_id.start, 0) // chunk_size
        stop_chunk = (segment_id.stop + chunk_size) // chunk_size
        obs = []
        act = []
        for chunk_id in range(start_chunk, stop_chunk):

            chunk_obs, chunk_act = cls._load_chunk(
                path / f"chunk_{chunk_id:05d}.pt", map_location
            )
            obs.append(chunk_obs)
            act.append(chunk_act)

        obs = torch.cat(obs, dim=0)
        act = torch.cat(act, dim=0)
        partition_start = start_chunk * chunk_size
        return cls(
            obs=obs,
            act=act,
            partition_start=partition_start,
            episode_length=episode_length,
            episode_id=segment_id.episode_id,
        )

    @classmethod
    def load_whole_episode(
        cls, path: Path, map_location: Optional[torch.device] = None
    ) -> EpisodePartition:
        episode_id = int(path.stem.split("_")[-1])
        chunk_files = sorted(path.glob("chunk_*.pt"))
        obs = []
        act = []
        for chunk_file in chunk_files:
            chunk_obs, chunk_act = cls._load_chunk(chunk_file, map_location)
            obs.append(chunk_obs)
            act.append(chunk_act)

        obs = torch.cat(obs, dim=0)
        act = torch.cat(act, dim=0)

        episode_length = len(obs)
        return cls(
            obs=obs,
            act=act,
            partition_start=0,
            episode_length=episode_length,
            episode_id=episode_id,
        )

    @classmethod
    def _load_chunk(cls, path: Path, map_location: Optional[torch.device] = None):
        map_location = map_location or torch.device("cpu")
        chunk_data = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )
        return (
            chunk_data["observations"],
            chunk_data["actions"].to(torch.int32),
        )
