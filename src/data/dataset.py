import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import Generator

import einops
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset as TorchDataset

from .batch import Batch
from .episode import Episode
from .segment import Segment, SegmentId


class Dataset(TorchDataset):

    def __init__(
        self,
        directory: Path,
        cache_in_ram: bool = False,
        use_manager: bool = False,
    ) -> None:
        super().__init__()

        self._directory = Path(directory).expanduser()
        self._cache_in_ram = cache_in_ram
        self._cache = mp.Manager().dict() if use_manager else {}
        with open(self._directory / "episodes_info.json", "r") as json_file:
            self.episodes_info = json.load(json_file)
        self._num_episodes = self.episodes_info["episodes_num"]
        self._lengths = np.array(
            [ep["length"] for ep in self.episodes_info["episodes"]]
        )

    @property
    def num_episodes(self) -> int:
        return self._num_episodes

    @property
    def lengths(self) -> np.ndarray:
        return self._lengths

    def __getitem__(self, segment_id: SegmentId) -> Segment:
        episode = self.load_episode(segment_id.episode_id)
        segment = make_segment(episode, segment_id)
        return segment

    def load_episode(self, episode_id: int) -> Episode:
        if self._cache_in_ram and episode_id in self._cache:
            episode = self._cache[episode_id]
        else:
            episode = Episode.load(self._get_episode_path(episode_id))
            episode.obs = episode.obs.mul_(0.18215)
            if self._cache_in_ram:
                self._cache[episode_id] = episode
        return episode

    def _get_episode_path(self, episode_id: int) -> Path:

        return self._directory / f"episode_{episode_id}.pt"


def collate_segments_to_batch(segments: list[Segment]) -> Batch:
    attrs = ("obs", "act")
    stack = (torch.stack([getattr(s, x) for s in segments]) for x in attrs)
    return Batch(*stack, [s.id for s in segments])


def make_segment(episode: Episode, segment_id: SegmentId) -> Segment:
    assert (
        segment_id.start < len(episode)
        and segment_id.stop > 0
        and segment_id.start < segment_id.stop
    )
    assert segment_id.stop <= len(episode)
    pad_len_left = max(0, -segment_id.start)

    def pad(x):
        return (
            F.pad(x, [0 for _ in range(2 * x.ndim - 2)] + [pad_len_left, 0])
            if pad_len_left > 0
            else x
        )

    start = max(0, segment_id.start)
    stop = min(len(episode), segment_id.stop)
    obs = pad(episode.obs[start:stop])
    act = pad(episode.act[start:stop])
    return Segment(
        obs,
        act,
        id=SegmentId(segment_id.episode_id, start, stop),
    )


class TestDatasetTraverser:

    def __init__(
        self, dataset: Dataset, batch_num_samples: int, seq_length: int
    ) -> None:
        self.dataset = dataset
        self.batch_num_samples = batch_num_samples
        self.seq_length = seq_length

    def __len__(self):
        return math.ceil(
            sum(
                [
                    math.floor(self.dataset.lengths[episode_id] / self.seq_length)
                    for episode_id in range(self.dataset.num_episodes)
                ]
            )
            / self.batch_num_samples
        )

    def __iter__(self) -> Generator[Batch, None, None]:
        chunks = []
        for episode_id in range(self.dataset.num_episodes):
            episode = self.dataset.load_episode(episode_id)
            for i in range(math.floor(len(episode) / self.seq_length)):
                start = i * self.seq_length
                stop = (i + 1) * self.seq_length
                segment = make_segment(
                    episode,
                    SegmentId(episode_id, start, stop),
                )
                chunks.append(segment)
            if chunks[-1].effective_size < 2:
                chunks.pop()

            while len(chunks) >= self.batch_num_samples:
                yield collate_segments_to_batch(chunks[: self.batch_num_samples])
                chunks = chunks[self.batch_num_samples :]

        if len(chunks) > 0:
            yield collate_segments_to_batch(chunks)
