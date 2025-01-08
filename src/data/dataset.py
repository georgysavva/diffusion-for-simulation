import json
import math
from pathlib import Path
from typing import Generator

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
    ) -> None:
        super().__init__()

        self._directory = Path(directory).expanduser()
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
        episode = Episode.load(self.get_episode_path(episode_id))
        episode.obs = episode.obs.mul_(0.18215)

        return episode

    def get_episode_path(self, episode_id: int) -> Path:

        return self._directory / f"episode_{episode_id}.pt"


def collate_segments_to_batch(segments: list[Segment]) -> Batch:
    attrs = ("obs", "act")
    stack = (torch.stack([getattr(s, x) for s in segments]) for x in attrs)
    return Batch(*stack)


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
    stop = segment_id.stop
    obs = pad(episode.obs[start:stop])
    act = pad(episode.act[start:stop])
    assert obs.size(0) == act.size(0)
    assert obs.size(0) == segment_id.stop - segment_id.start
    return Segment(
        obs,
        act,
    )


class TestDatasetTraverser:

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        seq_length: int,
        subsample_rate: int,
        rank: int,
        world_size: int,
        seed_seq_length: int,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.subsample_rate = subsample_rate
        self.rank = rank
        self.world_size = world_size
        self.seed_seq_length = seed_seq_length

    def __len__(self):
        return math.ceil(
            sum(
                [
                    len(
                        range(
                            self.seed_seq_length + 1 - self.seq_length,
                            self.dataset.lengths[episode_id] - self.seq_length + 1,
                            self.subsample_rate,
                        )
                    )
                    for episode_id in range(
                        self.rank, self.dataset.num_episodes, self.world_size
                    )
                ]
            )
            / self.batch_size
        )

    def __iter__(self) -> Generator[Batch, None, None]:
        chunks = []
        for episode_id in range(self.rank, self.dataset.num_episodes, self.world_size):
            episode = self.dataset.load_episode(episode_id)
            for start in range(
                self.seed_seq_length + 1 - self.seq_length,
                len(episode) - self.seq_length + 1,
                self.subsample_rate,
            ):
                stop = start + self.seq_length
                segment = make_segment(
                    episode,
                    SegmentId(episode_id, start, stop),
                )
                chunks.append(segment)

            while len(chunks) >= self.batch_size:
                yield collate_segments_to_batch(chunks[: self.batch_size])
                chunks = chunks[self.batch_size :]

        if len(chunks) > 0:
            yield collate_segments_to_batch(chunks)
