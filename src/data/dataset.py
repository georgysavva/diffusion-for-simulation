import json
import math
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset as TorchDataset

from .batch import Batch
from .episode_partition import EpisodePartition
from .segment import Segment, SegmentId


class Dataset(TorchDataset):

    def __init__(
        self,
        directory: Path,
        guarantee_full_seqs: bool,
    ) -> None:
        super().__init__()

        self._directory = Path(directory).expanduser()
        with open(self._directory / "episodes_info.json", "r") as json_file:
            self.episodes_info = json.load(json_file)
        self._num_episodes = self.episodes_info["episodes_num"]
        self._lengths = np.array(
            [ep["length"] for ep in self.episodes_info["episodes"]]
        )
        self._chunk_size = self.episodes_info["chunk_size"]
        self._guarantee_full_seqs = guarantee_full_seqs

    @property
    def num_episodes(self) -> int:
        return self._num_episodes

    @property
    def lengths(self) -> np.ndarray:
        return self._lengths

    def __getitem__(self, segment_id: SegmentId) -> Segment:
        episode_part = self.load_episode_for_segment(segment_id)
        segment = make_segment(episode_part, segment_id, self._guarantee_full_seqs)
        return segment

    def load_whole_episode(self, episode_id: int) -> EpisodePartition:
        episode = EpisodePartition.load_whole_episode(self.get_episode_path(episode_id))
        episode.obs = episode.obs.mul_(0.18215)
        return episode

    def load_episode_for_segment(self, segment_id: SegmentId) -> EpisodePartition:
        episode_part = EpisodePartition.load_for_segment(
            self.get_episode_path(segment_id.episode_id),
            segment_id,
            self.lengths[segment_id.episode_id],
            self._chunk_size,
        )
        episode_part.obs = episode_part.obs.mul_(0.18215)
        return episode_part

    def get_episode_path(self, episode_id: int) -> Path:

        return self._directory / f"episode_{episode_id}"


def collate_segments_to_batch(segments: list[Segment]) -> Batch:
    attrs = ("obs", "act")
    stack = (torch.stack([getattr(s, x) for s in segments]) for x in attrs)
    return Batch(*stack)


def make_segment(
    episode_part: EpisodePartition, segment_id: SegmentId, guarantee_full_seqs
) -> Segment:
    assert (
        segment_id.start < episode_part.episode_length
        and segment_id.stop > 0
        and segment_id.start < segment_id.stop
    )
    assert segment_id.stop <= episode_part.episode_length
    pad_len_left = max(0, -segment_id.start)
    if guarantee_full_seqs:
        assert pad_len_left == 0
        assert segment_id.start >= 0
        assert segment_id.stop <= episode_part.episode_length

    def pad(x):
        return (
            F.pad(x, [0 for _ in range(2 * x.ndim - 2)] + [pad_len_left, 0])
            if pad_len_left > 0
            else x
        )

    start = max(0, segment_id.start - episode_part.partition_start)
    stop = segment_id.stop - episode_part.partition_start
    obs = pad(episode_part.obs[start:stop])
    act = pad(episode_part.act[start:stop])
    assert obs.shape[0] == act.shape[0]
    assert obs.shape[0] == segment_id.stop - segment_id.start
    return Segment(
        obs,
        act,
    )


class TestDatasetTraverser:

    def __init__(
        self,
        dataset: Dataset,
        batch_num_samples: int,
        seq_length: int,
        subsample_rate: int,
    ) -> None:
        self.dataset = dataset
        self.batch_num_samples = batch_num_samples
        self.seq_length = seq_length
        self.subsample_rate = subsample_rate

    def __len__(self):
        return math.ceil(
            sum(
                [
                    len(
                        range(
                            0,
                            self.dataset.lengths[episode_id] - self.seq_length + 1,
                            self.subsample_rate,
                        )
                    )
                    for episode_id in range(self.dataset.num_episodes)
                ]
            )
            / self.batch_num_samples
        )

    def __iter__(self) -> Generator[Batch, None, None]:
        chunks = []
        for episode_id in range(self.dataset.num_episodes):
            episode = self.dataset.load_whole_episode(episode_id)
            for start in range(
                0, len(episode) - self.seq_length + 1, self.subsample_rate
            ):
                stop = start + self.seq_length
                segment = make_segment(
                    episode,
                    SegmentId(episode_id, start, stop),
                    guarantee_full_seqs=True,
                )
                chunks.append(segment)

            while len(chunks) >= self.batch_num_samples:
                yield collate_segments_to_batch(chunks[: self.batch_num_samples])
                chunks = chunks[self.batch_num_samples :]

        if len(chunks) > 0:
            yield collate_segments_to_batch(chunks)
