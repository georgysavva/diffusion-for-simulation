from typing import Generator, List, Optional

import numpy as np
import torch

from .dataset import Dataset
from .segment import SegmentId


class BatchSampler(torch.utils.data.Sampler):

    def __init__(
        self,
        dataset: Dataset,
        rank: int,
        world_size: int,
        batch_size: int,
        seed_seq_length: int,
        num_conditioning_steps: int,
        auto_regressive_length: int,
    ) -> None:
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.auto_regressive_length = auto_regressive_length
        self.seed_seq_length = seed_seq_length
        self.num_conditioning_steps = num_conditioning_steps

    def __len__(self):
        raise NotImplementedError

    def __iter__(self) -> Generator[List[SegmentId], None, None]:
        while True:
            yield self.sample()

    def sample(self) -> List[SegmentId]:
        num_episodes = self.dataset.num_episodes

        episodes_partition = np.arange(self.rank, num_episodes, self.world_size)
        short_episode_ids = np.where(
            self.dataset.lengths < self.seed_seq_length + self.auto_regressive_length
        )[0]
        episodes_partition = episodes_partition[
            ~np.isin(episodes_partition, short_episode_ids)
        ]
        episode_ids = np.random.choice(
            episodes_partition, size=self.batch_size, replace=True
        )
        stops = np.random.randint(
            low=self.seed_seq_length + self.auto_regressive_length,
            high=self.dataset.lengths[episode_ids] + 1,
        )
        starts = stops - (self.auto_regressive_length + self.num_conditioning_steps)

        return [SegmentId(*x) for x in zip(episode_ids, starts, stops)]
