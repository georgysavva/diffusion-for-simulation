from typing import Generator, List, Optional

import numpy as np
import torch

from .dataset import CSGOHdf5Dataset, Dataset
from .segment import SegmentId


class BatchSampler(torch.utils.data.Sampler):

    def __init__(
        self,
        dataset: Dataset,
        rank: int,
        world_size: int,
        batch_size: int,
        seq_length: int,
    ) -> None:
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.seq_length = seq_length

    def __len__(self):
        raise NotImplementedError

    def __iter__(self) -> Generator[List[SegmentId], None, None]:
        while True:
            yield self.sample()

    def sample(self) -> List[SegmentId]:
        num_episodes = self.dataset.num_episodes

        episodes_partition = np.arange(self.rank, num_episodes, self.world_size)
        max_eps = self.batch_size
        episode_ids = np.random.choice(episodes_partition, size=max_eps, replace=True)
        episode_ids = episode_ids.repeat(self.batch_size // max_eps)
        timesteps = np.random.randint(low=0, high=self.dataset.lengths[episode_ids])

        stops = np.minimum(
            self.dataset.lengths[episode_ids],
            timesteps + 1 + np.random.randint(0, self.seq_length, len(timesteps)),
        )
        starts = stops - self.seq_length

        return [SegmentId(*x) for x in zip(episode_ids, starts, stops)]
