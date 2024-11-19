import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

from ..utils import StateDictMixin
from .episode import Episode
from .segment import Segment, SegmentId
from .utils import make_segment


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
        self._default_path = self._directory / "info.pt"
        self._cache = mp.Manager().dict() if use_manager else {}

    def __getitem__(self, segment_id: SegmentId) -> Segment:
        episode = self.load_episode(segment_id.episode_id)
        segment = make_segment(episode, segment_id, should_pad=True)
        return segment

    def load_episode(self, episode_id: int) -> Episode:
        if self._cache_in_ram and episode_id in self._cache:
            episode = self._cache[episode_id]
        else:
            episode = Episode.load(self._get_episode_path(episode_id))
            if self._cache_in_ram:
                self._cache[episode_id] = episode
        return episode

    def _get_episode_path(self, episode_id: int) -> Path:
        n = 3  # number of hierarchies
        powers = np.arange(n)
        subfolders = np.floor((episode_id % 10 ** (1 + powers)) / 10**powers) * 10**powers
        subfolders = [int(x) for x in subfolders[::-1]]
        subfolders = "/".join([f"{x:0{n - i}d}" for i, x in enumerate(subfolders)])
        return self._directory / subfolders / f"{episode_id}.pt"


class CSGOHdf5Dataset(StateDictMixin, TorchDataset):
    def __init__(self, directory: Path) -> None:
        super().__init__()
        filenames = sorted(Path(directory).rglob("*.hdf5"), key=lambda x: int(x.stem.split("_")[-1]))
        self._filenames = {f"{x.parent.name}/{x.name}": x for x in filenames}
        self._length_one_episode = 1000
        self.num_episodes = len(self._filenames)
        self.num_steps = self._length_one_episode * self.num_episodes
        self.lengths = np.array([self._length_one_episode] * self.num_episodes, dtype=np.int64)

    def __len__(self) -> int:
        return self.num_steps

    def save_to_default_path(self) -> None:
        pass

    def __getitem__(self, segment_id: SegmentId) -> Segment:
        assert segment_id.start < self._length_one_episode and segment_id.stop > 0 and segment_id.start < segment_id.stop
        pad_len_right = max(0, segment_id.stop - self._length_one_episode)
        pad_len_left = max(0, -segment_id.start)

        start = max(0, segment_id.start)
        stop = min(self._length_one_episode, segment_id.stop)
        mask_padding = torch.cat((torch.zeros(pad_len_left), torch.ones(stop - start), torch.zeros(pad_len_right))).bool()

        with h5py.File(self._filenames[segment_id.episode_id], "r") as f:
            obs = torch.stack([torch.tensor(f[f"frame_{i}_x"][:]).flip(2).permute(2, 0, 1).div(255).mul(2).sub(1) for i in range(start, stop)])
            act = torch.tensor(np.array([f[f"frame_{i}_y"][:] for i in range(start, stop)]))

        def pad(x):
            right = F.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [pad_len_right]) if pad_len_right > 0 else x
            return F.pad(right, [0 for _ in range(2 * x.ndim - 2)] + [pad_len_left, 0]) if pad_len_left > 0 else right

        obs = pad(obs)
        act = pad(act)
        rew = torch.zeros(obs.size(0))
        end = torch.zeros(obs.size(0), dtype=torch.uint8)
        trunc = torch.zeros(obs.size(0), dtype=torch.uint8)
        return Segment(
            obs,
            act,
            rew,
            end,
            trunc,
            mask_padding,
            id=SegmentId(segment_id.episode_id, start, stop),
        )

    def load_episode(self, episode_id: int) -> Episode:  # used by DatasetTraverser
        s = self[SegmentId(episode_id, 0, self._length_one_episode)]
        return Episode(s.obs, s.act, s.rew, s.end, s.trunc)
