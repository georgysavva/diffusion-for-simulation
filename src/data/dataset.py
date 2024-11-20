import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset as TorchDataset

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
        self._cache = mp.Manager().dict() if use_manager else {}
        with open(self._directory / "episodes_info.json", "r") as json_file:
            episodes_info = json.load(json_file)
        self._num_episodes = episodes_info["episodes_num"]
        self._lengths = np.array([ep["length"] for ep in episodes_info["episodes"]])

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
            if self._cache_in_ram:
                self._cache[episode_id] = episode
        return episode

    def _get_episode_path(self, episode_id: int) -> Path:

        return self._directory / f"episode_{episode_id}.pt"
