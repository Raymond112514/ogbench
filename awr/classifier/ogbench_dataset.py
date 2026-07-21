"""Progress-classifier dataset from oracle-annotated OGBench transitions.

Requires an npz produced by awr/annotate_ogbench.py (ideally --data_percent 100 so that
`distance` is available contiguously for every transition within each episode).

Label for a chunk starting at t: 1.0 if s_{t+chunk_size} improved over s_t, i.e. the oracle
distance decreased (progress), else 0.0. Chunks that would cross an episode boundary are
dropped (no valid same-episode oracle label for s_{t+chunk_size}).
"""

from __future__ import annotations

import numpy as np


def episode_ends(terminals: np.ndarray, n: int) -> list[int]:
    ends = (np.where(np.asarray(terminals) > 0.5)[0] + 1).tolist()
    if not ends or ends[-1] != n:
        ends = ends + [n]
    return ends


def build_chunk_progress(
    observations: np.ndarray,
    actions: np.ndarray,
    terminals: np.ndarray,
    distance: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chunk-level (observation, flattened action chunk, progress label) samples."""
    n = len(observations)
    ends = episode_ends(terminals, n)
    obs_list, act_list, label_list = [], [], []
    start = 0
    for end in ends:
        t = start
        while t + chunk_size < end:  # need s_{t+chunk_size} in the same episode
            obs_list.append(observations[t])
            act_list.append(actions[t : t + chunk_size].reshape(-1))
            label_list.append(1.0 if distance[t + chunk_size] < distance[t] else 0.0)
            t += chunk_size
        start = end
    return (
        np.asarray(obs_list, np.float32),
        np.asarray(act_list, np.float32),
        np.asarray(label_list, np.float32),
    )


class ClassifierOgbenchDataset:
    """Progress-classifier dataset (chunked oracle-improvement labels)."""

    def __init__(self, observations: np.ndarray, actions: np.ndarray, labels: np.ndarray):
        self.data = {'observations': observations, 'actions': actions, 'labels': labels}
        self.size = len(observations)
        self.obs_dim = observations.shape[1]
        self.act_dim = actions.shape[1]

    @classmethod
    def from_annotated(cls, annotated: dict, chunk_size: int = 4) -> 'ClassifierOgbenchDataset':
        for k in ('observations', 'actions', 'terminals', 'distance'):
            if k not in annotated:
                raise ValueError(f'oracle-annotated data missing key {k!r}')
        obs, act, labels = build_chunk_progress(
            np.asarray(annotated['observations'], np.float32),
            np.asarray(annotated['actions'], np.float32),
            np.asarray(annotated['terminals']).reshape(-1),
            np.asarray(annotated['distance']),
            chunk_size,
        )
        if len(obs) == 0:
            raise ValueError('no chunk-progress samples built; check chunk_size vs. episode length')
        return cls(obs, act, labels)

    def add(self, observations: np.ndarray, actions: np.ndarray, labels: np.ndarray) -> None:
        """Append new chunk samples into this buffer in place (for online data collection)."""
        self.data['observations'] = np.concatenate([self.data['observations'], observations], axis=0)
        self.data['actions'] = np.concatenate([self.data['actions'], actions], axis=0)
        self.data['labels'] = np.concatenate([self.data['labels'], labels], axis=0)
        self.size = len(self.data['observations'])

    def subsample(self, percent: float, seed: int = 0) -> 'ClassifierOgbenchDataset':
        """Keep a random `percent`% of chunk samples (1–100). Deterministic given seed."""
        if not (0 < percent <= 100):
            raise ValueError(f'data_percent must be in (0, 100], got {percent}')
        if percent >= 100:
            return self
        n = max(1, int(round(self.size * (percent / 100.0))))
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(self.size, size=n, replace=False))
        return ClassifierOgbenchDataset(
            self.data['observations'][idx], self.data['actions'][idx], self.data['labels'][idx]
        )

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        idx = rng.integers(0, self.size, size=batch_size)
        return {k: v[idx] for k, v in self.data.items()}
