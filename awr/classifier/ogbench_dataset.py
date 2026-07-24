"""Progress-classifier dataset from oracle-annotated OGBench transitions.

Requires an npz produced by awr/annotate_ogbench.py (ideally --data_percent 100 so that
`distance` is available contiguously for every transition within each episode).

Label for a chunk starting at t: 1.0 if d(s_t) - d(s_{t+H}) >= H - tau (default tau=H-1 ⇒
threshold 1), else 0.0. Chunks that would cross an episode boundary are dropped.
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
    tau: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chunk-level (observation, flattened action chunk, progress label) samples."""
    from awr.oracle_utils import progress_label

    n = len(observations)
    ends = episode_ends(terminals, n)
    obs_list, act_list, label_list = [], [], []
    start = 0
    for end in ends:
        t = start
        while t + chunk_size < end:  # need s_{t+chunk_size} in the same episode
            obs_list.append(observations[t])
            act_list.append(actions[t : t + chunk_size].reshape(-1))
            label_list.append(
                progress_label(distance[t], distance[t + chunk_size], chunk_size, tau)
            )
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
    def from_annotated(
        cls, annotated: dict, chunk_size: int = 4, tau: int | None = None,
    ) -> 'ClassifierOgbenchDataset':
        for k in ('observations', 'actions', 'terminals', 'distance'):
            if k not in annotated:
                raise ValueError(f'oracle-annotated data missing key {k!r}')
        obs, act, labels = build_chunk_progress(
            np.asarray(annotated['observations'], np.float32),
            np.asarray(annotated['actions'], np.float32),
            np.asarray(annotated['terminals']).reshape(-1),
            np.asarray(annotated['distance']),
            chunk_size,
            tau=tau,
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

    def split_train_val(
        self, val_ratio: float, seed: int = 0
    ) -> tuple['ClassifierOgbenchDataset', 'ClassifierOgbenchDataset']:
        """Random train/val split of chunk samples (deterministic given seed)."""
        if not (0.0 <= val_ratio < 1.0):
            raise ValueError(f'val_ratio must be in [0, 1), got {val_ratio}')
        if val_ratio == 0.0 or self.size < 2:
            empty = ClassifierOgbenchDataset(
                self.data['observations'][:0], self.data['actions'][:0], self.data['labels'][:0]
            )
            return self, empty
        n_val = max(1, int(round(self.size * val_ratio)))
        n_val = min(n_val, self.size - 1)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.size)
        val_idx = np.sort(perm[:n_val])
        train_idx = np.sort(perm[n_val:])
        train = ClassifierOgbenchDataset(
            self.data['observations'][train_idx],
            self.data['actions'][train_idx],
            self.data['labels'][train_idx],
        )
        val = ClassifierOgbenchDataset(
            self.data['observations'][val_idx],
            self.data['actions'][val_idx],
            self.data['labels'][val_idx],
        )
        return train, val

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        idx = rng.integers(0, self.size, size=batch_size)
        return {k: v[idx] for k, v in self.data.items()}
