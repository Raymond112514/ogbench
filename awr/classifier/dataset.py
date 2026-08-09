"""Classifier dataset from oracle-annotated chunk rollouts."""

from __future__ import annotations

import numpy as np


def episode_ranges(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    starts = [0] + episode_ends[:-1].tolist()
    return list(zip(starts, episode_ends))


def build_samples(episode_ends, distance):
    indices, labels = [], []
    for start, end in episode_ranges(episode_ends):
        for t in range(start, end - 1):
            indices.append(t)
            labels.append(0.0 if distance[t] < distance[t + 1] else 1.0)
    return np.asarray(indices, np.int64), np.asarray(labels, np.float32)


def build_oracle_advantages(
    data: dict,
    mode: str = 'delta',
    num_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build AWR samples with oracle Δ or signed Δ-bin advantages.

    Annotated ``distance`` is at consecutive chunk boundaries, so
    Δ_t = d(s_t) - d(s_{t+H}).

    mode:
      - ``delta``: raw Δ (unbinned)
      - ``binned``: signed uniform bins of Δ over [-H, H] (odd num_bins)
    """
    from awr.oracle_utils import delta_bin_label

    if mode not in ('delta', 'binned'):
        raise ValueError(f'mode must be delta or binned, got {mode!r}')

    chunk_size = int(data['chunk_size']) if 'chunk_size' in data else 1
    distance = np.asarray(data['distance'])
    indices: list[int] = []
    advantages: list[float] = []
    for start, end in episode_ranges(data['episode_ends']):
        for t in range(start, end - 1):
            delta = float(distance[t] - distance[t + 1])
            if mode == 'delta':
                adv = delta
            else:
                adv = delta_bin_label(delta, chunk_size, num_bins)
            indices.append(t)
            advantages.append(adv)

    idx = np.asarray(indices, np.int64)
    adv = np.asarray(advantages, np.float32)
    observations = np.asarray(data['observations'][idx], np.float32)
    chunks = np.asarray(data['action_chunks'][idx], np.float32)
    actions = chunks.reshape(len(idx), -1)
    stats = {
        'num_chunks': int(len(adv)),
        'adv_mean': float(adv.mean()) if len(adv) else 0.0,
        'adv_std': float(adv.std()) if len(adv) else 0.0,
        'adv_min': float(adv.min()) if len(adv) else 0.0,
        'adv_max': float(adv.max()) if len(adv) else 0.0,
        'frac_positive': float(np.mean(adv > 0)) if len(adv) else 0.0,
        'mode': mode,
        'num_bins': int(num_bins) if mode == 'binned' else None,
    }
    return observations, actions, adv, stats


def merge_annotated(datas: list[dict]) -> dict:
    keys = [
        'observations', 'actions', 'next_observations', 'next_mjstate',
        'distance', 'action_chunks', 'chunk_masks',
    ]
    out = {}
    for k in keys:
        if k in datas[0]:
            out[k] = np.concatenate([d[k] for d in datas], axis=0)
    ends, offset = [], 0
    for d in datas:
        for e in d['episode_ends']:
            ends.append(int(e) + offset)
        offset = ends[-1]
    out['episode_ends'] = np.asarray(ends, np.int32)
    for k in ('goal_xyz', 'task_id', 'chunk_size', 'policy'):
        if k in datas[0]:
            out[k] = datas[0][k]
    return out


class ClassifierDataset:
    def __init__(self, data: dict | str):
        self.data = np.load(data, mmap_mode='r') if isinstance(data, str) else data
        self.episode_ends = self.data['episode_ends']
        self.indices, self.labels = build_samples(self.episode_ends, self.data['distance'])
        chunks = self.data['action_chunks']
        self.act_dim = chunks.shape[1] * chunks.shape[2]
        self.obs_dim = self.data['observations'].shape[1]
        self.chunk_size = int(self.data['chunk_size']) if 'chunk_size' in self.data else 1

    def __len__(self):
        return len(self.indices)

    def sample(self, batch_size, rng):
        sel = rng.integers(0, len(self), size=batch_size)
        idx = self.indices[sel]
        chunks = np.asarray(self.data['action_chunks'][idx], np.float32)
        return {
            'observations': np.asarray(self.data['observations'][idx], np.float32),
            'actions': chunks.reshape(len(idx), -1),
            'labels': self.labels[sel],
        }
