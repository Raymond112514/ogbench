"""Classifier dataset from oracle-annotated chunk rollouts."""

from __future__ import annotations

from pathlib import Path

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


def merge_annotated(paths: list[str | Path], out_path: str) -> str:
    datas = [dict(np.load(p, allow_pickle=False)) for p in paths]
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
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    return out_path


class ClassifierDataset:
    def __init__(self, path: str):
        self.data = np.load(path, mmap_mode='r')
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
