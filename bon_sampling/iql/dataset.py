"""IQL transition dataset from raw rollouts (env success, no oracle)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def episode_ranges(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    starts = [0] + episode_ends[:-1].tolist()
    return list(zip(starts, episode_ends))


def transitions_from_data(data: dict) -> dict[str, np.ndarray]:
    """Build (s, a_chunk, r, s', mask) from an in-memory rollout dict."""
    if 'successes' not in data:
        raise ValueError('missing per-step successes; re-collect rollouts')

    observations = np.asarray(data['observations'], np.float32)
    actions = np.asarray(data['actions'], np.float32)
    successes = np.asarray(data['successes'], bool)
    episode_ends = np.asarray(data['episode_ends'], np.int32)
    chunk_size = int(np.asarray(data.get('chunk_size', 1)))
    act_dim = actions.shape[-1]

    obs_list, act_list, next_list, rew_list, mask_list = [], [], [], [], []
    for start, end in episode_ranges(episode_ends):
        t = start
        while t < end:
            chunk_end = min(t + chunk_size, end)
            chunk = np.zeros((chunk_size, act_dim), np.float32)
            chunk[: chunk_end - t] = actions[t:chunk_end]
            success = bool(np.any(successes[t:chunk_end]))
            if chunk_end < end:
                next_obs = observations[chunk_end]
            else:
                next_obs = np.asarray(data['next_observations'][chunk_end - 1], np.float32)
            obs_list.append(observations[t])
            act_list.append(chunk.reshape(-1))
            next_list.append(next_obs)
            rew_list.append(0.0 if success else -1.0)
            mask_list.append(0.0 if success else 1.0)
            t = chunk_end

    return {
        'observations': np.asarray(obs_list, np.float32),
        'actions': np.asarray(act_list, np.float32),
        'next_observations': np.asarray(next_list, np.float32),
        'rewards': np.asarray(rew_list, np.float32),
        'masks': np.asarray(mask_list, np.float32),
    }


def transitions_from_rollouts(path: str | Path) -> dict[str, np.ndarray]:
    return transitions_from_data(dict(np.load(path, allow_pickle=False)))


def merge_transitions(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


class IQLDataset:
    def __init__(self, data: dict[str, np.ndarray]):
        self.data = data
        self.size = len(data['observations'])

    @classmethod
    def from_paths(cls, paths: list[str | Path]):
        return cls(merge_transitions([transitions_from_rollouts(p) for p in paths]))

    def add(self, new_data: dict[str, np.ndarray]) -> None:
        """Append new transitions into this buffer in place."""
        self.data = {k: np.concatenate([self.data[k], new_data[k]], axis=0) for k in self.data}
        self.size = len(self.data['observations'])

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        idx = rng.integers(0, self.size, size=batch_size)
        return {k: v[idx] for k, v in self.data.items()}
