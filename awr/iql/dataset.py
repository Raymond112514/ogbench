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


def concat_rollouts(rollouts: list[dict]) -> dict:
    """Concatenate raw rollout dicts (offset episode_ends)."""
    keys = [
        'observations', 'actions', 'next_observations', 'next_mjstate',
        'successes', 'episode_initial_mjstate',
    ]
    out = {k: np.concatenate([np.asarray(r[k]) for r in rollouts], axis=0) for k in keys if k in rollouts[0]}
    ends, offset = [], 0
    for r in rollouts:
        for e in r['episode_ends']:
            ends.append(int(e) + offset)
        offset = ends[-1]
    out['episode_ends'] = np.asarray(ends, np.int32)
    for k in ('goal_xyz', 'task_id', 'chunk_size'):
        if k in rollouts[0]:
            out[k] = rollouts[0][k]
    return out


class IQLDataset:
    def __init__(self, data: dict[str, np.ndarray]):
        self.data = data
        self.size = len(data['observations'])

    @classmethod
    def from_rollouts(cls, rollouts: list[dict]):
        parts = [transitions_from_data(r) for r in rollouts]
        return cls(merge_transitions(parts))

    @classmethod
    def from_paths(cls, paths: list[str | Path]):
        return cls(merge_transitions([transitions_from_rollouts(p) for p in paths]))

    @classmethod
    def from_ogbench(cls, train_dataset: dict, action_clip_eps: float = 1e-5):
        """Build from OGBench/FQL singletask dataset dict (same data FQL uses)."""
        keys = ('observations', 'actions', 'next_observations', 'rewards', 'masks')
        for k in keys:
            if k not in train_dataset:
                raise ValueError(f'OGBench dataset missing key {k!r}')
        actions = np.asarray(train_dataset['actions'], np.float32)
        if action_clip_eps is not None:
            actions = np.clip(actions, -1.0 + action_clip_eps, 1.0 - action_clip_eps)
        data = {
            'observations': np.asarray(train_dataset['observations'], np.float32),
            'actions': actions,
            'next_observations': np.asarray(train_dataset['next_observations'], np.float32),
            'rewards': np.asarray(train_dataset['rewards'], np.float32).reshape(-1),
            'masks': np.asarray(train_dataset['masks'], np.float32).reshape(-1),
        }
        return cls(data)

    def subsample(self, percent: float, seed: int = 0) -> 'IQLDataset':
        """Keep a random `percent`% of transitions (1–100). Deterministic given seed."""
        if not (0 < percent <= 100):
            raise ValueError(f'data_percent must be in (0, 100], got {percent}')
        if percent >= 100:
            return self
        n = max(1, int(round(self.size * (percent / 100.0))))
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(self.size, size=n, replace=False))
        return IQLDataset({k: v[idx] for k, v in self.data.items()})

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        idx = rng.integers(0, self.size, size=batch_size)
        return {k: v[idx] for k, v in self.data.items()}
