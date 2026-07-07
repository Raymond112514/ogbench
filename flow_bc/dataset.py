"""Chunked datasets for flow BC with action chunking."""

from __future__ import annotations

import dataclasses

import numpy as np


def terminals_from_episode_ends(episode_ends: np.ndarray, n_steps: int) -> np.ndarray:
    terminals = np.zeros(n_steps, dtype=np.float32)
    terminals[np.asarray(episode_ends, dtype=np.int64) - 1] = 1.0
    return terminals


def load_npz_as_trajectories(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load observations/actions/terminals from npz (OGBench or bon_sampling format)."""
    data = np.load(path, allow_pickle=False)
    observations = np.asarray(data['observations'], dtype=np.float32)
    actions = np.asarray(data['actions'], dtype=np.float32)
    if 'terminals' in data:
        terminals = np.asarray(data['terminals'], dtype=np.float32)
    elif 'episode_ends' in data:
        terminals = terminals_from_episode_ends(data['episode_ends'], len(observations))
    else:
        raise ValueError(f'{path} must contain terminals or episode_ends')
    return observations, actions, terminals


@dataclasses.dataclass
class ChunkedGCDataset:
    """Samples (s_t, g, action_chunk, mask) for goal-conditioned flow BC."""

    observations: np.ndarray
    actions: np.ndarray
    terminals: np.ndarray
    chunk_size: int = 4
    seed: int = 0

    def __post_init__(self):
        (terminal_locs,) = np.nonzero(self.terminals > 0)
        self.terminal_locs = terminal_locs
        all_idxs = np.arange(len(self.observations))
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, all_idxs)]
        self.valid_idxs = all_idxs[all_idxs < final_state_idxs]

        self.rng = np.random.default_rng(self.seed)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idxs = self.valid_idxs[self.rng.integers(0, len(self.valid_idxs), size=batch_size)]
        return self._make_batch(idxs)

    def _make_batch(self, idxs: np.ndarray) -> dict[str, np.ndarray]:
        B = len(idxs)
        act_dim = self.actions.shape[-1]
        H = self.chunk_size
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        chunks = np.zeros((B, H, act_dim), dtype=np.float32)
        masks = np.zeros((B, H), dtype=np.float32)
        for k in range(H):
            src = np.minimum(idxs + k, final_state_idxs)
            valid = (idxs + k) <= final_state_idxs
            chunks[:, k, :] = self.actions[src]
            masks[:, k] = valid.astype(np.float32)

        lo = np.minimum(idxs + 1, final_state_idxs)
        hi = final_state_idxs
        offsets = (self.rng.random(B) * (hi - lo + 1)).astype(int)
        goal_idxs = np.clip(lo + offsets, lo, hi)
        goals = self.observations[goal_idxs]

        return {
            'observations': self.observations[idxs].astype(np.float32),
            'goals': goals.astype(np.float32),
            'action_chunks': chunks,
            'chunk_masks': masks,
        }


@dataclasses.dataclass
class ChunkedDataset:
    """Samples (s_t, action_chunk, mask) without goal conditioning."""

    observations: np.ndarray
    actions: np.ndarray
    terminals: np.ndarray
    chunk_size: int = 4
    seed: int = 0

    def __post_init__(self):
        (terminal_locs,) = np.nonzero(self.terminals > 0)
        self.terminal_locs = terminal_locs
        all_idxs = np.arange(len(self.observations))
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, all_idxs)]
        self.valid_idxs = all_idxs[all_idxs < final_state_idxs]
        self.rng = np.random.default_rng(self.seed)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idxs = self.valid_idxs[self.rng.integers(0, len(self.valid_idxs), size=batch_size)]
        return self._make_batch(idxs)

    def _make_batch(self, idxs: np.ndarray) -> dict[str, np.ndarray]:
        B = len(idxs)
        act_dim = self.actions.shape[-1]
        H = self.chunk_size
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        chunks = np.zeros((B, H, act_dim), dtype=np.float32)
        masks = np.zeros((B, H), dtype=np.float32)
        for k in range(H):
            src = np.minimum(idxs + k, final_state_idxs)
            valid = (idxs + k) <= final_state_idxs
            chunks[:, k, :] = self.actions[src]
            masks[:, k] = valid.astype(np.float32)

        return {
            'observations': self.observations[idxs].astype(np.float32),
            'action_chunks': chunks,
            'chunk_masks': masks,
        }


def load_npz_dataset(path: str, chunk_size: int, seed: int, goal_condition: bool):
    observations, actions, terminals = load_npz_as_trajectories(path)
    if goal_condition:
        ds = ChunkedGCDataset(
            observations=observations,
            actions=actions,
            terminals=terminals,
            chunk_size=chunk_size,
            seed=seed,
        )
    else:
        ds = ChunkedDataset(
            observations=observations,
            actions=actions,
            terminals=terminals,
            chunk_size=chunk_size,
            seed=seed,
        )
    print(
        f'npz dataset {path}: {len(ds.valid_idxs)} valid steps, '
        f'obs_dim={observations.shape[1]}, act_dim={actions.shape[1]}, goal_condition={goal_condition}'
    )
    return ds
