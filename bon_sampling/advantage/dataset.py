from __future__ import annotations

from pathlib import Path

import numpy as np


def episode_ranges(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    starts = [0] + episode_ends[:-1].tolist()
    return list(zip(starts, episode_ends))


def is_chunk_data(data) -> bool:
    return 'action_chunks' in data


def value_target(d: np.ndarray, d_prime: np.ndarray, gamma: float) -> np.ndarray:
    """Q(s, a) = (gamma^{d'} - gamma^{d-1}) / (1 - gamma)."""
    return (np.power(gamma, d_prime) - np.power(gamma, d - 1.0)) / (1.0 - gamma)


def build_samples(
    episode_ends: np.ndarray,
    distance: np.ndarray,
    chunk_mode: bool,
    horizon: int = 1,
    tau: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (index, label) pairs.

    y=1 iff d(s) - d(s') >= H - tau. Default tau = H - 1 ⇒ threshold 1.

    chunk_mode: consecutive distance entries are chunk-boundary states (H = horizon).
    per_step (legacy): consecutive steps (H = 1).
    """
    from awr.oracle_utils import progress_label

    h = int(horizon) if chunk_mode else 1
    indices = []
    labels = []
    for start, end in episode_ranges(episode_ends):
        if chunk_mode:
            for t in range(start, end - 1):
                d_t = distance[t]
                d_tp1 = distance[t + 1]
                indices.append(t)
                labels.append(progress_label(d_t, d_tp1, h, tau))
        else:
            for t in range(start + 1, end):
                d_t = distance[t - 1]
                d_tp1 = distance[t]
                indices.append(t)
                labels.append(progress_label(d_t, d_tp1, 1, tau))
    return np.asarray(indices, np.int64), np.asarray(labels, np.float32)


def build_bin_samples(
    episode_ends: np.ndarray,
    distance: np.ndarray,
    chunk_mode: bool,
    horizon: int,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (index, class) pairs: class = Δ-bin index in {0, ..., w-1}."""
    from awr.oracle_utils import delta_bin_index

    h = int(horizon) if chunk_mode else 1
    indices = []
    labels = []
    for start, end in episode_ranges(episode_ends):
        if chunk_mode:
            for t in range(start, end - 1):
                indices.append(t)
                labels.append(delta_bin_index(float(distance[t] - distance[t + 1]), h, num_bins))
        else:
            for t in range(start + 1, end):
                indices.append(t)
                labels.append(delta_bin_index(float(distance[t - 1] - distance[t]), 1, num_bins))
    return np.asarray(indices, np.int64), np.asarray(labels, np.int32)


def build_regression_samples(
    episode_ends: np.ndarray,
    distance: np.ndarray,
    chunk_mode: bool,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (index, Q-target) pairs for consecutive in-episode transitions."""
    indices = []
    d_list, dp_list = [], []
    for start, end in episode_ranges(episode_ends):
        if chunk_mode:
            for t in range(start, end - 1):
                indices.append(t)
                d_list.append(distance[t])
                dp_list.append(distance[t + 1])
        else:
            for t in range(start + 1, end):
                indices.append(t)
                d_list.append(distance[t - 1])
                dp_list.append(distance[t])
    targets = value_target(
        np.asarray(d_list, np.float64),
        np.asarray(dp_list, np.float64),
        gamma,
    )
    return np.asarray(indices, np.int64), targets.astype(np.float32)


def split_episodes(num_episodes: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_episodes)
    n_val = max(1, int(num_episodes * val_ratio))
    val_eps = np.sort(perm[:n_val])
    train_eps = np.sort(perm[n_val:])
    return train_eps, val_eps


def mask_for_episodes(episode_ends: np.ndarray, episode_ids: np.ndarray, chunk_mode: bool) -> np.ndarray:
    ranges = episode_ranges(episode_ends)
    mask = np.zeros(len(episode_ends), dtype=bool)
    for ep in episode_ids:
        mask[ep] = True
    out = []
    for ep, (start, end) in enumerate(ranges):
        if not mask[ep]:
            continue
        if chunk_mode:
            for t in range(start, end - 1):
                out.append(t)
        else:
            for t in range(start + 1, end):
                out.append(t)
    return np.asarray(out, np.int64)


class AdvantageDataset:
    def __init__(
        self,
        path: str,
        episode_ids: np.ndarray | None = None,
        task: str = 'classifier',
        gamma: float = 0.99,
        tau: int | None = None,
        num_bins: int = 5,
    ):
        self.data = np.load(path, mmap_mode='r')
        self.observations = self.data['observations']
        self.distance = self.data['distance']
        self.episode_ends = self.data['episode_ends']
        self.chunk_mode = is_chunk_data(self.data)
        self.chunk_size = int(self.data['chunk_size']) if 'chunk_size' in self.data else 1
        self.task = task
        self.gamma = gamma
        self.tau = self.chunk_size - 1 if tau is None else int(tau)
        self.num_bins = int(num_bins)

        if self.chunk_mode:
            chunks = self.data['action_chunks']
            self.act_dim = chunks.shape[1] * chunks.shape[2]
        else:
            self.act_dim = self.data['actions'].shape[1]

        if task == 'classifier':
            all_indices, all_labels = build_samples(
                self.episode_ends, self.distance, self.chunk_mode,
                horizon=self.chunk_size, tau=self.tau,
            )
        elif task == 'bin_classifier':
            all_indices, all_labels = build_bin_samples(
                self.episode_ends, self.distance, self.chunk_mode,
                horizon=self.chunk_size, num_bins=self.num_bins,
            )
        elif task == 'regression':
            all_indices, all_labels = build_regression_samples(
                self.episode_ends, self.distance, self.chunk_mode, gamma
            )
        else:
            raise ValueError(
                f'unknown task={task!r}; expected classifier, bin_classifier, or regression'
            )

        if episode_ids is not None:
            allowed = set(mask_for_episodes(self.episode_ends, episode_ids, self.chunk_mode).tolist())
            keep = np.array([i in allowed for i in all_indices])
            self.indices = all_indices[keep]
            self.labels = all_labels[keep]
        else:
            self.indices = all_indices
            self.labels = all_labels

    def __len__(self) -> int:
        return len(self.indices)

    def get_batch(self, sel: np.ndarray) -> dict[str, np.ndarray]:
        idx = self.indices[sel]
        if self.chunk_mode:
            chunks = np.asarray(self.data['action_chunks'][idx], dtype=np.float32)
            actions = chunks.reshape(len(idx), -1)
        else:
            actions = np.asarray(self.data['actions'][idx], dtype=np.float32)
        return {
            'observations': np.asarray(self.observations[idx], dtype=np.float32),
            'actions': actions,
            'labels': self.labels[sel],
        }

    def label_balance(self) -> tuple[float, float]:
        if len(self.labels) == 0:
            return 0.0, 0.0
        return float(np.mean(self.labels == 0)), float(np.mean(self.labels == 1))

    def class_mass(self) -> np.ndarray:
        w = int(self.num_bins)
        if len(self.labels) == 0:
            return np.zeros(w, dtype=np.float64)
        counts = np.bincount(np.asarray(self.labels, np.int64), minlength=w).astype(np.float64)
        return counts / counts.sum()


def make_train_val(
    path: str,
    val_ratio: float,
    seed: int,
    task: str = 'classifier',
    gamma: float = 0.99,
    tau: int | None = None,
    num_bins: int = 5,
) -> tuple[AdvantageDataset, AdvantageDataset]:
    data = np.load(path, allow_pickle=False)
    num_episodes = len(data['episode_ends'])
    train_eps, val_eps = split_episodes(num_episodes, val_ratio, seed)
    return (
        AdvantageDataset(
            path, train_eps, task=task, gamma=gamma, tau=tau, num_bins=num_bins,
        ),
        AdvantageDataset(
            path, val_eps, task=task, gamma=gamma, tau=tau, num_bins=num_bins,
        ),
    )


def merge_annotated(paths: list[str], out_path: str) -> str:
    """Concatenate annotated rollouts; rebuild episode_ends offsets."""
    datas = [dict(np.load(p, allow_pickle=False)) for p in paths]
    keys = [
        'observations', 'actions', 'next_observations', 'next_mjstate',
        'distance', 'action_chunks', 'chunk_masks', 'chunk_boundary_indices',
    ]
    out = {}
    for k in keys:
        if k in datas[0]:
            out[k] = np.concatenate([d[k] for d in datas], axis=0)

    ends = []
    offset = 0
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
