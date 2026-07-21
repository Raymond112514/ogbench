"""
Annotate transitions with oracle distance

python bon_sampling/annotate_oracle_distance.py \
  --input bon_sampling/data/flowbc_rollouts_1000.npz \
  --output bon_sampling/data/flowbc_rollouts_1000_annotated.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import gymnasium
import numpy as np

import ogbench.manipspace  # noqa: F401
from bon_sampling.oracle_utils import oracle_distance
from bon_sampling.sim_state import get_sim_state
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle


def episode_ranges(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    starts = [0] + episode_ends[:-1].tolist()
    return list(zip(starts, episode_ends))


def chunk_boundary_indices(episode_ends: np.ndarray, chunk_size: int) -> np.ndarray:
    """Global indices of replan states: episode_start + k * chunk_size."""
    indices = []
    for start, end in episode_ranges(episode_ends):
        t = start
        while t < end:
            indices.append(t)
            t += chunk_size
    return np.asarray(indices, np.int64)


def episode_start_to_reset_seed(
    episode_ends: np.ndarray,
    episode_reset_seeds: np.ndarray | None,
    default_reset_seed: int,
) -> dict[int, int]:
    """Map global episode start index -> env reset seed."""
    out = {}
    starts = [0] + episode_ends[:-1].tolist()
    for ep_idx, start in enumerate(starts):
        if episode_reset_seeds is not None and ep_idx < len(episode_reset_seeds):
            out[int(start)] = int(episode_reset_seeds[ep_idx])
        else:
            out[int(start)] = default_reset_seed
    return out


def episode_start_to_ep_index(episode_ends: np.ndarray) -> dict[int, int]:
    out = {}
    starts = [0] + episode_ends[:-1].tolist()
    for ep_idx, start in enumerate(starts):
        out[int(start)] = ep_idx
    return out


def mjstate_at_obs(
    data,
    t: int,
    ep_start: int,
    ep_idx: int,
    policy_env,
    task_id: int,
    reset_seed: int,
):
    """MuJoCo state at observation index t (start of a chunk)."""
    if t > ep_start:
        return data['next_mjstate'][t - 1]
    if 'episode_initial_mjstate' in data:
        return np.asarray(data['episode_initial_mjstate'][ep_idx], dtype=np.float64)
    policy_env.reset(options=dict(task_id=task_id), seed=reset_seed)
    u = policy_env.unwrapped
    return get_sim_state(u._model, u._data)


def build_action_chunks(
    actions: np.ndarray,
    episode_ends: np.ndarray,
    chunk_size: int,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack actions[t:t+H] for each chunk-boundary index t."""
    act_dim = actions.shape[-1]
    chunks = np.zeros((len(indices), chunk_size, act_dim), dtype=np.float32)
    masks = np.zeros((len(indices), chunk_size), dtype=np.float32)
    ranges = dict(episode_ranges(episode_ends))
    # map global index -> episode end
    ep_bounds = {}
    for start, end in episode_ranges(episode_ends):
        for t in range(start, end):
            ep_bounds[t] = end

    for i, t in enumerate(indices):
        end = ep_bounds[t]
        for k in range(chunk_size):
            if t + k < end:
                chunks[i, k] = actions[t + k]
                masks[i, k] = 1.0
    return chunks, masks


def annotate_indices(
    worker_id: int,
    input_path: str,
    indices: np.ndarray,
    task_id: int,
    default_reset_seed: int,
    max_oracle_steps: int,
    warmup_steps: int,
    log_every: int,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(input_path, mmap_mode='r')
    episode_ends = data['episode_ends']
    episode_reset_seeds = (
        np.asarray(data['episode_reset_seeds'], dtype=np.int32)
        if 'episode_reset_seeds' in data
        else None
    )
    reset_seed_for_ep_start = episode_start_to_reset_seed(
        episode_ends, episode_reset_seeds, default_reset_seed
    )
    ep_index_for_start = episode_start_to_ep_index(episode_ends)
    ep_start_for = {}
    for start, end in episode_ranges(episode_ends):
        for t in range(start, end):
            ep_start_for[t] = start

    policy_env = gymnasium.make('cube-single-v0')
    oracle_env = gymnasium.make('cube-single-v0', mode='data_collection', terminate_at_goal=True)
    oracle_env.reset()
    oracle = CubeMarkovOracle(env=oracle_env)

    distances = np.zeros(len(indices), np.int32)
    for j, t in enumerate(indices):
        ep_start = ep_start_for[int(t)]
        ep_reset_seed = reset_seed_for_ep_start[ep_start]
        ep_idx = ep_index_for_start[ep_start]
        mjstate = mjstate_at_obs(
            data, int(t), ep_start, ep_idx, policy_env, task_id, ep_reset_seed
        )
        distances[j] = oracle_distance(
            oracle_env,
            oracle,
            mjstate,
            max_oracle_steps,
            warmup_steps=warmup_steps,
        )
        if log_every > 0 and ((j + 1) % log_every == 0 or j == 0):
            print(f'worker {worker_id}: {j + 1}/{len(indices)} t={t} d(s_t)={distances[j]}', flush=True)

    policy_env.close()
    oracle_env.close()
    print(f'worker {worker_id}: finished {len(indices)} chunk states', flush=True)
    return indices, distances


def subsample_arrays(data: dict, indices: np.ndarray, chunk_size: int) -> dict:
    """Keep only chunk-boundary rows; rebuild episode_ends."""
    episode_ends = data['episode_ends']

    new_episode_ends = []
    count = 0
    for start, end in episode_ranges(episode_ends):
        ep_count = len(range(start, end, chunk_size))
        count += ep_count
        new_episode_ends.append(count)

    out = {
        'observations': data['observations'][indices],
        'actions': data['actions'][indices],
        'next_observations': data['next_observations'][indices],
        'next_mjstate': data['next_mjstate'][indices],
        'episode_ends': np.asarray(new_episode_ends, np.int32),
        'chunk_boundary_indices': indices,
    }
    for key in ('goal_xyz', 'task_id', 'chunk_size', 'policy', 'reset_seed', 'episode_reset_seeds', 'episode_initial_mjstate'):
        if key in data:
            out[key] = data[key]

    action_chunks, chunk_masks = build_action_chunks(
        data['actions'], episode_ends, chunk_size, indices
    )
    out['action_chunks'] = action_chunks
    out['chunk_masks'] = chunk_masks
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='bon_sampling/data/bc_transitions.npz')
    p.add_argument('--output', default='bon_sampling/data/bc_transitions_annotated.npz')
    p.add_argument('--chunk_only', action='store_true', default=None,
                   help='Annotate chunk-boundary states only (default: on if chunk_size>1 in file)')
    p.add_argument('--chunk_size', type=int, default=None, help='Override chunk_size from file')
    p.add_argument('--reset_seed', type=int, default=None, help='Env reset seed for episode starts')
    p.add_argument('--max_oracle_steps', type=int, default=200)
    p.add_argument('--warmup_steps', type=int, default=2)
    p.add_argument('--num_workers', type=int, default=20)
    p.add_argument('--log_every', type=int, default=100)
    args = p.parse_args()

    data = dict(np.load(args.input, allow_pickle=False))
    if 'next_mjstate' not in data:
        raise SystemExit('missing next_mjstate; re-run collect_bc_transitions.py')

    chunk_size = int(args.chunk_size or data.get('chunk_size', 1))
    chunk_only = args.chunk_only if args.chunk_only is not None else (chunk_size > 1)
    task_id = int(data.get('task_id', 1))
    default_reset_seed = int(
        args.reset_seed if args.reset_seed is not None else data.get('reset_seed', 0)
    )
    if 'episode_initial_mjstate' in data:
        print(f'using stored episode_initial_mjstate ({len(data["episode_initial_mjstate"])} episodes)')
    elif 'episode_reset_seeds' in data:
        print(f'using per-episode reset seeds ({len(data["episode_reset_seeds"])} episodes)')
    else:
        print(f'using single reset_seed={default_reset_seed} for all episode starts')

    n = len(data['observations'])
    input_path = str(Path(args.input).resolve())

    if chunk_only:
        indices = chunk_boundary_indices(data['episode_ends'], chunk_size)
        print(
            f'chunk_only: {len(indices)} states from {n} transitions '
            f'(chunk_size={chunk_size}, ~{len(indices) / len(data["episode_ends"]):.1f} per episode)'
        )
    else:
        indices = np.arange(n, dtype=np.int64)
        print(f'annotating all {n} transitions')

    worker_chunks = np.array_split(indices, args.num_workers)
    dist_map: dict[int, int] = {}

    if args.num_workers <= 1:
        idx_out, dist = annotate_indices(
            0, input_path, indices, task_id, default_reset_seed,
            args.max_oracle_steps, args.warmup_steps, args.log_every,
        )
        for t, d in zip(idx_out, dist):
            dist_map[int(t)] = int(d)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = [
                pool.submit(
                    annotate_indices,
                    wid,
                    input_path,
                    chunk,
                    task_id,
                    default_reset_seed,
                    args.max_oracle_steps,
                    args.warmup_steps,
                    args.log_every,
                )
                for wid, chunk in enumerate(worker_chunks)
                if len(chunk) > 0
            ]
            for fut in as_completed(futures):
                idx_out, dist = fut.result()
                for t, d in zip(idx_out, dist):
                    dist_map[int(t)] = int(d)

    all_distances = np.asarray([dist_map[int(t)] for t in indices], np.int32)

    if chunk_only:
        out = subsample_arrays(data, indices, chunk_size)
        out['distance'] = all_distances
    else:
        out = {k: data[k] for k in data}
        out['distance'] = all_distances

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(f'saved {args.output}, mean d(s_t)={all_distances.mean():.1f}')


if __name__ == '__main__':
    main()
