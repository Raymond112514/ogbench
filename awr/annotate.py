"""Oracle-distance annotation for classifier labels (chunk boundaries)."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')


def episode_ranges(episode_ends):
    starts = [0] + episode_ends[:-1].tolist()
    return list(zip(starts, episode_ends))


def chunk_boundary_indices(episode_ends, chunk_size):
    indices = []
    for start, end in episode_ranges(episode_ends):
        t = start
        while t < end:
            indices.append(t)
            t += chunk_size
    return np.asarray(indices, np.int64)


def build_action_chunks(actions, episode_ends, chunk_size, indices):
    act_dim = actions.shape[-1]
    chunks = np.zeros((len(indices), chunk_size, act_dim), np.float32)
    masks = np.zeros((len(indices), chunk_size), np.float32)
    ep_end = {}
    for start, end in episode_ranges(episode_ends):
        for t in range(start, end):
            ep_end[t] = end
    for i, t in enumerate(indices):
        end = ep_end[int(t)]
        for k in range(chunk_size):
            if t + k < end:
                chunks[i, k] = actions[t + k]
                masks[i, k] = 1.0
    return chunks, masks


def _annotate_worker(worker_id, input_path, indices, task_id, max_oracle_steps, warmup_steps):
    import gymnasium

    import ogbench.manipspace  # noqa: F401
    from awr.oracle_utils import oracle_distance
    from awr.sim_state import get_sim_state
    from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle

    data = np.load(input_path, mmap_mode='r')
    episode_ends = data['episode_ends']
    starts = [0] + episode_ends[:-1].tolist()
    ep_start_for, ep_idx_for = {}, {}
    for ep_idx, (start, end) in enumerate(zip(starts, episode_ends)):
        ep_idx_for[int(start)] = ep_idx
        for t in range(start, end):
            ep_start_for[t] = start

    policy_env = gymnasium.make('cube-single-v0')
    oracle_env = gymnasium.make('cube-single-v0', mode='data_collection', terminate_at_goal=True)
    oracle_env.reset()
    oracle = CubeMarkovOracle(env=oracle_env)
    distances = np.zeros(len(indices), np.int32)

    for j, t in enumerate(indices):
        t = int(t)
        ep_start = ep_start_for[t]
        ep_idx = ep_idx_for[ep_start]
        if t > ep_start:
            mjstate = data['next_mjstate'][t - 1]
        else:
            mjstate = np.asarray(data['episode_initial_mjstate'][ep_idx], np.float64)
        distances[j] = oracle_distance(oracle_env, oracle, mjstate, max_oracle_steps, warmup_steps=warmup_steps)

    policy_env.close()
    oracle_env.close()
    print(f'worker {worker_id}: finished {len(indices)}', flush=True)
    return indices, distances


def annotate(input_path: str, output_path: str, num_workers: int = 10, max_oracle_steps: int = 200, warmup_steps: int = 2):
    data = dict(np.load(input_path, allow_pickle=False))
    chunk_size = int(data.get('chunk_size', 1))
    task_id = int(data.get('task_id', 1))
    indices = chunk_boundary_indices(data['episode_ends'], chunk_size)

    dist_map = {}
    chunks = np.array_split(indices, num_workers)
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futs = [
            pool.submit(_annotate_worker, wid, str(Path(input_path).resolve()), chunk, task_id, max_oracle_steps, warmup_steps)
            for wid, chunk in enumerate(chunks) if len(chunk) > 0
        ]
        for fut in as_completed(futs):
            idx_out, dist = fut.result()
            dist_map.update(zip(idx_out.tolist(), dist.tolist()))

    distance = np.asarray([dist_map[int(t)] for t in indices], np.int32)
    # subsample to chunk boundaries
    new_ends, count = [], 0
    for start, end in episode_ranges(data['episode_ends']):
        count += len(range(start, end, chunk_size))
        new_ends.append(count)

    action_chunks, chunk_masks = build_action_chunks(data['actions'], data['episode_ends'], chunk_size, indices)
    out = {
        'observations': data['observations'][indices],
        'actions': data['actions'][indices],
        'next_observations': data['next_observations'][indices],
        'next_mjstate': data['next_mjstate'][indices],
        'episode_ends': np.asarray(new_ends, np.int32),
        'action_chunks': action_chunks,
        'chunk_masks': chunk_masks,
        'distance': distance,
        'chunk_size': np.array(chunk_size),
        'task_id': data.get('task_id', np.array(1)),
    }
    if 'goal_xyz' in data:
        out['goal_xyz'] = data['goal_xyz']
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **out)
    return float(distance.mean())
