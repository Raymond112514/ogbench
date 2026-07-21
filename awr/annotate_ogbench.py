"""Label OGBench play transitions with cube-Markov oracle distance.

Yes — this is possible: OGBench stores per-transition `qpos`/`qvel`, which we restore
into MuJoCo, pin the singletask goal, and run the scripted oracle.

Example:
  python awr/annotate_ogbench.py \
    --env_name=cube-single-play-singletask-task1-v0 \
    --data_percent 10 \
    --num_workers 10 \
    --output awr/data/cube_single_task1_oracle10pct.npz
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')


def parse_task_id(env_name: str) -> int:
    m = re.search(r'-task(\d+)-', env_name)
    if m:
        return int(m.group(1))
    # Cube default singletask alias is task2.
    if 'cube' in env_name:
        return 2
    return 1


def task_goal_xyz(task_id: int) -> np.ndarray:
    import gymnasium

    import ogbench.manipspace  # noqa: F401

    env = gymnasium.make(f'cube-single-singletask-task{task_id}-v0')
    goal = env.unwrapped.task_infos[task_id - 1]['goal_xyzs'][0].copy()
    env.close()
    return np.asarray(goal, np.float64)


def _annotate_worker(worker_id, input_path, indices, goal_xyz, max_oracle_steps, warmup_steps):
    import gymnasium

    import ogbench.manipspace  # noqa: F401
    from awr.oracle_utils import oracle_distance_qpos
    from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle

    data = np.load(input_path, mmap_mode='r')
    oracle_env = gymnasium.make('cube-single-v0', mode='data_collection', terminate_at_goal=True)
    oracle_env.reset()
    oracle = CubeMarkovOracle(env=oracle_env)
    goal = np.asarray(goal_xyz, np.float64)
    distances = np.zeros(len(indices), np.int32)

    for j, t in enumerate(indices):
        t = int(t)
        distances[j] = oracle_distance_qpos(
            oracle_env,
            oracle,
            data['qpos'][t],
            data['qvel'][t],
            goal,
            max_oracle_steps,
            warmup_steps=warmup_steps,
        )
        if (j + 1) % 100 == 0:
            print(f'worker {worker_id}: {j + 1}/{len(indices)}', flush=True)

    oracle_env.close()
    print(f'worker {worker_id}: finished {len(indices)}', flush=True)
    return indices, distances


def annotate_states(qpos, qvel, goal_xyz, indices, num_workers, max_oracle_steps, warmup_steps):
    """Parallel oracle distances for selected state indices."""
    fd, tmp = tempfile.mkstemp(suffix='.npz')
    os.close(fd)
    try:
        np.savez(tmp, qpos=np.asarray(qpos), qvel=np.asarray(qvel))
        dist_map = {}
        chunks = np.array_split(np.asarray(indices, np.int64), num_workers)
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futs = [
                pool.submit(
                    _annotate_worker, wid, tmp, chunk, goal_xyz, max_oracle_steps, warmup_steps,
                )
                for wid, chunk in enumerate(chunks)
                if len(chunk) > 0
            ]
            for fut in as_completed(futs):
                idx_out, dist = fut.result()
                dist_map.update(zip(idx_out.tolist(), dist.tolist()))
        return np.asarray([dist_map[int(t)] for t in indices], np.int32)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env_name', default='cube-single-play-singletask-task1-v0')
    p.add_argument('--output', required=True, help='Output .npz path')
    p.add_argument('--data_percent', type=float, default=100.0, help='Percent of transitions to label (1–100)')
    p.add_argument('--seed', type=int, default=0, help='Subsample seed')
    p.add_argument('--num_workers', type=int, default=10)
    p.add_argument('--max_oracle_steps', type=int, default=200)
    p.add_argument('--warmup_steps', type=int, default=2)
    p.add_argument('--dataset_dir', default=None)
    p.add_argument('--label_next', action='store_true', default=True, help='Also label next-state distance')
    p.add_argument('--no_label_next', action='store_false', dest='label_next')
    args = p.parse_args()

    if not (0 < args.data_percent <= 100):
        p.error(f'--data_percent must be in (0, 100], got {args.data_percent}')

    import ogbench

    task_id = parse_task_id(args.env_name)
    goal_xyz = task_goal_xyz(task_id)
    print(f'env={args.env_name} task_id={task_id} goal_xyz={goal_xyz}')

    kwargs = dict(add_info=True)
    if args.dataset_dir is not None:
        kwargs['dataset_dir'] = args.dataset_dir

    _, train, _ = ogbench.make_env_and_datasets(args.env_name, **kwargs)
    if 'qpos' not in train or 'qvel' not in train:
        raise RuntimeError('Dataset missing qpos/qvel; pass add_info=True (script does this).')

    n = len(train['observations'])
    rng = np.random.default_rng(args.seed)
    if args.data_percent < 100:
        keep = max(1, int(round(n * args.data_percent / 100.0)))
        sel = np.sort(rng.choice(n, size=keep, replace=False))
    else:
        sel = np.arange(n, dtype=np.int64)
    print(f'labeling {len(sel)}/{n} transitions ({args.data_percent:g}%) with {args.num_workers} workers')

    # States to query: current, and next when available (non-terminal uses qpos[i+1]).
    state_indices = set(int(i) for i in sel)
    next_index = {}  # transition i -> state index for s'
    if args.label_next:
        terminals = np.asarray(train['terminals']).reshape(-1)
        for i in sel:
            i = int(i)
            if terminals[i] < 0.5 and i + 1 < n:
                # Within-traj: next obs physics is qpos[i+1].
                if np.allclose(train['observations'][i + 1], train['next_observations'][i]):
                    next_index[i] = i + 1
                    state_indices.add(i + 1)

    state_indices = np.asarray(sorted(state_indices), np.int64)
    print(f'unique states to oracle-label: {len(state_indices)}')

    state_dist = annotate_states(
        train['qpos'],
        train['qvel'],
        goal_xyz,
        state_indices,
        args.num_workers,
        args.max_oracle_steps,
        args.warmup_steps,
    )
    dist_map = {int(t): int(d) for t, d in zip(state_indices, state_dist)}

    distance = np.asarray([dist_map[int(i)] for i in sel], np.int32)
    out = {
        'observations': np.asarray(train['observations'][sel], np.float32),
        'actions': np.asarray(train['actions'][sel], np.float32),
        'next_observations': np.asarray(train['next_observations'][sel], np.float32),
        'rewards': np.asarray(train['rewards'][sel], np.float32).reshape(-1),
        'masks': np.asarray(train['masks'][sel], np.float32).reshape(-1),
        'terminals': np.asarray(train['terminals'][sel], np.float32).reshape(-1),
        'qpos': np.asarray(train['qpos'][sel], np.float32),
        'qvel': np.asarray(train['qvel'][sel], np.float32),
        'distance': distance,
        'goal_xyz': goal_xyz.astype(np.float64),
        'task_id': np.array(task_id, np.int32),
        'env_name': np.array(args.env_name),
        'data_percent': np.array(args.data_percent, np.float32),
        'indices': np.asarray(sel, np.int64),
    }
    if args.label_next:
        next_distance = np.full(len(sel), -1, np.int32)
        for j, i in enumerate(sel):
            i = int(i)
            if i in next_index:
                next_distance[j] = dist_map[next_index[i]]
        out['next_distance'] = next_distance
        # Improvement label: did oracle distance decrease?
        valid = next_distance >= 0
        improved = np.zeros(len(sel), np.float32)
        improved[valid] = (next_distance[valid] < distance[valid]).astype(np.float32)
        out['improved'] = improved

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(
        f'saved {out_path}  n={len(sel)}  mean_distance={float(distance.mean()):.2f}  '
        f'min={int(distance.min())} max={int(distance.max())}'
    )
    if args.label_next and 'improved' in out:
        valid = out['next_distance'] >= 0
        if valid.any():
            print(f'improved rate={float(out["improved"][valid].mean()):.3f} (over {int(valid.sum())} non-terminal)')


if __name__ == '__main__':
    main()
