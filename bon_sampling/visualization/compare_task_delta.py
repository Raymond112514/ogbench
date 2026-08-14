"""Compare oracle Δ / distance across cube-single tasks 1–5 (GCBC rollouts)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _mjstate_at(data, t, ep_start, ep_idx):
    if t == ep_start:
        return np.asarray(data["episode_initial_mjstate"][ep_idx], np.float64)
    return np.asarray(data["next_mjstate"][t - 1], np.float64)


def _annotate_worker(worker_id, input_path, indices, goal_xyz, max_oracle_steps, warmup_steps):
    import gymnasium

    import ogbench.manipspace  # noqa: F401
    from awr.oracle_utils import oracle_distance
    from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle

    data = dict(np.load(input_path, allow_pickle=False))
    ends = data["episode_ends"]
    starts = [0] + ends[:-1].tolist()
    t_to_ep = {}
    for ep_idx, (start, end) in enumerate(zip(starts, ends.tolist())):
        for t in range(int(start), int(end)):
            t_to_ep[t] = (int(start), ep_idx)

    oracle_env = gymnasium.make("cube-single-v0", mode="data_collection", terminate_at_goal=True)
    oracle_env.reset()
    oracle = CubeMarkovOracle(env=oracle_env)
    goal = np.asarray(goal_xyz, np.float64)
    distances = np.zeros(len(indices), np.int32)
    for j, t in enumerate(indices):
        t = int(t)
        ep_start, ep_idx = t_to_ep[t]
        mjstate = _mjstate_at(data, t, ep_start, ep_idx)
        distances[j] = oracle_distance(
            oracle_env, oracle, mjstate, max_oracle_steps,
            warmup_steps=warmup_steps, goal_xyz=goal,
        )
        if (j + 1) % 40 == 0:
            print(f"  oracle worker {worker_id}: {j + 1}/{len(indices)}", flush=True)
    oracle_env.close()
    return indices, distances


def annotate(data: dict, num_workers: int, max_oracle_steps: int = 200, warmup_steps: int = 2):
    from awr.annotate import chunk_boundary_indices

    n = len(data["observations"])
    indices = chunk_boundary_indices(data["episode_ends"], int(data["chunk_size"]))
    fd, tmp = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(
            tmp,
            next_mjstate=data["next_mjstate"],
            episode_initial_mjstate=data["episode_initial_mjstate"],
            episode_ends=data["episode_ends"],
        )
        dist_map = {}
        splits = np.array_split(indices, num_workers)
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futs = [
                pool.submit(
                    _annotate_worker, wid, tmp, split, data["goal_xyz"],
                    max_oracle_steps, warmup_steps,
                )
                for wid, split in enumerate(splits)
                if len(split) > 0
            ]
            for fut in as_completed(futs):
                idx_out, dist = fut.result()
                dist_map.update(zip(idx_out.tolist(), dist.tolist()))
        distance = np.full(n, -1, np.int32)
        for t, d in dist_map.items():
            distance[int(t)] = int(d)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return distance


def deltas_from_distance(distance, episode_ends, chunk_size):
    from awr.annotate import episode_ranges

    ds = []
    for start, end in episode_ranges(episode_ends):
        ts = list(range(start, end, chunk_size))
        for a, b in zip(ts, ts[1:]):
            if distance[a] < 0 or distance[b] < 0:
                continue
            ds.append(int(distance[a]) - int(distance[b]))
    return np.asarray(ds, np.int32)


def summarize(task_id, distance, deltas, chunk_size, success_rate):
    d = distance[distance >= 0]
    H = int(chunk_size)
    w = 5
    s = (2.0 * H) / w
    i = np.clip(np.floor((deltas.astype(np.float64) + H) / s), 0, w - 1).astype(int)
    ell = i - (w - 1) / 2.0
    bins, counts = np.unique(ell, return_counts=True)
    return {
        "task_id": int(task_id),
        "success_rate": float(success_rate),
        "n_states": int(len(d)),
        "n_chunks": int(len(deltas)),
        "d_mean": float(d.mean()) if len(d) else 0.0,
        "d_median": float(np.median(d)) if len(d) else 0.0,
        "d_p10": float(np.percentile(d, 10)) if len(d) else 0.0,
        "d_p90": float(np.percentile(d, 90)) if len(d) else 0.0,
        "delta_mean": float(deltas.mean()) if len(deltas) else 0.0,
        "delta_std": float(deltas.std()) if len(deltas) else 0.0,
        "delta_p10": float(np.percentile(deltas, 10)) if len(deltas) else 0.0,
        "delta_p90": float(np.percentile(deltas, 90)) if len(deltas) else 0.0,
        "frac_in_pm_H": float(np.mean((deltas >= -H) & (deltas <= H))) if len(deltas) else 0.0,
        "frac_gt0": float(np.mean(deltas > 0)) if len(deltas) else 0.0,
        "frac_lt0": float(np.mean(deltas < 0)) if len(deltas) else 0.0,
        "w5_mass": {str(float(b)): float(c / len(ell)) for b, c in zip(bins, counts)},
        "delta_hist": {str(int(k)): int(v) for k, v in zip(*np.unique(deltas, return_counts=True))},
    }


def main():
    from awr.collect import parallel_collect

    ckpt = "flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl"
    n_eps = int(os.environ.get("N_EPS", "25"))
    n_workers = int(os.environ.get("N_WORKERS", "8"))
    out_path = Path("bon_sampling/visualization/task_delta_compare.json")
    rows = []
    for task_id in (1, 2, 3, 4, 5):
        print(f"=== task {task_id}: collect {n_eps} eps ===", flush=True)
        data = parallel_collect(
            ckpt, "cube-single-v0", task_id, n_eps, n_workers, n_flow_steps=10,
        )
        print(
            f"task {task_id}: success={data['success_rate']:.3f} "
            f"transitions={len(data['actions'])}",
            flush=True,
        )
        print(f"=== task {task_id}: oracle annotate ===", flush=True)
        distance = annotate(data, n_workers)
        chunk_size = int(data["chunk_size"])
        deltas = deltas_from_distance(distance, data["episode_ends"], chunk_size)
        row = summarize(task_id, distance, deltas, chunk_size, data["success_rate"])
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
        np.savez_compressed(
            f"bon_sampling/visualization/delta_task{task_id}_ep{n_eps}.npz",
            distance=distance[distance >= 0],
            deltas=deltas,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
