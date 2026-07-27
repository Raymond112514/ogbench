"""Compare GCBC baseline vs oracle-lookahead action selection.

At each replan state s:
  1. Sample K action chunks from the GCBC policy.
  2. For each candidate a_i: step s -> s', measure oracle distance d(s'), restore s.
  3. Select a candidate:
       --select distance: execute argmin_i d(s'_i)  (continuous oracle ranking)
       --select binary:   label y_i=1 if d(s)-d(s'_i) >= H-tau else 0;
                          execute uniformly at random among argmax y
                          (all positives weighted equally; if none, uniform over K)

Usage (from ogbench/):
  python awr/eval_oracle_lookahead.py \\
    --policy_ckpt flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl \\
    --num_episodes 50 --bon_n 8 --num_workers 10 \\
    --select binary --tau 5
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')


def _random_seed() -> int:
    return int.from_bytes(os.urandom(4), 'big') & 0x7FFFFFFF


def _count_gpus() -> int:
    try:
        out = subprocess.check_output(['nvidia-smi', '-L'], text=True, stderr=subprocess.DEVNULL)
        return sum(1 for line in out.splitlines() if line.strip().startswith('GPU '))
    except (OSError, subprocess.CalledProcessError):
        return 0


def _available_gpu_ids(jax_device: int | None) -> list[int]:
    n = _count_gpus()
    if n == 0:
        return []
    base = 0 if jax_device is None else jax_device
    if base >= n:
        return []
    return list(range(base, n))


def _resolve_parallel_config(
    jax_platform: str,
    num_workers: int,
    jax_device: int | None,
) -> tuple[str, int, list[int]]:
    gpu_ids = _available_gpu_ids(jax_device)

    if jax_platform == 'auto':
        jax_platform = 'gpu' if num_workers == 1 and gpu_ids else 'cpu'

    if jax_platform == 'gpu':
        if not gpu_ids:
            print('warning: no GPUs available; using jax_platform=cpu for workers')
            return 'cpu', num_workers, []
        if num_workers > len(gpu_ids):
            print(
                f'warning: requested {num_workers} workers but only {len(gpu_ids)} GPU(s) '
                f'available ({gpu_ids}); capping workers to {len(gpu_ids)}'
            )
            num_workers = len(gpu_ids)
        worker_gpu_ids = gpu_ids[:num_workers]
        print(f'GPU workers: {list(zip(range(num_workers), worker_gpu_ids))}')
        return 'gpu', num_workers, worker_gpu_ids

    return 'cpu', num_workers, []


def capture_state(env) -> np.ndarray:
    from awr.oracle_utils import capture_sim_state

    return capture_sim_state(env)


def restore_policy_state(env, mjstate: np.ndarray, goal_xyz: np.ndarray) -> None:
    """Exact restore for candidate probing / commit (no oracle warmup)."""
    from awr.oracle_utils import pin_goal
    from awr.sim_state import set_sim_state

    u = env.unwrapped
    set_sim_state(u._model, u._data, mjstate)
    pin_goal(env, goal_xyz)


def execute_chunk(env, ob, chunk, max_steps, steps_so_far):
    steps = steps_so_far
    success = False
    for action in np.asarray(chunk):
        if steps >= max_steps:
            break
        ob, _, term, trunc, info = env.step(np.clip(action, -1.0, 1.0))
        steps += 1
        success = bool(info.get('success', False))
        if term or trunc:
            return ob, success, steps, True
    return ob, success, steps, False


def probe_next_state(probe_env, mjstate, goal_xyz, chunk) -> np.ndarray:
    """Step a candidate from s on a side env; return mjstate of s'."""
    probe_env.reset()  # clear TimeLimit; physics overwritten next
    restore_policy_state(probe_env, mjstate, goal_xyz)
    for action in np.asarray(chunk):
        _, _, term, trunc, _ = probe_env.step(np.clip(action, -1.0, 1.0))
        if term or trunc:
            break
    return capture_state(probe_env)


def _candidate_distances(
    probe_env,
    oracle_env,
    oracle,
    mjstate: np.ndarray,
    goal_xyz: np.ndarray,
    candidates: np.ndarray,
    max_oracle_steps: int,
    warmup_steps: int,
) -> np.ndarray:
    from awr.oracle_utils import oracle_distance

    distances = np.empty(len(candidates), dtype=np.int32)
    for i, chunk in enumerate(candidates):
        next_state = probe_next_state(probe_env, mjstate, goal_xyz, chunk)
        distances[i] = oracle_distance(
            oracle_env,
            oracle,
            next_state,
            max_oracle_steps,
            warmup_steps=warmup_steps,
            goal_xyz=goal_xyz,
        )
    return distances


def select_oracle_lookahead(
    probe_env,
    oracle_env,
    oracle,
    mjstate: np.ndarray,
    goal_xyz: np.ndarray,
    candidates: np.ndarray,
    max_oracle_steps: int,
    warmup_steps: int,
) -> int:
    """Pick the candidate with lowest oracle distance after one chunk step."""
    distances = _candidate_distances(
        probe_env, oracle_env, oracle, mjstate, goal_xyz, candidates,
        max_oracle_steps, warmup_steps,
    )
    return int(np.argmin(distances))


def select_oracle_binary(
    probe_env,
    oracle_env,
    oracle,
    mjstate: np.ndarray,
    goal_xyz: np.ndarray,
    candidates: np.ndarray,
    max_oracle_steps: int,
    warmup_steps: int,
    horizon: int,
    tau: int | None,
    rng: np.random.Generator,
) -> int:
    """Pick uniformly among candidates with best binary progress label.

    y=1 iff d(s) - d(s') >= H - tau (default tau=H-1 ⇒ threshold 1).
    """
    from awr.oracle_utils import oracle_distance, progress_label

    d_s = oracle_distance(
        oracle_env, oracle, mjstate, max_oracle_steps,
        warmup_steps=warmup_steps, goal_xyz=goal_xyz,
    )
    d_next = _candidate_distances(
        probe_env, oracle_env, oracle, mjstate, goal_xyz, candidates,
        max_oracle_steps, warmup_steps,
    )
    labels = np.asarray(
        [progress_label(d_s, int(d), horizon, tau) for d in d_next],
        dtype=np.float32,
    )
    best = float(labels.max())
    pool = np.flatnonzero(labels == best)
    return int(rng.choice(pool))


def method_name(select: str, bon_n: int, tau: int | None, horizon: int) -> str:
    if select == 'distance':
        return f'oracle_lookahead_k{bon_n}'
    resolved_tau = (horizon - 1) if tau is None else int(tau)
    return f'oracle_binary_k{bon_n}_tau{resolved_tau}'


def _make_sample_candidates_fn(apply_fn, num_samples, chunk_size, act_dim, n_flow_steps, goal_condition):
    import jax
    import jax.numpy as jnp

    from flow_bc.model import sample_action_chunks

    @jax.jit
    def sample_candidates(params, observation, goal, key):
        return sample_action_chunks(
            params,
            apply_fn,
            jnp.asarray(observation),
            jnp.asarray(goal) if goal_condition else None,
            key,
            num_samples,
            chunk_size,
            act_dim,
            n_flow_steps=n_flow_steps,
            goal_condition=goal_condition,
        )

    return sample_candidates


def rollout_bc(env, params, sample_one_fn, task_id, reset_seed, policy_seed, max_steps, goal_condition):
    import jax
    import jax.numpy as jnp

    ob, info = env.reset(options=dict(task_id=task_id), seed=reset_seed)
    goal = info['goal'] if goal_condition else None
    key = jax.random.PRNGKey(policy_seed)
    success, steps, done = False, 0, False

    while steps < max_steps and not done:
        key, sample_key = jax.random.split(key)
        chunk = sample_one_fn(
            params,
            jnp.asarray(ob),
            jnp.asarray(goal) if goal_condition else None,
            sample_key,
        )
        ob, success, steps, done = execute_chunk(env, ob, chunk, max_steps, steps)
    return success, steps


def rollout_oracle_lookahead(
    env,
    probe_env,
    oracle_env,
    oracle,
    params,
    sample_candidates_fn,
    task_id,
    reset_seed,
    policy_seed,
    max_steps,
    goal_xyz,
    max_oracle_steps,
    warmup_steps,
    goal_condition,
    select: str = 'distance',
    horizon: int = 1,
    tau: int | None = None,
    tiebreak_seed: int = 0,
):
    import jax
    import jax.numpy as jnp

    ob, info = env.reset(options=dict(task_id=task_id), seed=reset_seed)
    goal = info['goal'] if goal_condition else None
    key = jax.random.PRNGKey(policy_seed)
    rng = np.random.default_rng(tiebreak_seed)
    success, steps, done = False, 0, False

    while steps < max_steps and not done:
        mjstate = capture_state(env)
        key, sample_key = jax.random.split(key)
        candidates = np.asarray(
            sample_candidates_fn(
                params,
                jnp.asarray(ob),
                jnp.asarray(goal) if goal_condition else None,
                sample_key,
            )
        )
        if select == 'binary':
            pick = select_oracle_binary(
                probe_env, oracle_env, oracle, mjstate, goal_xyz, candidates,
                max_oracle_steps, warmup_steps, horizon, tau, rng,
            )
        else:
            pick = select_oracle_lookahead(
                probe_env, oracle_env, oracle, mjstate, goal_xyz, candidates,
                max_oracle_steps, warmup_steps,
            )
        ob, success, steps, done = execute_chunk(env, ob, candidates[pick], max_steps, steps)
    return success, steps


def _run_batch(
    worker_id: int,
    episode_indices: np.ndarray,
    policy_ckpt: str,
    env_name: str,
    task_id: int,
    max_steps: int,
    n_flow_steps: int,
    bon_n: int,
    max_oracle_steps: int,
    warmup_steps: int,
    reset_seed: int,
    seed: int,
    randomize_resets: bool,
    egl_device: int | None,
    jax_platform: str,
    gpu_id: int | None,
    select: str,
    tau: int | None,
) -> list[tuple[int, tuple[bool, int], tuple[bool, int]]]:
    if jax_platform == 'gpu' and gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        os.environ['JAX_PLATFORMS'] = 'cuda'
    else:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        os.environ['JAX_PLATFORMS'] = 'cpu'

    if egl_device is not None:
        egl = egl_device + (gpu_id if gpu_id is not None else worker_id)
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(egl)

    import gymnasium

    import ogbench.manipspace  # noqa: F401
    from flow_bc.checkpoint import load_flow_bc
    from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle

    policy, params, meta = load_flow_bc(policy_ckpt)
    act_dim = meta['act_dim']
    chunk_size = meta['chunk_size']
    goal_condition = meta['goal_condition']

    sample_one_fn = _make_sample_candidates_fn(
        policy.apply, 1, chunk_size, act_dim, n_flow_steps, goal_condition
    )

    def sample_one(params_, observation, goal, key):
        # sample_action_chunks returns (K, H, A); squeeze K=1 for baseline.
        return sample_one_fn(params_, observation, goal, key)[0]

    sample_candidates_fn = _make_sample_candidates_fn(
        policy.apply, bon_n, chunk_size, act_dim, n_flow_steps, goal_condition
    )

    env = gymnasium.make(env_name)
    probe_env = gymnasium.make(env_name)
    oracle_env = gymnasium.make('cube-single-v0', mode='data_collection', terminate_at_goal=True)
    oracle_env.reset()
    oracle = CubeMarkovOracle(env=oracle_env)
    goal_xyz = env.unwrapped.task_infos[task_id - 1]['goal_xyzs'][0].copy()

    out: list[tuple[int, tuple[bool, int], tuple[bool, int]]] = []
    for ep in episode_indices:
        if randomize_resets:
            episode_reset_seed = _random_seed()
            policy_seed = _random_seed()
        else:
            episode_reset_seed = reset_seed + int(ep)
            policy_seed = seed + int(ep)

        bc = rollout_bc(
            env,
            params,
            sample_one,
            task_id,
            episode_reset_seed,
            policy_seed,
            max_steps,
            goal_condition,
        )
        lookahead = rollout_oracle_lookahead(
            env,
            probe_env,
            oracle_env,
            oracle,
            params,
            sample_candidates_fn,
            task_id,
            episode_reset_seed,
            policy_seed,
            max_steps,
            goal_xyz,
            max_oracle_steps,
            warmup_steps,
            goal_condition,
            select=select,
            horizon=chunk_size,
            tau=tau,
            tiebreak_seed=policy_seed + 17,
        )
        out.append((int(ep), bc, lookahead))

    env.close()
    probe_env.close()
    oracle_env.close()
    print(f'worker {worker_id}: finished {len(episode_indices)} episodes', flush=True)
    return out


def parallel_evaluate(
    policy_ckpt: str,
    env_name: str,
    task_id: int,
    num_episodes: int,
    num_workers: int,
    max_steps: int,
    n_flow_steps: int,
    bon_n: int,
    max_oracle_steps: int,
    warmup_steps: int,
    reset_seed: int,
    seed: int,
    randomize_resets: bool,
    egl_device: int | None,
    jax_platform: str,
    worker_gpu_ids: list[int],
    select: str,
    tau: int | None,
) -> tuple[list[tuple[bool, int]], list[tuple[bool, int]]]:
    episode_splits = np.array_split(np.arange(num_episodes), num_workers)
    merged: list[tuple[int, tuple[bool, int], tuple[bool, int]]] = []
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
        futures = [
            pool.submit(
                _run_batch,
                wid,
                split,
                str(Path(policy_ckpt).resolve()),
                env_name,
                task_id,
                max_steps,
                n_flow_steps,
                bon_n,
                max_oracle_steps,
                warmup_steps,
                reset_seed,
                seed,
                randomize_resets,
                egl_device,
                jax_platform,
                worker_gpu_ids[wid] if worker_gpu_ids else None,
                select,
                tau,
            )
            for wid, split in enumerate(episode_splits)
            if len(split) > 0
        ]
        for fut in futures:
            merged.extend(fut.result())

    merged.sort(key=lambda x: x[0])
    bc_results = [bc for _, bc, _ in merged]
    lookahead_results = [la for _, _, la in merged]
    return bc_results, lookahead_results


def summary_row(method: str, results) -> dict:
    successes = np.array([r[0] for r in results], dtype=np.float32)
    lengths = np.array([r[1] for r in results], dtype=np.float32)
    return {
        'method': method,
        'success_rate': float(successes.mean()),
        'success_count': int(successes.sum()),
        'avg_length': float(lengths.mean()),
        'num_episodes': len(results),
    }


def summarize(method: str, results) -> dict:
    row = summary_row(method, results)
    print(
        f'{method}: success={row["success_rate"]:.3f} '
        f'({row["success_count"]}/{row["num_episodes"]}) '
        f'avg_len={row["avg_length"]:.1f}'
    )
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--policy_ckpt', default='flow_bc/checkpoints/cube_single_gcbc/best.pkl')
    p.add_argument('--n_flow_steps', type=int, default=10)
    p.add_argument('--env_name', default='cube-single-v0')
    p.add_argument('--task_id', type=int, default=1)
    p.add_argument('--num_episodes', type=int, default=50)
    p.add_argument('--num_workers', type=int, default=10)
    p.add_argument('--max_steps', type=int, default=None)
    p.add_argument('--bon_n', type=int, default=8, help='Number of action-chunk candidates (K)')
    p.add_argument(
        '--select',
        choices=['distance', 'binary'],
        default='distance',
        help='distance: argmin d(s\'); binary: uniform among max progress label',
    )
    p.add_argument(
        '--tau',
        type=int,
        default=None,
        help='(--select binary) slack; y=1 iff d(s)-d(s\') >= H-tau. Default tau=H-1',
    )
    p.add_argument('--max_oracle_steps', type=int, default=200)
    p.add_argument('--warmup_steps', type=int, default=2)
    p.add_argument(
        '--fixed_seeds',
        action='store_true',
        help='Use deterministic reset_seed+i / seed+i (default: random per episode)',
    )
    p.add_argument('--reset_seed', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--egl_device', type=int, default=None)
    p.add_argument('--jax_platform', choices=['auto', 'cpu', 'gpu'], default='auto')
    p.add_argument('--jax_device', type=int, default=None)
    p.add_argument('--wandb_project', default='ogbench_bon')
    p.add_argument('--wandb_name', default=None)
    p.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
    args = p.parse_args()

    if args.egl_device is not None:
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.egl_device)

    jax_platform, num_workers, worker_gpu_ids = _resolve_parallel_config(
        args.jax_platform, args.num_workers, args.jax_device
    )

    import gymnasium
    import wandb

    import ogbench.manipspace  # noqa: F401
    from flow_bc.checkpoint import read_ckpt_meta

    policy_meta = read_ckpt_meta(args.policy_ckpt)
    chunk_size = policy_meta['chunk_size']
    env_name = policy_meta.get('eval_env_name', args.env_name)
    tmp_env = gymnasium.make(env_name)
    max_steps = args.max_steps or tmp_env.spec.max_episode_steps
    task_name = tmp_env.unwrapped.task_infos[args.task_id - 1]['task_name']
    tmp_env.close()

    if args.tau is None:
        resolved_tau = chunk_size - 1
    else:
        resolved_tau = int(args.tau)
    la_name = method_name(args.select, args.bon_n, args.tau, chunk_size)

    randomize_resets = not args.fixed_seeds
    reset_mode = (
        'random per episode'
        if randomize_resets
        else f'fixed reset_seeds={args.reset_seed}..{args.reset_seed + args.num_episodes - 1}'
    )

    select_desc = (
        f'select=distance (argmin d)'
        if args.select == 'distance'
        else f'select=binary tau={resolved_tau} thr={chunk_size - resolved_tau}'
    )
    print(
        f'env={env_name} task_id={args.task_id} ({task_name}) '
        f'max_steps={max_steps} chunk_size={chunk_size} K={args.bon_n} '
        f'{select_desc} episodes={args.num_episodes} workers={num_workers} '
        f'jax_platform={jax_platform} {reset_mode}'
    )

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config={
            **vars(args),
            'chunk_size': chunk_size,
            'env_name': env_name,
            'task_name': task_name,
            'resolved_tau': resolved_tau,
            'progress_threshold': chunk_size - resolved_tau if args.select == 'binary' else None,
            'la_name': la_name,
        },
    )

    bc_results, lookahead_results = parallel_evaluate(
        args.policy_ckpt,
        env_name,
        args.task_id,
        args.num_episodes,
        num_workers,
        max_steps,
        args.n_flow_steps,
        args.bon_n,
        args.max_oracle_steps,
        args.warmup_steps,
        args.reset_seed,
        args.seed,
        randomize_resets,
        args.egl_device,
        jax_platform,
        worker_gpu_ids,
        args.select,
        args.tau,
    )

    print(f'episodes={args.num_episodes}')
    bc_row = summarize('baseline', bc_results)
    la_row = summarize(la_name, lookahead_results)

    wandb.log({
        'baseline/success_rate': bc_row['success_rate'],
        'lookahead/success_rate': la_row['success_rate'],
        'task_id': args.task_id,
        'tau': resolved_tau,
        'bon_n': args.bon_n,
        'select': args.select,
    })
    wandb.finish()


if __name__ == '__main__':
    main()