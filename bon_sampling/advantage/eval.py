"""Compare flow-BC rollouts vs BoN reranking with advantage classifier/regressor.

Usage (from ogbench/):
  # Baseline + BoN K=4,8,16,32,64 on 100 episodes, 10 workers, CSV output
  python bon_sampling/advantage/eval.py \\
    --policy_ckpt flow_bc/checkpoints/cube_single_gcbc/best.pkl \\
    --advantage_ckpt bon_sampling/advantage/checkpoints_flowbc100/best.pkl \\
    --num_episodes 100 --num_workers 10 \\
    --bon_n_list 4,8,16,32,64 \\
    --output_csv bon_sampling/advantage/results/flowbc100_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

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


def rollout_bc(
    env, params, apply_fn, task_id, reset_seed, policy_seed,
    chunk_size, act_dim, n_flow_steps, max_steps, goal_condition=True,
):
    import jax
    import jax.numpy as jnp

    from flow_bc.model import sample_action_chunk

    ob, info = env.reset(options=dict(task_id=task_id), seed=reset_seed)
    goal = info['goal'] if goal_condition else None
    key = jax.random.PRNGKey(policy_seed)
    success, steps, done = False, 0, False

    while steps < max_steps and not done:
        key, sample_key = jax.random.split(key)
        chunk = sample_action_chunk(
            params, apply_fn, jnp.asarray(ob), jnp.asarray(goal) if goal_condition else None, sample_key,
            chunk_size=chunk_size, act_dim=act_dim, n_flow_steps=n_flow_steps,
            goal_condition=goal_condition,
        )
        ob, success, steps, done = execute_chunk(env, ob, chunk, max_steps, steps)
    return success, steps


def _make_sample_candidates_fn(
    apply_fn,
    num_samples: int,
    chunk_size: int,
    act_dim: int,
    n_flow_steps: int,
    goal_condition: bool,
):
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


def rollout_bon(
    env, params, apply_fn, select_fn, adv_params, sample_candidates_fn,
    task_id, reset_seed, policy_seed,
    chunk_size, act_dim, n_flow_steps, max_steps, goal_condition=True,
):
    import jax
    import jax.numpy as jnp

    ob, info = env.reset(options=dict(task_id=task_id), seed=reset_seed)
    goal = info['goal'] if goal_condition else None
    key = jax.random.PRNGKey(policy_seed)
    success, steps, done = False, 0, False

    while steps < max_steps and not done:
        key, sample_key, _ = jax.random.split(key, 3)
        candidates = sample_candidates_fn(
            params,
            jnp.asarray(ob),
            jnp.asarray(goal) if goal_condition else None,
            sample_key,
        )
        pick = int(select_fn(adv_params, jnp.asarray(ob), candidates))
        chunk = candidates[pick]
        ob, success, steps, done = execute_chunk(env, ob, chunk, max_steps, steps)
    return success, steps


def _load_advantage(checkpoint_path: str, advantage_mode: str):
    import jax

    from bon_sampling.advantage.model import AdvantageClassifier, AdvantageRegressor

    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    ckpt_mode = ckpt.get('mode', 'classifier')
    mode = ckpt_mode if advantage_mode == 'auto' else advantage_mode
    if mode == 'regression':
        model = AdvantageRegressor(hidden=ckpt['hidden'])
    else:
        model = AdvantageClassifier(hidden=ckpt['hidden'])
    return model, ckpt['params'], mode, ckpt.get('chunk_size', 1)


def _make_select_fn(model, mode: str):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def select_chunk(params, observation, chunks):
        flat = chunks.reshape(chunks.shape[0], -1)
        obs = jnp.broadcast_to(observation[None], (flat.shape[0], observation.shape[0]))
        preds = model.apply(params, obs, flat)
        if mode == 'regression':
            return jnp.argmax(preds)
        return jnp.argmax(jax.nn.sigmoid(preds))

    return select_chunk


def _run_batch(
    worker_id: int,
    episode_indices: np.ndarray,
    policy_ckpt: str,
    advantage_ckpt: str,
    advantage_mode: str,
    env_name: str,
    task_id: int,
    max_steps: int,
    n_flow_steps: int,
    bon_ns: list[int],
    reset_seed: int,
    seed: int,
    randomize_resets: bool,
    egl_device: int | None,
    jax_platform: str,
    gpu_id: int | None,
) -> list[tuple[int, tuple[bool, int], dict[int, tuple[bool, int]]]]:
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

    policy, params, meta = load_flow_bc(policy_ckpt)
    act_dim = meta['act_dim']
    chunk_size = meta['chunk_size']
    goal_condition = meta['goal_condition']

    adv_model, adv_params, mode, adv_chunk_size = _load_advantage(advantage_ckpt, advantage_mode)
    if adv_chunk_size != chunk_size:
        print(
            f'worker {worker_id}: warning policy chunk_size={chunk_size} '
            f'!= advantage chunk_size={adv_chunk_size}',
            flush=True,
        )
    select_fn = _make_select_fn(adv_model, mode)
    sample_candidates_fns = {
        bon_n: _make_sample_candidates_fn(
            policy.apply, bon_n, chunk_size, act_dim, n_flow_steps, goal_condition
        )
        for bon_n in bon_ns
    }

    env = gymnasium.make(env_name)
    out: list[tuple[int, tuple[bool, int], dict[int, tuple[bool, int]]]] = []
    for ep in episode_indices:
        if randomize_resets:
            episode_reset_seed = _random_seed()
            policy_seed = _random_seed()
        else:
            episode_reset_seed = reset_seed + int(ep)
            policy_seed = seed + int(ep)
        bc = rollout_bc(
            env, params, policy.apply, task_id,
            episode_reset_seed, policy_seed,
            chunk_size, act_dim, n_flow_steps, max_steps, goal_condition,
        )
        bon_by_k = {}
        for bon_n in bon_ns:
            bon_by_k[bon_n] = rollout_bon(
                env, params, policy.apply, select_fn, adv_params, sample_candidates_fns[bon_n],
                task_id, episode_reset_seed, policy_seed,
                chunk_size, act_dim, n_flow_steps, max_steps, goal_condition,
            )
        out.append((int(ep), bc, bon_by_k))

    env.close()
    print(f'worker {worker_id}: finished {len(episode_indices)} episodes', flush=True)
    return out


def parallel_evaluate(
    policy_ckpt: str,
    advantage_ckpt: str,
    advantage_mode: str,
    env_name: str,
    task_id: int,
    num_episodes: int,
    num_workers: int,
    max_steps: int,
    n_flow_steps: int,
    bon_ns: list[int],
    reset_seed: int,
    seed: int,
    randomize_resets: bool,
    egl_device: int | None,
    jax_platform: str,
    worker_gpu_ids: list[int],
) -> tuple[list[tuple[bool, int]], dict[int, list[tuple[bool, int]]]]:
    episode_splits = np.array_split(np.arange(num_episodes), num_workers)
    merged: list[tuple[int, tuple[bool, int], dict[int, tuple[bool, int]]]] = []
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
        futures = [
            pool.submit(
                _run_batch,
                wid,
                split,
                str(Path(policy_ckpt).resolve()),
                str(Path(advantage_ckpt).resolve()),
                advantage_mode,
                env_name,
                task_id,
                max_steps,
                n_flow_steps,
                bon_ns,
                reset_seed,
                seed,
                randomize_resets,
                egl_device,
                jax_platform,
                worker_gpu_ids[wid] if worker_gpu_ids else None,
            )
            for wid, split in enumerate(episode_splits)
            if len(split) > 0
        ]
        for fut in futures:
            merged.extend(fut.result())

    merged.sort(key=lambda x: x[0])
    bc_results = [bc for _, bc, _ in merged]
    bon_results_by_k = {bon_n: [] for bon_n in bon_ns}
    for _, _, bon_by_k in merged:
        for bon_n in bon_ns:
            bon_results_by_k[bon_n].append(bon_by_k[bon_n])
    return bc_results, bon_results_by_k


def summary_row(method: str, results, bon_k: str | int = '') -> dict:
    successes = np.array([r[0] for r in results], dtype=np.float32)
    lengths = np.array([r[1] for r in results], dtype=np.float32)
    return {
        'method': method,
        'bon_k': bon_k,
        'success_rate': float(successes.mean()),
        'success_count': int(successes.sum()),
        'avg_length': float(lengths.mean()),
        'num_episodes': len(results),
    }


def summarize(method: str, results, bon_k: str | int = '') -> dict:
    row = summary_row(method, results, bon_k)
    label = method if bon_k == '' else f'{method}-{bon_k}'
    print(
        f'{label}: success={row["success_rate"]:.3f} '
        f'({row["success_count"]}/{row["num_episodes"]}) '
        f'avg_len={row["avg_length"]:.1f}'
    )
    return row


def write_results_csv(
    path: str,
    bc_results: list[tuple[bool, int]],
    bon_results_by_k: dict[int, list[tuple[bool, int]]],
) -> list[dict]:
    summaries = [summary_row('baseline', bc_results)]
    for bon_n in sorted(bon_results_by_k):
        summaries.append(summary_row('bon', bon_results_by_k[bon_n], bon_n))

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'method', 'bon_k', 'success', 'length'])
        for ep, (success, length) in enumerate(bc_results):
            writer.writerow([ep, 'baseline', '', int(success), length])
        for bon_n in sorted(bon_results_by_k):
            for ep, (success, length) in enumerate(bon_results_by_k[bon_n]):
                writer.writerow([ep, 'bon', bon_n, int(success), length])
        writer.writerow([])
        writer.writerow(['summary', 'method', 'bon_k', 'success_rate', 'success_count', 'avg_length', 'num_episodes'])
        for row in summaries:
            writer.writerow([
                '',
                row['method'],
                row['bon_k'],
                f'{row["success_rate"]:.4f}',
                row['success_count'],
                f'{row["avg_length"]:.1f}',
                row['num_episodes'],
            ])
    print(f'wrote results to {out_path}')
    return summaries


def parse_bon_ns(bon_n: int, bon_n_list: str | None) -> list[int]:
    if bon_n_list:
        return [int(x.strip()) for x in bon_n_list.split(',') if x.strip()]
    return [bon_n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy_ckpt', default='flow_bc/checkpoints/best.pkl')
    p.add_argument('--advantage_ckpt', default='bon_sampling/advantage/checkpoints/best.pkl')
    p.add_argument('--advantage_mode', choices=('classifier', 'regression', 'auto'), default='auto')
    p.add_argument('--n_flow_steps', type=int, default=10)
    p.add_argument('--env_name', default='cube-single-v0')
    p.add_argument('--task_id', type=int, default=1)
    p.add_argument('--num_episodes', type=int, default=50)
    p.add_argument('--num_workers', type=int, default=10,
                   help='Parallel env rollouts (spawn workers; JAX on CPU by default)')
    p.add_argument('--max_steps', type=int, default=None,
                   help='Episode cap (default: env max_episode_steps)')
    p.add_argument('--bon_n', type=int, default=8)
    p.add_argument(
        '--bon_n_list',
        default=None,
        help='Comma-separated BoN sample counts, e.g. 4,8,16,32,64 (overrides --bon_n)',
    )
    p.add_argument('--output_csv', default=None, help='Path to write per-episode and summary CSV')
    p.add_argument(
        '--fixed_seeds',
        action='store_true',
        help='Use deterministic reset_seed+i / seed+i (default: random per episode, like flow_bc eval)',
    )
    p.add_argument('--reset_seed', type=int, default=0,
                   help='With --fixed_seeds: env reset seed for episode i is reset_seed + i')
    p.add_argument('--seed', type=int, default=0,
                   help='With --fixed_seeds: policy RNG for episode i is seed + i')
    p.add_argument('--egl_device', type=int, default=None)
    p.add_argument(
        '--jax_platform',
        choices=['auto', 'cpu', 'gpu'],
        default='auto',
        help='JAX backend: auto uses cpu for multi-worker, gpu for single-worker',
    )
    p.add_argument('--jax_device', type=int, default=None)
    args = p.parse_args()
    bon_ns = parse_bon_ns(args.bon_n, args.bon_n_list)

    if args.egl_device is not None:
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.egl_device)

    jax_platform, num_workers, worker_gpu_ids = _resolve_parallel_config(
        args.jax_platform, args.num_workers, args.jax_device
    )

    import gymnasium
    import ogbench.manipspace  # noqa: F401
    from flow_bc.checkpoint import read_ckpt_meta

    policy_meta = read_ckpt_meta(args.policy_ckpt)
    chunk_size = policy_meta['chunk_size']
    env_name = policy_meta.get('eval_env_name', args.env_name)
    tmp_env = gymnasium.make(env_name)
    max_steps = args.max_steps or tmp_env.spec.max_episode_steps
    task_name = tmp_env.unwrapped.task_infos[args.task_id - 1]['task_name']
    tmp_env.close()

    randomize_resets = not args.fixed_seeds
    reset_mode = (
        f'random per episode (like flow_bc eval / collect_transitions)'
        if randomize_resets
        else f'fixed reset_seeds={args.reset_seed}..{args.reset_seed + args.num_episodes - 1}'
    )

    print(
        f'env={env_name} task_id={args.task_id} ({task_name}) '
        f'max_steps={max_steps} chunk_size={chunk_size} bon_ns={bon_ns} '
        f'advantage_mode={args.advantage_mode} episodes={args.num_episodes} '
        f'workers={num_workers} jax_platform={jax_platform} {reset_mode}'
    )

    bc_results, bon_results_by_k = parallel_evaluate(
        args.policy_ckpt,
        args.advantage_ckpt,
        args.advantage_mode,
        env_name,
        args.task_id,
        args.num_episodes,
        num_workers,
        max_steps,
        args.n_flow_steps,
        bon_ns,
        args.reset_seed,
        args.seed,
        randomize_resets,
        args.egl_device,
        jax_platform,
        worker_gpu_ids,
    )

    print(f'episodes={args.num_episodes}')
    summarize('baseline', bc_results)
    for bon_n in bon_ns:
        summarize('bon', bon_results_by_k[bon_n], bon_n)

    if args.output_csv:
        write_results_csv(args.output_csv, bc_results, bon_results_by_k)


if __name__ == '__main__':
    main()
