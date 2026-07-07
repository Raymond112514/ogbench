"""Evaluate a trained flow-BC checkpoint"""

from __future__ import annotations
import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault('MUJOCO_GL', 'egl')

from flow_bc.eval_worker import count_gpus


def _available_gpu_ids(jax_device: int | None) -> list[int]:
    n = count_gpus()
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


def _run_batch(
    worker_id: int,
    batch_size: int,
    ckpt_path: str,
    env_name: str,
    task_id: int,
    max_steps: int,
    n_flow_steps: int,
    egl_device: int | None,
    jax_platform: str,
    gpu_id: int | None,
) -> list[tuple[bool, int]]:
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
    import jax
    import jax.numpy as jnp

    import ogbench.manipspace 
    from flow_bc.checkpoint import load_flow_bc
    from flow_bc.model import sample_action_chunk

    model, params, meta = load_flow_bc(ckpt_path)
    env = gymnasium.make(env_name)
    results: list[tuple[bool, int]] = []

    for _ in range(batch_size):
        ob, info = env.reset(options=dict(task_id=task_id))
        goal = info['goal'] if meta['goal_condition'] else None
        key = jax.random.PRNGKey(int.from_bytes(os.urandom(4), 'big') & 0x7FFFFFFF)
        success, steps, done = False, 0, False

        while steps < max_steps and not done:
            key, sample_key = jax.random.split(key)
            chunk = np.asarray(
                sample_action_chunk(
                    params,
                    model.apply,
                    jnp.asarray(ob),
                    jnp.asarray(goal) if meta['goal_condition'] else None,
                    sample_key,
                    chunk_size=meta['chunk_size'],
                    act_dim=meta['act_dim'],
                    n_flow_steps=n_flow_steps,
                    goal_condition=meta['goal_condition'],
                )
            )
            for action in chunk:
                if steps >= max_steps:
                    break
                ob, _, term, trunc, info = env.step(np.clip(action, -1.0, 1.0))
                steps += 1
                success = bool(info.get('success', False))
                if term or trunc:
                    done = True
                    break
        results.append((success, steps))

    env.close()
    print(f'worker {worker_id}: finished {batch_size} episodes', flush=True)
    return results


def parallel_evaluate(
    ckpt_path: str,
    env_name: str,
    task_id: int,
    num_episodes: int,
    num_workers: int,
    max_steps: int,
    n_flow_steps: int,
    egl_device: int | None,
    jax_platform: str,
    worker_gpu_ids: list[int],
) -> list[tuple[bool, int]]:
    batches = [len(b) for b in np.array_split(np.arange(num_episodes), num_workers)]
    results: list[tuple[bool, int]] = []
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
        futures = [
            pool.submit(
                _run_batch,
                wid,
                batch_size,
                str(Path(ckpt_path).resolve()),
                env_name,
                task_id,
                max_steps,
                n_flow_steps,
                egl_device,
                jax_platform,
                worker_gpu_ids[wid] if worker_gpu_ids else None,
            )
            for wid, batch_size in enumerate(batches)
            if batch_size > 0
        ]
        for fut in futures:
            results.extend(fut.result())
    return results


def summarize(name: str, results: list[tuple[bool, int]]) -> None:
    successes = np.array([r[0] for r in results], dtype=np.float32)
    lengths = np.array([r[1] for r in results], dtype=np.float32)
    print(
        f'{name}: success={successes.mean():.3f} ({successes.sum():.0f}/{len(results)}) '
        f'avg_len={lengths.mean():.1f}'
    )


def main():
    p = argparse.ArgumentParser(description='Evaluate a Flow-BC policy (randomized resets).')
    p.add_argument('--ckpt', default='flow_bc/checkpoints_cube_single/best.pkl')
    p.add_argument('--env_name', default='cube-single-v0')
    p.add_argument('--task_id', type=int, default=1)
    p.add_argument('--num_episodes', type=int, default=100)
    p.add_argument('--num_workers', type=int, default=10)
    p.add_argument('--n_flow_steps', type=int, default=10)
    p.add_argument('--max_steps', type=int, default=None)
    p.add_argument('--egl_device', type=int, default=None)
    p.add_argument(
        '--jax_platform',
        choices=['auto', 'cpu', 'gpu'],
        default='auto',
        help='JAX backend: auto uses cpu for multi-worker, gpu for single-worker',
    )
    p.add_argument('--jax_device', type=int, default=None)
    args = p.parse_args()

    if args.egl_device is not None:
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.egl_device)

    jax_platform, num_workers, worker_gpu_ids = _resolve_parallel_config(
        args.jax_platform, args.num_workers, args.jax_device
    )

    import gymnasium
    import ogbench.manipspace  
    from flow_bc.checkpoint import read_ckpt_meta

    meta = read_ckpt_meta(args.ckpt)
    env_name = meta.get('eval_env_name', args.env_name)
    tmp_env = gymnasium.make(env_name)
    max_steps = args.max_steps or tmp_env.spec.max_episode_steps
    task_name = tmp_env.unwrapped.task_infos[args.task_id - 1]['task_name']
    tmp_env.close()

    print(
        f'ckpt={args.ckpt} env={env_name} task_id={args.task_id} ({task_name}) '
        f'max_steps={max_steps} chunk_size={meta["chunk_size"]} '
        f'goal_condition={meta["goal_condition"]} episodes={args.num_episodes} '
        f'workers={num_workers} jax_platform={jax_platform} (randomized resets)'
    )

    results = parallel_evaluate(
        args.ckpt,
        env_name,
        args.task_id,
        args.num_episodes,
        num_workers,
        max_steps,
        args.n_flow_steps,
        args.egl_device,
        jax_platform,
        worker_gpu_ids,
    )
    summarize('Flow-BC', results)


if __name__ == '__main__':
    main()
