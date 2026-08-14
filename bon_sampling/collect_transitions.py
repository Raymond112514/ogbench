"""
Collect transitions from flow-BC rollouts

MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  python bon_sampling/collect_transitions.py \
  --checkpoint flow_bc/checkpoints/cube_single_gcbc/best.pkl \
  --num_rollouts 100 \
  --num_workers 10 \
  --jax_platform cpu
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

NUM_VIDEO_EPISODES = 10


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


def _random_policy_key():
    import jax

    return jax.random.PRNGKey(int.from_bytes(os.urandom(4), 'big') & 0x7FFFFFFF)


def collect_rollout(
    env,
    params,
    apply_fn,
    task_id: int,
    chunk_size: int,
    act_dim: int,
    n_flow_steps: int,
    max_steps: int,
    goal_condition: bool,
    record_frames: bool = False,
    bon=None,
    action_fn=None,
):
    import jax.numpy as jnp

    from bon_sampling.sim_state import get_sim_state
    from flow_bc.model import sample_action_chunk

    u = env.unwrapped
    obs_list, act_list, next_obs_list, state_list, success_list = [], [], [], [], []
    frames = []
    ob, info = env.reset(options=dict(task_id=task_id))
    initial_mjstate = get_sim_state(u._model, u._data)
    if record_frames:
        frames.append(env.render())
    goal = info['goal'] if goal_condition else None
    key = _random_policy_key()
    steps = 0
    success = False

    while steps < max_steps:
        import jax

        key, sample_key = jax.random.split(key)
        if action_fn is not None:
            chunk = np.asarray(action_fn(jnp.asarray(ob), sample_key))
        elif bon is None:
            chunk = np.asarray(
                sample_action_chunk(
                    params,
                    apply_fn,
                    jnp.asarray(ob),
                    jnp.asarray(goal) if goal_condition else None,
                    sample_key,
                    chunk_size=chunk_size,
                    act_dim=act_dim,
                    n_flow_steps=n_flow_steps,
                    goal_condition=goal_condition,
                )
            )
        else:
            sample_candidates_fn, select_fn, adv_params = bon
            candidates = sample_candidates_fn(
                params, jnp.asarray(ob), jnp.asarray(goal) if goal_condition else None, sample_key
            )
            pick = int(select_fn(adv_params, jnp.asarray(ob), candidates))
            chunk = np.asarray(candidates[pick])
        for k in range(chunk_size):
            if steps >= max_steps:
                break
            action = chunk[k]
            next_ob, _, terminated, truncated, info = env.step(action)
            step_success = bool(info.get('success', False))
            if record_frames:
                frames.append(env.render())
            obs_list.append(ob.copy())
            act_list.append(action.copy())
            next_obs_list.append(next_ob.copy())
            state_list.append(get_sim_state(u._model, u._data))
            success_list.append(step_success)
            ob = next_ob
            steps += 1
            if terminated or truncated:
                success = step_success
                return obs_list, act_list, next_obs_list, state_list, success_list, success, frames, initial_mjstate

    return obs_list, act_list, next_obs_list, state_list, success_list, success, frames, initial_mjstate


def _run_batch(
    worker_id: int,
    episode_indices: np.ndarray,
    ckpt_path: str,
    env_name: str,
    task_id: int,
    max_steps: int,
    n_flow_steps: int,
    egl_device: int | None,
    jax_platform: str,
    gpu_id: int | None,
    advantage_ckpt: str | None,
    advantage_mode: str,
    bon_n: int,
    select_mode: str = 'bon',
    q_ascent_steps: int = 10,
    q_ascent_lr: float = 0.1,
) -> list[dict]:
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

    model, params, meta = load_flow_bc(ckpt_path)
    chunk_size = meta['chunk_size']
    act_dim = meta['act_dim']
    goal_condition = meta['goal_condition']

    bon = None
    action_fn = None
    if advantage_ckpt is not None:
        import pickle

        with open(advantage_ckpt, 'rb') as f:
            ckpt_meta = pickle.load(f)
        if ckpt_meta.get('mode') == 'iql' and select_mode == 'q_ascent':
            from bon_sampling.iql.train import load_iql, make_q_ascent_fn

            agent, _ = load_iql(advantage_ckpt)
            action_fn = make_q_ascent_fn(
                agent, chunk_size, act_dim, n_steps=q_ascent_steps, lr=q_ascent_lr,
            )
        else:
            from bon_sampling.advantage.eval import _make_sample_candidates_fn

            sample_candidates_fn = _make_sample_candidates_fn(
                model.apply, bon_n, chunk_size, act_dim, n_flow_steps, goal_condition
            )
            if ckpt_meta.get('mode') == 'iql':
                from bon_sampling.iql.train import load_iql, make_select_fn

                agent, _ = load_iql(advantage_ckpt)
                select_fn = make_select_fn(agent)
                bon = (sample_candidates_fn, select_fn, agent.network.params)
            else:
                from bon_sampling.advantage.eval import _load_advantage, _make_select_fn

                adv_model, adv_params, adv_mode, _ = _load_advantage(advantage_ckpt, advantage_mode)
                select_fn = _make_select_fn(adv_model, adv_mode)
                bon = (sample_candidates_fn, select_fn, adv_params)

    env = gymnasium.make(env_name)
    results: list[dict] = []

    for ep in episode_indices:
        ep = int(ep)
        record_frames = ep < NUM_VIDEO_EPISODES
        obs, acts, next_obs, states, step_successes, success, frames, initial_mjstate = collect_rollout(
            env,
            params,
            model.apply,
            task_id,
            chunk_size,
            act_dim,
            n_flow_steps,
            max_steps,
            goal_condition,
            record_frames=record_frames,
            bon=bon,
            action_fn=action_fn,
        )
        results.append(
            {
                'ep': ep,
                'observations': obs,
                'actions': acts,
                'next_observations': next_obs,
                'next_mjstate': states,
                'initial_mjstate': initial_mjstate,
                'step_successes': step_successes,
                'success': success,
                'frames': frames if record_frames else [],
            }
        )
        print(f'worker {worker_id} rollout {ep}: {len(acts)} transitions success={success}', flush=True)

    env.close()
    print(f'worker {worker_id}: finished {len(episode_indices)} episodes', flush=True)
    return results


def parallel_collect(
    ckpt_path: str,
    env_name: str,
    task_id: int,
    num_rollouts: int,
    num_workers: int,
    max_steps: int,
    n_flow_steps: int,
    egl_device: int | None,
    jax_platform: str,
    worker_gpu_ids: list[int],
    advantage_ckpt: str | None = None,
    advantage_mode: str = 'auto',
    bon_n: int = 8,
    select_mode: str = 'bon',
    q_ascent_steps: int = 10,
    q_ascent_lr: float = 0.1,
) -> tuple[list, list, list, list, list, list, list, list, list]:
    episode_splits = np.array_split(np.arange(num_rollouts), num_workers)
    merged: list[dict] = []
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
        futures = [
            pool.submit(
                _run_batch,
                wid,
                split,
                str(Path(ckpt_path).resolve()),
                env_name,
                task_id,
                max_steps,
                n_flow_steps,
                egl_device,
                jax_platform,
                worker_gpu_ids[wid] if worker_gpu_ids else None,
                advantage_ckpt,
                advantage_mode,
                bon_n,
                select_mode,
                q_ascent_steps,
                q_ascent_lr,
            )
            for wid, split in enumerate(episode_splits)
            if len(split) > 0
        ]
        for fut in futures:
            merged.extend(fut.result())

    merged.sort(key=lambda r: r['ep'])

    all_obs, all_act, all_next_obs, all_state, all_step_succ = [], [], [], [], []
    episode_ends, successes = [], []
    episode_initial_mjstate: list[np.ndarray] = []
    video_frames: list[np.ndarray] = []
    for row in merged:
        all_obs.extend(row['observations'])
        all_act.extend(row['actions'])
        all_next_obs.extend(row['next_observations'])
        all_state.extend(row['next_mjstate'])
        all_step_succ.extend(row['step_successes'])
        episode_ends.append(len(all_act))
        successes.append(row['success'])
        episode_initial_mjstate.append(row['initial_mjstate'])
        if row['frames']:
            video_frames.extend(row['frames'])

    return (
        all_obs, all_act, all_next_obs, all_state, all_step_succ,
        episode_ends, successes, video_frames, episode_initial_mjstate,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='flow_bc/checkpoints_cube_single/best.pkl')
    p.add_argument('--n_flow_steps', type=int, default=10)
    p.add_argument('--env_name', default='cube-single-v0')
    p.add_argument('--task_id', type=int, default=1)
    p.add_argument('--num_rollouts', type=int, default=100)
    p.add_argument('--num_workers', type=int, default=10)
    p.add_argument('--video_fps', type=int, default=20)
    p.add_argument('--max_steps', type=int, default=None)
    p.add_argument('--output_dir', default='bon_sampling/data')
    p.add_argument('--egl_device', type=int, default=None)
    p.add_argument(
        '--jax_platform',
        choices=['auto', 'cpu', 'gpu'],
        default='auto',
        help='JAX backend: auto uses cpu for multi-worker, gpu for single-worker',
    )
    p.add_argument('--jax_device', type=int, default=None)
    p.add_argument('--advantage_ckpt', default=None, help='Classifier checkpoint; enables BoN action selection')
    p.add_argument('--advantage_mode', choices=('classifier', 'regression', 'bin_classifier', 'auto'), default='auto')
    p.add_argument('--bon_n', type=int, default=8, help='Number of candidate chunks per BoN step')
    args = p.parse_args()

    if args.egl_device is not None:
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.egl_device)

    jax_platform, num_workers, worker_gpu_ids = _resolve_parallel_config(
        args.jax_platform, args.num_workers, args.jax_device
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'flowbc_rollouts_{args.num_rollouts}'
    npz_path = output_dir / f'{stem}.npz'
    mp4_path = output_dir / f'{stem}.mp4'

    import gymnasium
    import imageio.v2 as imageio

    import ogbench.manipspace  # noqa: F401
    from flow_bc.checkpoint import read_ckpt_meta

    meta = read_ckpt_meta(args.checkpoint)
    tmp_env = gymnasium.make(args.env_name)
    max_steps = args.max_steps or tmp_env.spec.max_episode_steps
    task_name = tmp_env.unwrapped.task_infos[args.task_id - 1]['task_name']
    goal_xyz = tmp_env.unwrapped.task_infos[args.task_id - 1]['goal_xyzs'][0].copy()
    tmp_env.close()

    bon_str = f'BoN(n={args.bon_n}, ckpt={args.advantage_ckpt})' if args.advantage_ckpt else 'off'
    print(
        f'checkpoint={args.checkpoint} env={args.env_name} task_id={args.task_id} ({task_name}) '
        f'max_steps={max_steps} chunk_size={meta["chunk_size"]} '
        f'goal_condition={meta["goal_condition"]} rollouts={args.num_rollouts} '
        f'workers={num_workers} jax_platform={jax_platform} bon={bon_str} (randomized resets)'
    )

    all_obs, all_act, all_next_obs, all_state, all_step_succ, episode_ends, successes, video_frames, episode_initial_mjstate = (
        parallel_collect(
            args.checkpoint,
            args.env_name,
            args.task_id,
            args.num_rollouts,
            num_workers,
            max_steps,
            args.n_flow_steps,
            args.egl_device,
            jax_platform,
            worker_gpu_ids,
            args.advantage_ckpt,
            args.advantage_mode,
            args.bon_n,
        )
    )

    np.savez_compressed(
        npz_path,
        observations=np.asarray(all_obs, np.float32),
        actions=np.asarray(all_act, np.float32),
        next_observations=np.asarray(all_next_obs, np.float32),
        next_mjstate=np.asarray(all_state, np.float64),
        successes=np.asarray(all_step_succ, np.bool_),
        episode_initial_mjstate=np.asarray(episode_initial_mjstate, np.float64),
        goal_xyz=goal_xyz,
        task_id=np.array(args.task_id),
        episode_ends=np.asarray(episode_ends, np.int32),
        chunk_size=np.array(meta['chunk_size']),
        policy='bon' if args.advantage_ckpt else 'flow_bc',
    )

    if video_frames:
        with imageio.get_writer(
            mp4_path, fps=args.video_fps, codec='libx264', quality=8
        ) as writer:
            for frame in video_frames:
                writer.append_data(frame)
        print(
            f'saved video -> {mp4_path} ({len(video_frames)} frames, '
            f'first {NUM_VIDEO_EPISODES} episodes)'
        )

    success_rate = float(np.mean(successes)) if successes else 0.0
    num_successes = int(np.sum(successes))
    print(f'saved {len(all_act)} transitions -> {npz_path}')
    print(
        f'done: success_rate={success_rate:.3f} '
        f'({num_successes}/{args.num_rollouts})'
    )


if __name__ == '__main__':
    main()
