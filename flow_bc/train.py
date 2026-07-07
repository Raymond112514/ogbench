"""Train a flow-BC policy"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'impls'))

os.environ.setdefault('MUJOCO_GL', 'egl')

from flow_bc.eval_worker import count_gpus

PRESETS = {
    'cube_single_gcbc': dict(
        env_name='cube-single-play-v0',
        eval_env_name='cube-single-v0',
        task_id=1,
        goal_condition=True,
        checkpoint_dir='flow_bc/checkpoints/cube_single_gcbc',
    ),
    'cube_double_gcbc': dict(
        env_name='cube-double-play-v0',
        eval_env_name='cube-double-v0',
        task_id=2,
        goal_condition=True,
        checkpoint_dir='flow_bc/checkpoints/cube_double_gcbc',
    ),
}


def _configure_jax_platform():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        '--jax_platform',
        choices=['auto', 'cpu', 'gpu'],
        default='auto',
        help='JAX backend: auto (default), cpu, or gpu/cuda',
    )
    pre.add_argument(
        '--jax_device',
        type=int,
        default=None,
        help='Pin JAX to one GPU via CUDA_VISIBLE_DEVICES (set before JAX import)',
    )
    known, _ = pre.parse_known_args()
    if known.jax_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(known.jax_device)
    if known.jax_platform == 'cpu':
        os.environ['JAX_PLATFORMS'] = 'cpu'
    elif known.jax_platform == 'gpu':
        os.environ['JAX_PLATFORMS'] = 'cuda'
    elif known.jax_platform == 'auto' and count_gpus() > 0:
        os.environ.setdefault('JAX_PLATFORMS', 'cuda')


_configure_jax_platform()

import gymnasium
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm import trange

import ogbench
import ogbench.manipspace  # noqa: F401

from flow_bc.checkpoint import save_ckpt
from flow_bc.dataset import ChunkedGCDataset, load_npz_dataset
from flow_bc.eval_worker import assign_egl_gpus, _start_env_worker
from flow_bc.model import VelocityNet, cond_dim, train_step, sample_action_chunk


def load_dataset(env_name, chunk_size, seed):
    _, raw, _ = ogbench.make_env_and_datasets(env_name, compact_dataset=True)
    ds = ChunkedGCDataset(
        observations=raw['observations'],
        actions=raw['actions'],
        terminals=raw['terminals'],
        chunk_size=chunk_size,
        seed=seed,
    )
    print(
        f'dataset: {len(ds.valid_idxs)} valid steps, obs_dim={ds.observations.shape[1]}, '
        f'act_dim={ds.actions.shape[1]}'
    )
    return ds


def evaluate(
    params,
    apply_fn,
    env,
    task_id,
    num_episodes,
    chunk_size,
    act_dim,
    n_flow_steps,
    rng,
    goal_condition=True,
    video_episodes=1,
):
    successes = []
    ep_lengths = []
    frames = []
    key = rng
    for ep in range(num_episodes):
        key, ep_key = jax.random.split(key)
        ob, info = env.reset(options=dict(task_id=task_id))
        record = ep < video_episodes
        if record:
            frames.append(env.render())
        goal = info['goal'] if goal_condition else None
        done = False
        steps = 0
        success = False
        chunk_key = ep_key

        while not done:
            chunk_key, sample_key = jax.random.split(chunk_key)
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
            for k in range(chunk_size):
                ob, _, term, trunc, info = env.step(chunk[k])
                if record:
                    frames.append(env.render())
                steps += 1
                done = term or trunc
                success = bool(info.get('success', False))
                if done:
                    break

        successes.append(float(success))
        ep_lengths.append(steps)

    return float(np.mean(successes)), float(np.mean(ep_lengths)), frames


def parallel_evaluate(
    params,
    apply_fn,
    eval_env_name: str,
    task_id: int,
    num_episodes: int,
    num_workers: int,
    chunk_size: int,
    act_dim: int,
    n_flow_steps: int,
    goal_condition: bool,
    max_steps: int,
    egl_device: int | None,
    video_episodes: int,
    exclude_gpus: tuple[int, ...],
    rng,
) -> tuple[float, float, list]:
    num_workers = min(num_workers, num_episodes)
    egl_gpus = assign_egl_gpus(num_workers, egl_device, exclude_gpus=exclude_gpus)
    ctx = mp.get_context('spawn')

    pipes: list = []
    processes: list[mp.Process] = []
    for wid in range(num_workers):
        conn, proc = _start_env_worker(
            ctx,
            eval_env_name,
            task_id,
            max_steps,
            goal_condition,
            egl_gpus[wid],
        )
        pipes.append(conn)
        processes.append(proc)

    episode_queue = list(range(num_episodes))
    video_budget = video_episodes
    active: dict[int, dict] = {}
    all_results: list[tuple[bool, int]] = []
    all_frames: list[np.ndarray] = []
    key = rng

    def launch_episode(wid: int) -> None:
        nonlocal video_budget
        if not episode_queue:
            return
        ep_idx = episode_queue.pop(0)
        record = video_budget > 0
        if record:
            video_budget -= 1
        key_local = jax.random.fold_in(key, ep_idx)
        active[wid] = {'record': record, 'key': key_local}
        pipes[wid].send(('episode', record))

    for wid in range(num_workers):
        launch_episode(wid)

    try:
        while active:
            progressed = False
            for wid in list(active.keys()):
                if not pipes[wid].poll():
                    continue
                progressed = True
                msg = pipes[wid].recv()
                if msg[0] == 'chunk_request':
                    ob, goal = msg[1], msg[2]
                    ep_key = active[wid]['key']
                    ep_key, sample_key = jax.random.split(ep_key)
                    active[wid]['key'] = ep_key
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
                    pipes[wid].send(chunk)
                elif msg[0] == 'episode_done':
                    success, steps, frames = msg[1], msg[2], msg[3]
                    all_results.append((success, steps))
                    if frames:
                        all_frames.extend(frames)
                    del active[wid]
                    launch_episode(wid)
            if not progressed:
                time.sleep(0.001)
    finally:
        for conn in pipes:
            try:
                conn.send(('stop',))
            except (BrokenPipeError, OSError):
                pass
            conn.close()
        for proc in processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()

    successes = np.array([r[0] for r in all_results], dtype=np.float32)
    lengths = np.array([r[1] for r in all_results], dtype=np.float32)
    return float(successes.mean()), float(lengths.mean()), all_frames


def _apply_preset(args):
    if not args.preset:
        return
    explicit = set()
    for token in sys.argv[1:]:
        if not token.startswith('--'):
            continue
        name = token.split('=')[0][2:]
        explicit.add(name.replace('-', '_'))
        if name in ('no_goal_condition', 'no-goal_condition'):
            explicit.add('goal_condition')

    for key, value in PRESETS[args.preset].items():
        if key not in explicit:
            setattr(args, key, value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--preset',
        choices=sorted(PRESETS),
        default=None,
        help='Training recipe: cube_single_gcbc or cube_double_gcbc (play data, goal-conditioned)',
    )
    p.add_argument('--env_name', default='cube-single-play-v0')
    p.add_argument('--dataset_npz', default=None, help='Offline npz instead of OGBench play data')
    p.add_argument('--eval_env_name', default='cube-single-v0')
    p.add_argument('--task_id', type=int, default=1)
    p.add_argument('--chunk_size', type=int, default=4)
    p.add_argument('--train_steps', type=int, default=100000)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 512, 512])
    p.add_argument('--n_flow_steps', type=int, default=10)
    p.add_argument('--eval_interval', type=int, default=5000)
    p.add_argument('--eval_episodes', type=int, default=10)
    p.add_argument(
        '--eval_workers',
        type=int,
        default=0,
        help='Parallel eval envs (0 = match eval_episodes; 1 = sequential in training process)',
    )
    p.add_argument('--no_eval_video', action='store_true', help='Skip eval video for faster eval')
    p.add_argument('--eval_fps', type=int, default=20)
    p.add_argument('--log_interval', type=int, default=500)
    p.add_argument('--save_interval', type=int, default=10000)
    p.add_argument('--checkpoint_dir', default='flow_bc/checkpoints')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--egl_device', type=int, default=None)
    p.add_argument(
        '--jax_platform',
        choices=['auto', 'cpu', 'gpu'],
        default='auto',
        help='JAX backend: auto (default), cpu, or gpu/cuda',
    )
    p.add_argument(
        '--jax_device',
        type=int,
        default=None,
        help='Pin JAX to one GPU via CUDA_VISIBLE_DEVICES',
    )
    p.set_defaults(goal_condition=True)
    p.add_argument('--goal_condition', action='store_true', dest='goal_condition')
    p.add_argument('--no_goal_condition', action='store_false', dest='goal_condition')
    p.add_argument('--no-goal_condition', action='store_false', dest='goal_condition')
    args = p.parse_args()
    _apply_preset(args)

    if args.egl_device is not None:
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.egl_device)

    if args.dataset_npz:
        dataset = load_npz_dataset(
            args.dataset_npz, args.chunk_size, args.seed, goal_condition=args.goal_condition
        )
    else:
        if not args.goal_condition:
            raise SystemExit('--no_goal_condition requires --dataset_npz')
        dataset = load_dataset(args.env_name, args.chunk_size, args.seed)

    obs_dim = dataset.observations.shape[1]
    act_dim = dataset.actions.shape[1]
    out_dim = args.chunk_size * act_dim
    input_cond_dim = cond_dim(obs_dim, goal_condition=args.goal_condition)

    model = VelocityNet(hidden_dims=tuple(args.hidden_dims), out_dim=out_dim)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = model.init(
        init_key,
        jnp.zeros((1, out_dim)),
        jnp.zeros((1,)),
        jnp.zeros((1, input_cond_dim)),
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(args.lr),
    )

    env = gymnasium.make(args.eval_env_name)
    max_eval_steps = env.spec.max_episode_steps
    eval_video_dir = os.path.join(args.checkpoint_dir, 'eval')
    os.makedirs(eval_video_dir, exist_ok=True)
    task_name = env.unwrapped.task_infos[args.task_id - 1]['task_name']
    eval_workers = args.eval_episodes if args.eval_workers == 0 else args.eval_workers
    eval_workers = max(1, min(eval_workers, args.eval_episodes))
    eval_video_episodes = 0 if args.no_eval_video else (1 if eval_workers > 1 else args.eval_episodes)
    exclude_gpus: tuple[int, ...]
    if args.jax_device is not None:
        exclude_gpus = (args.jax_device,)
    elif jax.default_backend() == 'gpu':
        exclude_gpus = (0,)
    else:
        exclude_gpus = ()
    best_success = -1.0
    extra_meta = {
        'env_name': args.env_name,
        'eval_env_name': args.eval_env_name,
        'task_id': args.task_id,
        'task_name': task_name,
        'dataset_npz': args.dataset_npz,
    }

    train_src = args.dataset_npz or args.env_name
    print(f'jax: platform={args.jax_platform} backend={jax.default_backend()} devices={jax.devices()}')
    print(
        f'flow BC | train={train_src} | eval={args.eval_env_name} task_id={args.task_id} ({task_name})'
    )
    print(
        f'chunk={args.chunk_size} | obs={obs_dim} | act={act_dim} | cond={input_cond_dim} | '
        f'goal_condition={args.goal_condition}'
    )
    print(
        f'eval: {args.eval_episodes} episodes, {eval_workers} env worker(s), '
        f'video={"off" if eval_video_episodes == 0 else f"{eval_video_episodes} ep(s)"}'
    )

    for step in trange(1, args.train_steps + 1, desc='train'):
        key, step_key = jax.random.split(key)
        batch = dataset.sample(args.batch_size)
        batch_jax = {k: jnp.asarray(v) for k, v in batch.items()}
        state, loss = train_step(
            state,
            batch_jax,
            step_key,
            goal_condition=args.goal_condition,
            advantage_condition=False,
        )

        if step % args.log_interval == 0 or step == 1:
            print(f'step {step}: loss={float(loss):.4f}')

        if step % args.eval_interval == 0 or step == 1:
            key, eval_key = jax.random.split(key)
            if eval_workers <= 1:
                sr, avg_len, eval_frames = evaluate(
                    state.params,
                    state.apply_fn,
                    env,
                    args.task_id,
                    args.eval_episodes,
                    args.chunk_size,
                    act_dim,
                    args.n_flow_steps,
                    eval_key,
                    goal_condition=args.goal_condition,
                    video_episodes=eval_video_episodes,
                )
            else:
                sr, avg_len, eval_frames = parallel_evaluate(
                    state.params,
                    state.apply_fn,
                    args.eval_env_name,
                    args.task_id,
                    args.eval_episodes,
                    eval_workers,
                    args.chunk_size,
                    act_dim,
                    args.n_flow_steps,
                    args.goal_condition,
                    max_eval_steps,
                    args.egl_device,
                    eval_video_episodes,
                    exclude_gpus,
                    eval_key,
                )
            eval_video_path = os.path.join(eval_video_dir, f'step_{step:07d}.mp4')
            if eval_video_episodes > 0 and eval_frames:
                with imageio.get_writer(
                    eval_video_path, fps=args.eval_fps, codec='libx264', quality=8
                ) as writer:
                    for frame in eval_frames:
                        writer.append_data(frame)
                video_msg = f'video={eval_video_path} ({len(eval_frames)} frames)'
            else:
                video_msg = 'video=skipped'
            print(
                f'  eval step {step}: success={sr:.3f} '
                f'({int(sr * args.eval_episodes)}/{args.eval_episodes}) avg_len={avg_len:.1f} '
                f'{video_msg}'
            )
            if sr > best_success:
                best_success = sr
                save_ckpt(
                    os.path.join(args.checkpoint_dir, 'best.pkl'),
                    state.params,
                    obs_dim,
                    act_dim,
                    args.chunk_size,
                    args.hidden_dims,
                    step,
                    goal_condition=args.goal_condition,
                    advantage_condition=False,
                    extra_meta=extra_meta,
                )

        if step % args.save_interval == 0:
            save_ckpt(
                os.path.join(args.checkpoint_dir, f'step_{step:07d}.pkl'),
                state.params,
                obs_dim,
                act_dim,
                args.chunk_size,
                args.hidden_dims,
                step,
                goal_condition=args.goal_condition,
                advantage_condition=False,
                extra_meta=extra_meta,
            )

    save_ckpt(
        os.path.join(args.checkpoint_dir, 'final.pkl'),
        state.params,
        obs_dim,
        act_dim,
        args.chunk_size,
        args.hidden_dims,
        args.train_steps,
        goal_condition=args.goal_condition,
        advantage_condition=False,
        extra_meta=extra_meta,
    )
    print(f'done. best success={best_success:.3f}')
    env.close()


if __name__ == '__main__':
    main()
