"""Collect transitions (s, a, s', mjstate) from flow-BC chunked rollouts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import gymnasium
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np

import ogbench.manipspace 

from bon_sampling.sim_state import get_sim_state
from flow_bc.checkpoint import load_flow_bc
from flow_bc.model import sample_action_chunk


def collect_rollout(
    env,
    params,
    apply_fn,
    task_id: int,
    key,
    chunk_size: int,
    act_dim: int,
    n_flow_steps: int,
    max_steps: int,
    goal_condition: bool,
    record_frames: bool = False,
):
    u = env.unwrapped
    obs_list, act_list, next_obs_list, state_list = [], [], [], []
    frames = []
    ob, info = env.reset(options=dict(task_id=task_id))
    if record_frames:
        frames.append(env.render())
    goal = info['goal'] if goal_condition else None
    steps = 0
    success = False

    while steps < max_steps:
        key, sample_key = jax.random.split(key)
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
            if steps >= max_steps:
                break
            action = chunk[k]
            next_ob, _, terminated, truncated, info = env.step(action)
            if record_frames:
                frames.append(env.render())
            obs_list.append(ob.copy())
            act_list.append(action.copy())
            next_obs_list.append(next_ob.copy())
            state_list.append(get_sim_state(u._model, u._data))
            ob = next_ob
            steps += 1
            if terminated or truncated:
                success = bool(info.get('success', False))
                return obs_list, act_list, next_obs_list, state_list, success, frames

    return obs_list, act_list, next_obs_list, state_list, success, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='flow_bc/checkpoints_cube_single_expert/best.pkl')
    p.add_argument('--n_flow_steps', type=int, default=10)
    p.add_argument('--env_name', default='cube-single-v0')
    p.add_argument('--task_id', type=int, default=1)
    p.add_argument('--num_rollouts', type=int, default=100)
    p.add_argument('--num_videos', type=int, default=10)
    p.add_argument('--video_fps', type=int, default=20)
    p.add_argument('--max_steps', type=int, default=None)
    p.add_argument('--output', default='bon_sampling/data/expert_bc_rollouts.npz')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    video_dir = Path(args.output).with_suffix('') / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)

    model, params, meta = load_flow_bc(args.checkpoint)
    chunk_size = meta['chunk_size']
    act_dim = meta['act_dim']
    goal_condition = meta['goal_condition']
    print(
        f'loaded flow BC: chunk_size={chunk_size} obs_dim={meta["obs_dim"]} act_dim={act_dim} '
        f'goal_condition={goal_condition}'
    )

    env = gymnasium.make(args.env_name)
    max_steps = args.max_steps or env.spec.max_episode_steps
    task_name = env.unwrapped.task_infos[args.task_id - 1]['task_name']
    goal_xyz = env.unwrapped.task_infos[args.task_id - 1]['goal_xyzs'][0].copy()
    print(f'env={args.env_name} task_id={args.task_id} ({task_name}) max_steps={max_steps}')

    all_obs, all_act, all_next_obs, all_state, episode_ends, successes = [], [], [], [], [], []
    key = jax.random.PRNGKey(int.from_bytes(os.urandom(4), 'big') & 0x7FFFFFFF)
    for i in range(args.num_rollouts):
        key, rollout_key = jax.random.split(key)
        record_frames = i < args.num_videos
        obs, acts, next_obs, states, success, frames = collect_rollout(
            env,
            params,
            model.apply,
            args.task_id,
            rollout_key,
            chunk_size,
            act_dim,
            args.n_flow_steps,
            max_steps,
            goal_condition,
            record_frames=record_frames,
        )
        all_obs.extend(obs)
        all_act.extend(acts)
        all_next_obs.extend(next_obs)
        all_state.extend(states)
        episode_ends.append(len(all_act))
        successes.append(success)
        print(f'rollout {i}: {len(acts)} transitions success={success}')
        if record_frames:
            video_path = video_dir / f'rollout_{i:02d}.mp4'
            with imageio.get_writer(
                video_path, fps=args.video_fps, codec='libx264', quality=8
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)
            print(f'  saved video -> {video_path} ({len(frames)} frames)')

    np.savez_compressed(
        args.output,
        observations=np.asarray(all_obs, np.float32),
        actions=np.asarray(all_act, np.float32),
        next_observations=np.asarray(all_next_obs, np.float32),
        next_mjstate=np.asarray(all_state, np.float64),
        goal_xyz=goal_xyz,
        task_id=np.array(args.task_id),
        episode_ends=np.asarray(episode_ends, np.int32),
        chunk_size=np.array(chunk_size),
        policy='flow_bc',
    )
    success_rate = float(np.mean(successes)) if successes else 0.0
    num_successes = int(np.sum(successes))
    print(f'saved {len(all_act)} transitions -> {args.output}')
    print(
        f'done: success_rate={success_rate:.3f} '
        f'({num_successes}/{args.num_rollouts})'
    )
    env.close()


if __name__ == '__main__':
    main()
