"""Collect oracle expert demonstrations"""

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
import mujoco
import numpy as np

import ogbench.manipspace  
from bon_sampling.sim_state import get_sim_state
from ogbench.manipspace import lie
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle


def set_oracle_target_to_goal(env: gymnasium.Env, goal_xyz: np.ndarray) -> None:
    env_u = env.unwrapped
    env_u._target_block = 0
    env_u._target_task = 'cube'
    mocap_id = env_u._cube_target_mocap_ids[0]
    env_u._data.mocap_pos[mocap_id] = goal_xyz
    env_u._data.mocap_quat[mocap_id] = lie.SO3.identity().wxyz
    mujoco.mj_forward(env_u._model, env_u._data)
    env_u.pre_step()
    env_u.post_step()


def sync_oracle_env_from_layout(layout_env, oracle_env, goal_xyz: np.ndarray):
    """Copy task-layout physics into oracle env and pin target to task goal."""
    qpos = layout_env.unwrapped._data.qpos.copy()
    qvel = layout_env.unwrapped._data.qvel.copy()
    oracle_env.reset()
    oracle_env.unwrapped.set_state(qpos, qvel)
    set_oracle_target_to_goal(oracle_env, goal_xyz)


def collect_oracle_episode(
    layout_env,
    oracle_env,
    oracle,
    task_id: int,
    goal_xyz: np.ndarray,
    max_steps: int,
):
    u = oracle_env.unwrapped
    layout_env.reset(options=dict(task_id=task_id))
    sync_oracle_env_from_layout(layout_env, oracle_env, goal_xyz)

    ob = u.compute_observation()
    info = u.get_reset_info()
    oracle.reset(ob, info)

    obs_list, act_list, next_obs_list, state_list, frames = [], [], [], [], []
    success = False

    frames.append(oracle_env.render())

    for _ in range(max_steps):
        if oracle.done or u._success:
            success = bool(u._success)
            break
        action = np.clip(np.asarray(oracle.select_action(ob, info)), -1.0, 1.0)
        next_ob, _, terminated, truncated, info = oracle_env.step(action)
        obs_list.append(ob.copy())
        act_list.append(action.copy())
        next_obs_list.append(next_ob.copy())
        state_list.append(get_sim_state(u._model, u._data))
        frames.append(oracle_env.render())
        ob = next_ob
        if terminated or truncated:
            success = bool(info.get('success', False))
            break

    return obs_list, act_list, next_obs_list, state_list, frames, success


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env_name', default='cube-single-v0')
    p.add_argument('--task_id', type=int, default=1)
    p.add_argument('--num_episodes', type=int, default=100)
    p.add_argument('--chunk_size', type=int, default=4)
    p.add_argument('--max_steps', type=int, default=None)
    p.add_argument('--output', default='flow_bc/data/expert_task1.npz')
    p.add_argument('--video_output', default=None)
    p.add_argument('--fps', type=int, default=20)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    video_output = args.video_output or str(Path(args.output).with_suffix('.mp4'))

    layout_env = gymnasium.make(args.env_name, terminate_at_goal=True)
    oracle_env = gymnasium.make(
        args.env_name, mode='data_collection', terminate_at_goal=True
    )
    oracle = CubePlanOracle(env=oracle_env, noise=0.0, noise_smoothing=0.0)

    max_steps = args.max_steps or layout_env.spec.max_episode_steps
    task_info = layout_env.unwrapped.task_infos[args.task_id - 1]
    task_name = task_info['task_name']
    goal_xyz = task_info['goal_xyzs'][0].copy()
    print(f'collect expert demos: env={args.env_name} task_id={args.task_id} ({task_name})')

    all_obs, all_act, all_next_obs, all_state, episode_ends, all_frames = [], [], [], [], [], []
    attempts = 0
    saved = 0

    while saved < args.num_episodes:
        attempts += 1
        obs, acts, next_obs, states, frames, success = collect_oracle_episode(
            layout_env,
            oracle_env,
            oracle,
            args.task_id,
            goal_xyz,
            max_steps,
        )
        if not success:
            print(f'attempt {attempts}: failed ({len(acts)} steps), retrying')
            continue

        all_obs.extend(obs)
        all_act.extend(acts)
        all_next_obs.extend(next_obs)
        all_state.extend(states)
        all_frames.extend(frames)
        episode_ends.append(len(all_act))
        saved += 1
        print(f'episode {saved}/{args.num_episodes}: {len(acts)} steps (attempt {attempts})')

    np.savez_compressed(
        args.output,
        observations=np.asarray(all_obs, np.float32),
        actions=np.asarray(all_act, np.float32),
        next_observations=np.asarray(all_next_obs, np.float32),
        next_mjstate=np.asarray(all_state, np.float64),
        goal_xyz=goal_xyz,
        task_id=np.array(args.task_id),
        episode_ends=np.asarray(episode_ends, np.int32),
        chunk_size=np.array(args.chunk_size),
        policy='oracle',
    )
    print(f'saved {len(all_act)} transitions from {saved} episodes -> {args.output}')

    os.makedirs(os.path.dirname(video_output) or '.', exist_ok=True)
    with imageio.get_writer(video_output, fps=args.fps, codec='libx264', quality=8) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    print(f'saved {len(all_frames)} frames from {saved} episodes -> {video_output}')

    layout_env.close()
    oracle_env.close()


if __name__ == '__main__':
    main()
