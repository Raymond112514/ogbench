"""
Verify mjstate recovery by replaying policy chunks then oracle rollouts per replan.

MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  python bon_sampling/visualization/verify_oracle_recovery.py \
  --input bon_sampling/data/flowbc_rollouts_100_annotated.npz \
  --output_dir bon_sampling/visualization/flowbc_rollouts_100 \
  --num_traj 10 \
  --warmup_steps 0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import gymnasium
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

import ogbench.manipspace  # noqa: F401
from bon_sampling.oracle_utils import oracle_seed, restore_sim_state
from bon_sampling.sim_state import set_sim_state
from ogbench.manipspace import lie
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle


def pin_goal(env, goal_xyz):
    u = env.unwrapped
    mid = u._cube_target_mocap_ids[0]
    u._data.mocap_pos[mid] = goal_xyz
    u._data.mocap_quat[mid] = lie.SO3.identity().wxyz
    import mujoco

    mujoco.mj_forward(u._model, u._data)
    u.pre_step()
    u.post_step()


POLICY_BAR = (30, 90, 200)
ORACLE_BAR = (30, 140, 60)


def label_frame(frame: np.ndarray, text: str, role: str) -> np.ndarray:
    bar_color = POLICY_BAR if role == 'policy' else ORACLE_BAR
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 28], fill=bar_color)
    draw.text((8, 6), text, fill=(255, 255, 255))
    return np.asarray(img)


def render_env(env, goal_xyz) -> np.ndarray:
    pin_goal(env, goal_xyz)
    return env.render()


def infer_rollouts_path(annotated_path: str) -> str:
    p = Path(annotated_path)
    if p.stem.endswith('_annotated'):
        return str(p.with_name(p.stem[: -len('_annotated')] + p.suffix))
    raise ValueError(f'cannot infer rollout npz from {annotated_path}; pass --rollouts')


def mjstate_at_chunk_start(
    full_data,
    ann_data,
    idx: int,
    ep_start_row: int,
    ep_idx: int,
    policy_env,
    task_id: int,
):
    """State at replan time s_t (matches annotate_oracle_distance on full rollouts)."""
    from bon_sampling.sim_state import get_sim_state

    global_t = int(ann_data['chunk_boundary_indices'][idx])
    ep_start_global = int(ann_data['chunk_boundary_indices'][ep_start_row])
    if global_t == ep_start_global:
        if 'episode_initial_mjstate' in full_data:
            return np.asarray(full_data['episode_initial_mjstate'][ep_idx], dtype=np.float64)
        policy_env.reset(options=dict(task_id=task_id), seed=0)
        u = policy_env.unwrapped
        return get_sim_state(u._model, u._data)
    return np.asarray(full_data['next_mjstate'][global_t - 1], dtype=np.float64)


def restore_policy_env(env, mjstate, goal_xyz):
    u = env.unwrapped
    set_sim_state(u._model, u._data, mjstate)
    pin_goal(env, goal_xyz)


def replay_policy_chunk(env, mjstate, goal_xyz, actions, label: str) -> list[np.ndarray]:
    restore_policy_env(env, mjstate, goal_xyz)
    frames = [label_frame(render_env(env, goal_xyz), label, 'policy')]
    for i, action in enumerate(actions):
        env.step(np.clip(action, -1.0, 1.0))
        frames.append(
            label_frame(render_env(env, goal_xyz), f'{label} step {i + 1}/{len(actions)}', 'policy')
        )
    return frames


def replay_oracle(
    oracle_env,
    oracle,
    mjstate,
    goal_xyz,
    label: str,
    max_steps: int,
    warmup_steps: int,
) -> list[np.ndarray]:
    restore_sim_state(oracle_env, mjstate, warmup_steps=warmup_steps)
    pin_goal(oracle_env, goal_xyz)
    frames = [label_frame(oracle_env.render(), label, 'oracle')]

    if oracle_env.unwrapped._success:
        return frames

    seed = oracle_seed(mjstate)
    np.random.seed(seed)
    ob = oracle_env.unwrapped.compute_observation()
    info = oracle_env.unwrapped.get_reset_info()
    oracle.reset(ob, info)

    steps = 0
    resets = 0
    while steps < max_steps:
        if oracle_env.unwrapped._success:
            break
        if oracle.done:
            resets += 1
            np.random.seed(seed + resets)
            ob = oracle_env.unwrapped.compute_observation()
            info = oracle_env.unwrapped.get_reset_info()
            oracle.reset(ob, info)
            continue
        action = np.clip(np.asarray(oracle.select_action(ob, info)), -1.0, 1.0)
        ob, _, term, _, info = oracle_env.step(action)
        steps += 1
        frames.append(label_frame(oracle_env.render(), f'{label} step {steps}/{max_steps}', 'oracle'))
        if term and oracle_env.unwrapped._success:
            break
    return frames


def trajectory_slices(episode_ends, num_traj):
    starts = [0] + episode_ends[:-1].tolist()
    return list(zip(starts, episode_ends))[:num_traj]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='bon_sampling/data/flowbc_rollouts_100_annotated.npz')
    p.add_argument('--output_dir', default='bon_sampling/visualization/flowbc_rollouts_100')
    p.add_argument('--num_traj', type=int, default=10)
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--max_oracle_steps', type=int, default=200)
    p.add_argument('--rollouts', default=None,
                   help='Full rollout npz (default: infer from --input by dropping _annotated)')
    p.add_argument('--warmup_steps', type=int, default=2)
    args = p.parse_args()

    ann_data = dict(np.load(args.input, allow_pickle=False))
    for key in ('action_chunks', 'episode_ends', 'goal_xyz', 'task_id', 'chunk_boundary_indices'):
        if key not in ann_data:
            raise SystemExit(f'missing {key}; use annotated rollout npz')

    rollouts_path = args.rollouts or infer_rollouts_path(args.input)
    full_data = dict(np.load(rollouts_path, allow_pickle=False))
    if 'next_mjstate' not in full_data:
        raise SystemExit(f'missing next_mjstate in {rollouts_path}')

    os.makedirs(args.output_dir, exist_ok=True)
    task_id = int(ann_data['task_id'])
    goal_xyz = ann_data['goal_xyz']
    distances = ann_data['distance'] if 'distance' in ann_data else None
    print(f'rollouts={rollouts_path}')

    policy_env = gymnasium.make('cube-single-v0')
    policy_env.reset(options=dict(task_id=task_id))
    oracle_env = gymnasium.make('cube-single-v0', mode='data_collection', terminate_at_goal=True)
    oracle_env.reset()
    oracle = CubeMarkovOracle(env=oracle_env)

    trajs = trajectory_slices(ann_data['episode_ends'], args.num_traj)
    input_stem = Path(args.input).stem

    for ep_idx, (start, end) in enumerate(trajs):
        frames: list[np.ndarray] = []
        for k, idx in enumerate(range(start, end)):
            mjstate = mjstate_at_chunk_start(
                full_data, ann_data, idx, start, ep_idx, policy_env, task_id,
            )
            chunk = ann_data['action_chunks'][idx]
            mask = ann_data['chunk_masks'][idx] if 'chunk_masks' in ann_data else np.ones(len(chunk))
            actions = [chunk[i] for i in range(len(chunk)) if mask[i] > 0]
            dist_str = f' d={int(distances[idx])}' if distances is not None else ''
            frames.extend(
                replay_policy_chunk(
                    policy_env, mjstate, goal_xyz, actions,
                    f'Ep {ep_idx} chunk {k} | POLICY',
                )
            )
            frames.extend(
                replay_oracle(
                    oracle_env, oracle, mjstate, goal_xyz,
                    f'Ep {ep_idx} chunk {k} | ORACLE{dist_str}',
                    args.max_oracle_steps,
                    args.warmup_steps,
                )
            )

        out = os.path.join(args.output_dir, f'{input_stem}_verify_ep{ep_idx:02d}.mp4')
        with imageio.get_writer(out, fps=args.fps, codec='libx264', quality=8) as w:
            for f in frames:
                w.append_data(f)
        print(f'saved {out} ({len(frames)} frames, {end - start} chunks)')

    policy_env.close()
    oracle_env.close()


if __name__ == '__main__':
    main()
