"""Visualize oracle distance along BC trajectories.

MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  python bon_sampling/visualization/visualize_distance.py \
  --input bon_sampling/data/flowbc_rollouts_100_annotated.npz \
  --output_dir bon_sampling/visualization/flowbc_rollouts_100 \
  --num_traj 10 \
  --fps 20
"""

from __future__ import annotations

import argparse
import os
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault('MUJOCO_GL', 'egl')

import gymnasium
import imageio.v2 as imageio
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image

import ogbench.manipspace  
from bon_sampling.sim_state import set_sim_state
from ogbench.manipspace import lie


def pin_goal(env, goal_xyz):
    u = env.unwrapped
    mid = u._cube_target_mocap_ids[0]
    u._data.mocap_pos[mid] = goal_xyz
    u._data.mocap_quat[mid] = lie.SO3.identity().wxyz
    mujoco.mj_forward(u._model, u._data)
    u.pre_step()
    u.post_step()


def restore_scene(env, mjstate, goal_xyz):
    u = env.unwrapped
    set_sim_state(u._model, u._data, mjstate)
    pin_goal(env, goal_xyz)
    return env.render()


def trajectory_slices(episode_ends, num_traj):
    starts = [0] + episode_ends[:-1].tolist()
    return list(zip(starts, episode_ends))[:num_traj]


def render_distance_plot(distances, t, height, width):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    xs = np.arange(t + 1)
    ax.plot(xs, distances[: t + 1], color='steelblue', linewidth=2)
    ax.scatter([t], [distances[t]], color='crimson', s=40, zorder=3)
    ax.set_xlim(0, max(len(distances) - 1, 1))
    ax.set_ylim(0, max(distances.max() * 1.05, 1))
    ax.set_xlabel('timestep')
    ax.set_ylabel('oracle steps')
    ax.set_title(f't={t}, distance={distances[t]}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert('RGB'))


def combine(scene, plot):
    h = scene.shape[0]
    if plot.shape[0] != h:
        scale = h / plot.shape[0]
        new_w = int(plot.shape[1] * scale)
        plot = np.asarray(Image.fromarray(plot).resize((new_w, h)))
    return np.concatenate([scene, plot], axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='bon_sampling/bc_transitions_annotated.npz')
    p.add_argument('--output_dir', default='bon_sampling/distance_viz')
    p.add_argument('--num_traj', type=int, default=5)
    p.add_argument('--fps', type=int, default=20)
    args = p.parse_args()

    data = dict(np.load(args.input, allow_pickle=False))
    if 'next_mjstate' not in data:
        raise SystemExit('missing next_mjstate; re-run collect_bc_transitions.py')

    os.makedirs(args.output_dir, exist_ok=True)
    env = gymnasium.make('cube-single-v0')
    env.reset(options=dict(task_id=int(data['task_id'])))
    goal_xyz = data['goal_xyz']
    env.render()
    trajs = trajectory_slices(data['episode_ends'], args.num_traj)
    input_stem = Path(args.input).stem
    out = os.path.join(args.output_dir, f'{input_stem}_distances_{args.num_traj}.mp4')

    all_frames: list[np.ndarray] = []

    for i, (start, end) in enumerate(trajs):
        states = data['next_mjstate'][start:end]
        distances = data['distance'][start:end]
        for t in range(len(distances)):
            scene = restore_scene(env, states[t], goal_xyz)
            plot = render_distance_plot(distances, t, scene.shape[0], scene.shape[1])
            all_frames.append(combine(scene, plot))
        print(f'episode {i}: {len(distances)} frames')

    with imageio.get_writer(out, fps=args.fps, codec='libx264', quality=8) as w:
        for f in all_frames:
            w.append_data(f)
    print(f'saved {out} ({len(all_frames)} frames, {len(trajs)} episodes)')

    env.close()


if __name__ == '__main__':
    main()
