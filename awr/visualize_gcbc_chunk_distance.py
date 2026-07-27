"""Collect GCBC rollouts, oracle-label chunk-boundary states, viz pairwise progress.

For each episode:
  * Roll out GCBC, snapshot mjstate at every chunk boundary (t = 0, H, 2H, ...).
  * Oracle-distance each boundary state.
  * Emit frames that show two consecutive boundary states side-by-side, with a
    thick border on the lower-distance (better) state.

All episodes are written into a single mp4.

Usage (from ogbench/):
  python awr/visualize_gcbc_chunk_distance.py \\
    --policy_ckpt flow_bc/checkpoints/cube_single_gcbc/best.pkl \\
    --num_episodes 5 --task_id 1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

BETTER_BORDER = (40, 200, 80)
WORSE_BORDER = (60, 60, 60)
GAP_COLOR = (20, 20, 20)
TITLE_BG = (25, 25, 25)


def capture_state(env) -> np.ndarray:
    from awr.oracle_utils import capture_sim_state

    return capture_sim_state(env)


def render_state(env, mjstate: np.ndarray, goal_xyz: np.ndarray) -> np.ndarray:
    from awr.oracle_utils import pin_goal
    from awr.sim_state import set_sim_state

    u = env.unwrapped
    set_sim_state(u._model, u._data, mjstate)
    pin_goal(env, goal_xyz)
    return np.asarray(env.render())


def draw_border(frame: np.ndarray, color: tuple[int, int, int], width: int = 6) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    for i in range(width):
        draw.rectangle([i, i, img.width - 1 - i, img.height - 1 - i], outline=color)
    return np.asarray(img)


def annotate_panel(frame: np.ndarray, title: str, distance: int, better: bool) -> np.ndarray:
    border = BETTER_BORDER if better else WORSE_BORDER
    framed = draw_border(frame, border, width=6 if better else 3)
    img = Image.fromarray(framed)
    draw = ImageDraw.Draw(img)
    label = f'{title}  d={distance}'
    if better:
        label += '  BEST'
    draw.rectangle([0, 0, img.width, 24], fill=TITLE_BG)
    draw.text((8, 5), label, fill=BETTER_BORDER if better else (220, 220, 220))
    return np.asarray(img)


def side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    left_d: int,
    right_d: int,
    left_title: str,
    right_title: str,
    ep_header: str,
    gap: int = 8,
) -> np.ndarray:
    left_better = left_d <= right_d
    right_better = right_d < left_d  # strict: ties highlight left only
    left_p = annotate_panel(left, left_title, left_d, left_better)
    right_p = annotate_panel(right, right_title, right_d, right_better)

    h = max(left_p.shape[0], right_p.shape[0])
    spacer = np.full((h, gap, 3), GAP_COLOR, dtype=np.uint8)
    body = np.concatenate([left_p, spacer, right_p], axis=1)

    header_h = 28
    canvas = np.full((header_h + body.shape[0], body.shape[1], 3), TITLE_BG, dtype=np.uint8)
    canvas[header_h:] = body
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    draw.text((8, 6), ep_header, fill=(255, 255, 255))
    return np.asarray(img)


def collect_chunk_boundary_states(
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
) -> tuple[list[np.ndarray], list[int], bool]:
    """Roll GCBC; return (mjstates at t=0,H,..., success)."""
    import jax
    import jax.numpy as jnp

    from flow_bc.model import sample_action_chunk

    ob, info = env.reset(options=dict(task_id=task_id))
    goal = info['goal'] if goal_condition else None
    states = [capture_state(env)]
    steps_at_boundary = [0]
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
        for action in chunk:
            if steps >= max_steps:
                break
            ob, _, term, trunc, info = env.step(np.clip(action, -1.0, 1.0))
            steps += 1
            success = bool(info.get('success', False))
            if term or trunc:
                if steps != steps_at_boundary[-1]:
                    states.append(capture_state(env))
                    steps_at_boundary.append(steps)
                return states, steps_at_boundary, success
        states.append(capture_state(env))
        steps_at_boundary.append(steps)

    return states, steps_at_boundary, success


def oracle_label_states(
    oracle_env,
    oracle,
    states: list[np.ndarray],
    goal_xyz: np.ndarray,
    max_oracle_steps: int,
    warmup_steps: int,
) -> np.ndarray:
    from awr.oracle_utils import oracle_distance

    distances = np.empty(len(states), dtype=np.int32)
    for i, mjstate in enumerate(states):
        distances[i] = oracle_distance(
            oracle_env,
            oracle,
            mjstate,
            max_oracle_steps,
            warmup_steps=warmup_steps,
            goal_xyz=goal_xyz,
        )
    return distances


def episode_comparison_frames(
    render_env,
    states: list[np.ndarray],
    distances: np.ndarray,
    timesteps: list[int],
    goal_xyz: np.ndarray,
    ep_idx: int,
    success: bool,
) -> list[np.ndarray]:
    frames = []
    n = len(states)
    for i in range(n - 1):
        left = render_state(render_env, states[i], goal_xyz)
        right = render_state(render_env, states[i + 1], goal_xyz)
        improved = distances[i + 1] < distances[i]
        header = (
            f'ep {ep_idx}  pair {i}/{n - 2}  '
            f't={timesteps[i]}→{timesteps[i + 1]}  '
            f'd={distances[i]}→{distances[i + 1]}  '
            f'{"IMPROVED" if improved else "WORSE/TIE"}  '
            f'success={success}'
        )
        frames.append(
            side_by_side(
                left,
                right,
                int(distances[i]),
                int(distances[i + 1]),
                f's_t  t={timesteps[i]}',
                f"s_t' t={timesteps[i + 1]}",
                header,
            )
        )
    return frames


def save_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec='libx264', quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--policy_ckpt', default='flow_bc/checkpoints/cube_single_gcbc/best.pkl')
    p.add_argument('--env_name', default='cube-single-v0')
    p.add_argument('--task_id', type=int, default=1)
    p.add_argument('--num_episodes', type=int, default=5)
    p.add_argument('--n_flow_steps', type=int, default=10)
    p.add_argument('--max_steps', type=int, default=None)
    p.add_argument('--max_oracle_steps', type=int, default=200)
    p.add_argument('--warmup_steps', type=int, default=2)
    p.add_argument('--fps', type=int, default=2, help='Slow fps so each pair is readable')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', default='awr/results/gcbc_chunk_distance/compare.mp4')
    p.add_argument('--egl_device', type=int, default=None)
    args = p.parse_args()

    if args.egl_device is not None:
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.egl_device)

    import gymnasium
    import jax

    import ogbench.manipspace  # noqa: F401
    from flow_bc.checkpoint import load_flow_bc
    from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle

    policy, params, meta = load_flow_bc(args.policy_ckpt)
    env_name = meta.get('eval_env_name', args.env_name)
    chunk_size = int(meta['chunk_size'])
    act_dim = meta['act_dim']
    goal_condition = meta['goal_condition']

    env = gymnasium.make(env_name)
    render_env = gymnasium.make(env_name)
    oracle_env = gymnasium.make('cube-single-v0', mode='data_collection', terminate_at_goal=True)
    oracle_env.reset()
    oracle = CubeMarkovOracle(env=oracle_env)

    max_steps = args.max_steps or env.spec.max_episode_steps
    goal_xyz = env.unwrapped.task_infos[args.task_id - 1]['goal_xyzs'][0].copy()
    task_name = env.unwrapped.task_infos[args.task_id - 1]['task_name']
    render_env.reset(options=dict(task_id=args.task_id))

    key = jax.random.PRNGKey(args.seed)
    all_frames: list[np.ndarray] = []

    print(
        f'env={env_name} task_id={args.task_id} ({task_name}) '
        f'episodes={args.num_episodes} chunk_size={chunk_size} -> {args.output}'
    )

    for ep in range(args.num_episodes):
        key, ep_key = jax.random.split(key)
        states, timesteps, success = collect_chunk_boundary_states(
            env,
            params,
            policy.apply,
            args.task_id,
            ep_key,
            chunk_size,
            act_dim,
            args.n_flow_steps,
            max_steps,
            goal_condition,
        )
        distances = oracle_label_states(
            oracle_env,
            oracle,
            states,
            goal_xyz,
            args.max_oracle_steps,
            args.warmup_steps,
        )
        frames = episode_comparison_frames(
            render_env, states, distances, timesteps, goal_xyz, ep, success
        )
        all_frames.extend(frames)
        print(
            f'ep {ep}: steps={timesteps[-1]} boundaries={len(states)} '
            f'pairs={len(frames)} success={success} '
            f'd: {distances.tolist()}'
        )

    out_path = Path(args.output)
    save_mp4(out_path, all_frames, args.fps)
    print(f'saved {out_path} ({len(all_frames)} frames, {args.num_episodes} episodes)')

    env.close()
    render_env.close()
    oracle_env.close()


if __name__ == '__main__':
    main()
