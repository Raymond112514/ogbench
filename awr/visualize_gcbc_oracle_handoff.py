"""GCBC partial rollout, then oracle completion — save labeled mp4s.

For each sample:
  1. Roll out the GCBC policy until a random timestep t.
  2. Sync physics into a data_collection oracle env and finish with CubeMarkovOracle.
  3. Write an mp4 with a colored bar: blue = policy, green = oracle.

Usage (from ogbench/):
  python awr/visualize_gcbc_oracle_handoff.py \\
    --policy_ckpt flow_bc/checkpoints/cube_single_gcbc/best.pkl \\
    --num_samples 5 --task_id 1
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

POLICY_BAR = (30, 90, 200)
ORACLE_BAR = (30, 140, 60)


def label_frame(frame: np.ndarray, text: str, role: str) -> np.ndarray:
    bar_color = POLICY_BAR if role == 'policy' else ORACLE_BAR
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 28], fill=bar_color)
    draw.text((8, 6), text, fill=(255, 255, 255))
    return np.asarray(img)


def render_labeled(env, text: str, role: str, goal_xyz: np.ndarray) -> np.ndarray:
    from awr.oracle_utils import pin_goal

    pin_goal(env, goal_xyz)
    return label_frame(env.render(), text, role)


def run_policy_until(
    env,
    params,
    apply_fn,
    goal,
    goal_xyz,
    key,
    t: int,
    chunk_size: int,
    act_dim: int,
    n_flow_steps: int,
    goal_condition: bool,
    max_steps: int,
):
    """Roll GCBC for up to `t` env steps; return frames, steps, done, success, key."""
    import jax
    import jax.numpy as jnp

    from flow_bc.model import sample_action_chunk

    frames = [render_labeled(env, f'POLICY step 0/{t}', 'policy', goal_xyz)]
    ob = env.unwrapped.compute_observation()
    steps = 0
    done = False
    success = False

    while steps < t and steps < max_steps and not done:
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
            if steps >= t or steps >= max_steps:
                break
            ob, _, term, trunc, info = env.step(np.clip(action, -1.0, 1.0))
            steps += 1
            success = bool(info.get('success', False))
            frames.append(
                render_labeled(env, f'POLICY step {steps}/{t}', 'policy', goal_xyz)
            )
            if term or trunc:
                done = True
                break

    return frames, steps, done, success, key


def run_oracle_completion(
    oracle_env,
    oracle,
    mjstate: np.ndarray,
    goal_xyz: np.ndarray,
    policy_steps: int,
    max_oracle_steps: int,
    warmup_steps: int,
):
    """Restore s_t into the oracle env and finish with the scripted controller."""
    from awr.oracle_utils import oracle_seed, restore_sim_state

    restore_sim_state(
        oracle_env, mjstate, warmup_steps=warmup_steps, goal_xyz=goal_xyz
    )
    frames = [
        render_labeled(
            oracle_env, f'ORACLE handoff @ t={policy_steps}', 'oracle', goal_xyz
        )
    ]
    if oracle_env.unwrapped._success:
        return frames, True

    seed = oracle_seed(
        np.concatenate(
            [np.asarray(mjstate[:32], np.float64), np.asarray(goal_xyz, np.float64).reshape(-1)]
        )
    )
    np.random.seed(seed)
    ob = oracle_env.unwrapped.compute_observation()
    info = oracle_env.unwrapped.get_reset_info()
    oracle.reset(ob, info)

    steps = 0
    resets = 0
    while steps < max_oracle_steps:
        if oracle_env.unwrapped._success:
            return frames, True
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
        frames.append(
            render_labeled(
                oracle_env,
                f'ORACLE step {steps}/{max_oracle_steps} (after t={policy_steps})',
                'oracle',
                goal_xyz,
            )
        )
        if term and oracle_env.unwrapped._success:
            return frames, True
    return frames, bool(oracle_env.unwrapped._success)


def make_sample(
    env,
    oracle_env,
    oracle,
    params,
    apply_fn,
    meta,
    task_id: int,
    goal_xyz: np.ndarray,
    min_t: int,
    max_t: int,
    n_flow_steps: int,
    max_steps: int,
    max_oracle_steps: int,
    warmup_steps: int,
    rng: np.random.Generator,
):
    import jax

    from awr.oracle_utils import capture_sim_state

    t = int(rng.integers(min_t, max_t + 1))
    reset_seed = int(rng.integers(0, 2**31 - 1))
    policy_seed = int(rng.integers(0, 2**31 - 1))

    _, info = env.reset(options=dict(task_id=task_id), seed=reset_seed)
    goal = info['goal'] if meta['goal_condition'] else None
    key = jax.random.PRNGKey(policy_seed)

    policy_frames, steps, done, success, _ = run_policy_until(
        env,
        params,
        apply_fn,
        goal,
        goal_xyz,
        key,
        t,
        meta['chunk_size'],
        meta['act_dim'],
        n_flow_steps,
        meta['goal_condition'],
        max_steps,
    )

    if done:
        return policy_frames, steps, 0, success, t

    mjstate = capture_sim_state(env)
    oracle_frames, success = run_oracle_completion(
        oracle_env,
        oracle,
        mjstate,
        goal_xyz,
        policy_steps=steps,
        max_oracle_steps=max_oracle_steps,
        warmup_steps=warmup_steps,
    )
    return policy_frames + oracle_frames, steps, len(oracle_frames) - 1, success, t


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
    p.add_argument('--num_samples', type=int, default=5)
    p.add_argument('--min_t', type=int, default=20, help='Inclusive lower bound for random handoff timestep')
    p.add_argument('--max_t', type=int, default=100, help='Inclusive upper bound for random handoff timestep')
    p.add_argument('--n_flow_steps', type=int, default=10)
    p.add_argument('--max_steps', type=int, default=None)
    p.add_argument('--max_oracle_steps', type=int, default=200)
    p.add_argument('--warmup_steps', type=int, default=2)
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output_dir', default='awr/results/gcbc_oracle_handoff')
    p.add_argument('--egl_device', type=int, default=None)
    args = p.parse_args()

    if args.egl_device is not None:
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.egl_device)

    import gymnasium

    import ogbench.manipspace  # noqa: F401
    from flow_bc.checkpoint import load_flow_bc
    from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle

    policy, params, meta = load_flow_bc(args.policy_ckpt)
    env_name = meta.get('eval_env_name', args.env_name)

    env = gymnasium.make(env_name)
    oracle_env = gymnasium.make('cube-single-v0', mode='data_collection', terminate_at_goal=True)
    oracle_env.reset()
    oracle = CubeMarkovOracle(env=oracle_env)

    max_steps = args.max_steps or env.spec.max_episode_steps
    goal_xyz = env.unwrapped.task_infos[args.task_id - 1]['goal_xyzs'][0].copy()
    task_name = env.unwrapped.task_infos[args.task_id - 1]['task_name']

    min_t = max(1, args.min_t)
    max_t = min(args.max_t, max_steps - 1)
    if min_t > max_t:
        raise SystemExit(f'invalid handoff range: min_t={min_t} > max_t={max_t}')

    out_dir = Path(args.output_dir)
    rng = np.random.default_rng(args.seed)

    print(
        f'env={env_name} task_id={args.task_id} ({task_name}) '
        f'chunk_size={meta["chunk_size"]} num_samples={args.num_samples} '
        f'handoff_t=[{min_t},{max_t}] -> {out_dir}'
    )

    for i in range(args.num_samples):
        frames, policy_steps, oracle_steps, success, t = make_sample(
            env,
            oracle_env,
            oracle,
            params,
            policy.apply,
            meta,
            args.task_id,
            goal_xyz,
            min_t,
            max_t,
            args.n_flow_steps,
            max_steps,
            args.max_oracle_steps,
            args.warmup_steps,
            rng,
        )
        out_path = out_dir / f'task{args.task_id}_sample{i:02d}_t{t}.mp4'
        save_mp4(out_path, frames, args.fps)
        print(
            f'sample {i}: t={t} policy_steps={policy_steps} '
            f'oracle_steps={oracle_steps} success={success} '
            f'frames={len(frames)} -> {out_path}'
        )

    env.close()
    oracle_env.close()


if __name__ == '__main__':
    main()
