import mujoco
import numpy as np

from awr.sim_state import get_sim_state, set_sim_state


def prepare_oracle_env(env):
    u = env.unwrapped
    u._target_block = 0
    u._target_task = 'cube'
    u.pre_step()
    u.post_step()


def pin_goal(env, goal_xyz):
    """Pin cube-single target mocap to a fixed task goal."""
    from ogbench.manipspace import lie

    u = env.unwrapped
    mid = u._cube_target_mocap_ids[0]
    u._data.mocap_pos[mid] = np.asarray(goal_xyz, dtype=np.float64)
    u._data.mocap_quat[mid] = lie.SO3.identity().wxyz


def oracle_seed(mjstate):
    return int(np.abs(np.sum(mjstate[:32] * 1000)).astype(np.int64) % (2**31))


def progress_label(d_t, d_next, horizon: int, tau: int | None = None) -> float:
    """Binary progress: 1 if d(s) - d(s') >= H - tau.

    Default tau = H - 1 ⇒ threshold 1 (same as strict d(s') < d(s) when distances are ints).
    """
    h = int(horizon)
    if h < 1:
        raise ValueError(f'horizon must be >= 1, got {h}')
    if tau is None:
        tau = h - 1
    tau = int(tau)
    if not (0 <= tau < h):
        raise ValueError(f'tau must be in [0, H), got tau={tau} H={h}')
    threshold = h - tau
    return 1.0 if (int(d_t) - int(d_next)) >= threshold else 0.0


def delta_bin_label(delta: float, horizon: int, num_bins: int) -> float:
    """Signed uniform bins of Δ over [-H, H]; odd num_bins; end bins absorb tails.

    Splits [-H, H] into ``num_bins`` equal pieces of width s = 2H / num_bins:
      i = clip(floor((Δ + H) / s), 0, num_bins - 1)
      ℓ = i - (num_bins - 1) / 2

    So ℓ ∈ {-(w-1)/2, ..., +(w-1)/2} with w = num_bins (must be odd).
    """
    h = int(horizon)
    w = int(num_bins)
    if h < 1:
        raise ValueError(f'horizon must be >= 1, got {h}')
    if w < 1 or w % 2 == 0:
        raise ValueError(f'num_bins must be odd and >= 1, got {w}')
    s = (2.0 * h) / w
    i = int(np.floor((float(delta) + h) / s))
    i = int(np.clip(i, 0, w - 1))
    return float(i - (w - 1) / 2.0)


def capture_sim_state(env):
    u = env.unwrapped
    return get_sim_state(u._model, u._data)


def warmup_physics(env, n_steps=2):
    u = env.unwrapped
    for _ in range(n_steps):
        mujoco.mj_step(u._model, u._data)
    mujoco.mj_rnePostConstraint(u._model, u._data)
    u.pre_step()
    u.post_step()


def restore_sim_state(env, mjstate, warmup_steps=2, goal_xyz=None):
    u = env.unwrapped
    set_sim_state(u._model, u._data, mjstate)
    if goal_xyz is not None:
        pin_goal(env, goal_xyz)
    prepare_oracle_env(env)
    if warmup_steps > 0:
        warmup_physics(env, warmup_steps)


def restore_from_qpos_qvel(env, qpos, qvel, goal_xyz, warmup_steps=2):
    """Restore physics from OGBench qpos/qvel and pin the task goal."""
    u = env.unwrapped
    u.set_state(np.asarray(qpos, np.float64), np.asarray(qvel, np.float64))
    pin_goal(env, goal_xyz)
    prepare_oracle_env(env)
    if warmup_steps > 0:
        warmup_physics(env, warmup_steps)


def run_oracle_until_success(oracle_env, oracle, max_steps, seed=0):
    np.random.seed(seed)
    ob = oracle_env.unwrapped.compute_observation()
    info = oracle_env.unwrapped.get_reset_info()
    oracle.reset(ob, info)

    steps = 0
    resets = 0
    while steps < max_steps:
        if oracle_env.unwrapped._success:
            return steps, True
        if oracle.done:
            resets += 1
            np.random.seed(seed + resets)
            ob = oracle_env.unwrapped.compute_observation()
            info = oracle_env.unwrapped.get_reset_info()
            oracle.reset(ob, info)
            continue
        action = np.clip(np.asarray(oracle.select_action(ob, info)), -1.0, 1.0)
        ob, _, terminated, truncated, info = oracle_env.step(action)
        steps += 1
        if terminated and oracle_env.unwrapped._success:
            return steps, True
    return max_steps, bool(oracle_env.unwrapped._success)


def oracle_distance(oracle_env, oracle, mjstate, max_steps, warmup_steps=2, goal_xyz=None):
    restore_sim_state(oracle_env, mjstate, warmup_steps=warmup_steps, goal_xyz=goal_xyz)
    if oracle_env.unwrapped._success:
        return 0
    seed_src = mjstate if goal_xyz is None else np.concatenate(
        [np.asarray(mjstate[:32], np.float64), np.asarray(goal_xyz, np.float64).reshape(-1)]
    )
    steps, _ = run_oracle_until_success(
        oracle_env, oracle, max_steps, seed=oracle_seed(seed_src)
    )
    return steps


def oracle_distance_qpos(oracle_env, oracle, qpos, qvel, goal_xyz, max_steps, warmup_steps=2):
    """Oracle steps-to-goal from an OGBench (qpos, qvel) state."""
    restore_from_qpos_qvel(oracle_env, qpos, qvel, goal_xyz, warmup_steps=warmup_steps)
    if oracle_env.unwrapped._success:
        return 0
    seed_src = np.concatenate(
        [np.asarray(qpos, np.float64).reshape(-1)[:32], np.asarray(goal_xyz, np.float64).reshape(-1)]
    )
    steps, _ = run_oracle_until_success(
        oracle_env, oracle, max_steps, seed=oracle_seed(seed_src)
    )
    return steps
