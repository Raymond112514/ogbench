import mujoco
import numpy as np

from bon_sampling.sim_state import get_sim_state, set_sim_state


def prepare_oracle_env(env):
    u = env.unwrapped
    u._target_block = 0
    u._target_task = 'cube'
    u.pre_step()
    u.post_step()


def oracle_seed(mjstate):
    return int(np.abs(np.sum(mjstate[:32] * 1000)).astype(np.int64) % (2**31))


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


def restore_sim_state(env, mjstate, warmup_steps=2):
    u = env.unwrapped
    set_sim_state(u._model, u._data, mjstate)
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


def oracle_distance(oracle_env, oracle, mjstate, max_steps, warmup_steps=2):
    restore_sim_state(oracle_env, mjstate, warmup_steps=warmup_steps)
    if oracle_env.unwrapped._success:
        return 0
    steps, _ = run_oracle_until_success(
        oracle_env, oracle, max_steps, seed=oracle_seed(mjstate)
    )
    return steps
