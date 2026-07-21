"""Parallel rollout collection with flow-BC or an AWR actor."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')


def _random_key():
    import jax

    return jax.random.PRNGKey(int.from_bytes(os.urandom(4), 'big') & 0x7FFFFFFF)


def collect_rollout_flow(env, params, apply_fn, task_id, chunk_size, act_dim, n_flow_steps, max_steps, goal_condition):
    import jax
    import jax.numpy as jnp

    from awr.sim_state import get_sim_state
    from flow_bc.model import sample_action_chunk

    u = env.unwrapped
    obs_list, act_list, next_obs_list, state_list, success_list = [], [], [], [], []
    ob, info = env.reset(options=dict(task_id=task_id))
    initial_mjstate = get_sim_state(u._model, u._data)
    goal = info['goal'] if goal_condition else None
    key = _random_key()
    steps, success = 0, False

    while steps < max_steps:
        key, sample_key = jax.random.split(key)
        chunk = np.asarray(
            sample_action_chunk(
                params, apply_fn, jnp.asarray(ob),
                jnp.asarray(goal) if goal_condition else None, sample_key,
                chunk_size=chunk_size, act_dim=act_dim, n_flow_steps=n_flow_steps,
                goal_condition=goal_condition,
            )
        )
        for k in range(chunk_size):
            if steps >= max_steps:
                break
            next_ob, _, term, trunc, info = env.step(chunk[k])
            step_success = bool(info.get('success', False))
            obs_list.append(ob.copy())
            act_list.append(chunk[k].copy())
            next_obs_list.append(next_ob.copy())
            state_list.append(get_sim_state(u._model, u._data))
            success_list.append(step_success)
            ob = next_ob
            steps += 1
            if term or trunc:
                return obs_list, act_list, next_obs_list, state_list, success_list, step_success, initial_mjstate
    return obs_list, act_list, next_obs_list, state_list, success_list, success, initial_mjstate


def collect_rollout_actor(env, actor_ckpt, task_id, max_steps):
    import jax

    from awr.policy import sample_action_chunk
    from awr.sim_state import get_sim_state

    u = env.unwrapped
    chunk_size = actor_ckpt['chunk_size']
    obs_list, act_list, next_obs_list, state_list, success_list = [], [], [], [], []
    ob, info = env.reset(options=dict(task_id=task_id))
    initial_mjstate = get_sim_state(u._model, u._data)
    key = _random_key()
    steps, success = 0, False

    while steps < max_steps:
        key, sample_key = jax.random.split(key)
        chunk = sample_action_chunk(actor_ckpt, ob, sample_key)
        for k in range(chunk_size):
            if steps >= max_steps:
                break
            next_ob, _, term, trunc, info = env.step(chunk[k])
            step_success = bool(info.get('success', False))
            obs_list.append(ob.copy())
            act_list.append(chunk[k].copy())
            next_obs_list.append(next_ob.copy())
            state_list.append(get_sim_state(u._model, u._data))
            success_list.append(step_success)
            ob = next_ob
            steps += 1
            if term or trunc:
                return obs_list, act_list, next_obs_list, state_list, success_list, step_success, initial_mjstate
    return obs_list, act_list, next_obs_list, state_list, success_list, success, initial_mjstate


def _run_batch(worker_id, episode_indices, policy_ckpt, actor_ckpt, env_name, task_id, max_steps, n_flow_steps):
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import gymnasium

    import ogbench.manipspace  # noqa: F401

    env = gymnasium.make(env_name)
    results = []
    if actor_ckpt is None:
        from flow_bc.checkpoint import load_flow_bc

        model, params, meta = load_flow_bc(policy_ckpt)
        chunk_size, act_dim = meta['chunk_size'], meta['act_dim']
        goal_condition = meta['goal_condition']
        for ep in episode_indices:
            out = collect_rollout_flow(
                env, params, model.apply, task_id, chunk_size, act_dim, n_flow_steps, max_steps, goal_condition
            )
            obs, acts, next_obs, states, step_succ, success, init_s = out
            results.append(dict(
                ep=int(ep), observations=obs, actions=acts, next_observations=next_obs,
                next_mjstate=states, step_successes=step_succ, success=success, initial_mjstate=init_s,
            ))
            print(f'worker {worker_id} ep {ep}: T={len(acts)} success={success}', flush=True)
    else:
        from awr.policy import load_actor

        actor = load_actor(actor_ckpt) if isinstance(actor_ckpt, str) else actor_ckpt
        for ep in episode_indices:
            out = collect_rollout_actor(env, actor, task_id, max_steps)
            obs, acts, next_obs, states, step_succ, success, init_s = out
            results.append(dict(
                ep=int(ep), observations=obs, actions=acts, next_observations=next_obs,
                next_mjstate=states, step_successes=step_succ, success=success, initial_mjstate=init_s,
            ))
            print(f'worker {worker_id} ep {ep}: T={len(acts)} success={success}', flush=True)
    env.close()
    return results


def parallel_collect(
    policy_ckpt: str,
    env_name: str,
    task_id: int,
    num_episodes: int,
    num_workers: int,
    n_flow_steps: int = 10,
    actor_ckpt: str | dict | None = None,
    max_steps: int | None = None,
):
    import gymnasium

    import ogbench.manipspace  # noqa: F401
    from flow_bc.checkpoint import read_ckpt_meta

    meta = read_ckpt_meta(policy_ckpt)
    tmp = gymnasium.make(env_name)
    max_steps = max_steps or tmp.spec.max_episode_steps
    goal_xyz = tmp.unwrapped.task_infos[task_id - 1]['goal_xyzs'][0].copy()
    tmp.close()

    # Persist actor to a temp path for spawn workers if dict given
    actor_path = None
    if isinstance(actor_ckpt, dict):
        import tempfile

        from awr.policy import save_actor

        fd, actor_path = tempfile.mkstemp(suffix='.pkl')
        os.close(fd)
        save_actor(actor_path, actor_ckpt)
        actor_arg = actor_path
    else:
        actor_arg = actor_ckpt

    splits = np.array_split(np.arange(num_episodes), num_workers)
    merged = []
    ctx = mp.get_context('spawn')
    try:
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
            futs = [
                pool.submit(
                    _run_batch, wid, split, str(Path(policy_ckpt).resolve()), actor_arg,
                    env_name, task_id, max_steps, n_flow_steps,
                )
                for wid, split in enumerate(splits) if len(split) > 0
            ]
            for fut in futs:
                merged.extend(fut.result())
    finally:
        if actor_path is not None and os.path.exists(actor_path):
            os.remove(actor_path)

    merged.sort(key=lambda r: r['ep'])
    all_obs, all_act, all_next, all_state, all_succ = [], [], [], [], []
    ends, ep_succ, inits = [], [], []
    for row in merged:
        all_obs.extend(row['observations'])
        all_act.extend(row['actions'])
        all_next.extend(row['next_observations'])
        all_state.extend(row['next_mjstate'])
        all_succ.extend(row['step_successes'])
        ends.append(len(all_act))
        ep_succ.append(row['success'])
        inits.append(row['initial_mjstate'])

    return {
        'observations': np.asarray(all_obs, np.float32),
        'actions': np.asarray(all_act, np.float32),
        'next_observations': np.asarray(all_next, np.float32),
        'next_mjstate': np.asarray(all_state, np.float64),
        'successes': np.asarray(all_succ, np.bool_),
        'episode_initial_mjstate': np.asarray(inits, np.float64),
        'goal_xyz': goal_xyz,
        'task_id': np.array(task_id),
        'episode_ends': np.asarray(ends, np.int32),
        'chunk_size': np.array(meta['chunk_size']),
        'success_rate': float(np.mean(ep_succ)) if ep_succ else 0.0,
    }


def save_rollouts(path: str | Path, data: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in data.items() if k != 'success_rate'})
    return str(path)
