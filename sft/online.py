"""SFT online loop: GCBC → collect → oracle improve labels → binary-advantage AWR.

Round k:
  1. Collect `episodes_per_round` episodes with pi_k (pi_0 = flow-BC GCBC).
  2. Oracle-label each action chunk: y=1 if d(s_{t+H}) < d(s_t), else 0.
  3. Fit a new Gaussian policy with AWR using advantage = y  →  pi_{k+1}.

Labeled data accumulates across rounds by default.

Usage (from ogbench/):
  python sft/online.py \\
    --policy_ckpt flow_bc/checkpoints/cube_single_gcbc/best.pkl \\
    --task_id 1 --rounds 50 --episodes_per_round 100
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def _capture(env) -> np.ndarray:
    from awr.oracle_utils import capture_sim_state

    return capture_sim_state(env)


def collect_episodes_flow(
    env,
    params,
    apply_fn,
    meta: dict,
    task_id: int,
    num_episodes: int,
    n_flow_steps: int,
    max_steps: int,
    seed: int,
) -> tuple[dict, float]:
    import jax
    import jax.numpy as jnp

    from flow_bc.model import sample_action_chunk

    chunk_size = meta['chunk_size']
    act_dim = meta['act_dim']
    goal_condition = meta['goal_condition']
    key = jax.random.PRNGKey(seed)

    observations, actions, next_observations, successes = [], [], [], []
    next_mjstate, episode_initial_mjstate, episode_ends = [], [], []

    for _ in range(num_episodes):
        ob, info = env.reset(options=dict(task_id=task_id))
        goal = info['goal'] if goal_condition else None
        episode_initial_mjstate.append(_capture(env))
        steps = 0
        done = False
        while steps < max_steps and not done:
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
            for a in chunk:
                if steps >= max_steps or done:
                    break
                prev = np.asarray(ob, np.float32)
                ob, _, term, trunc, info = env.step(np.clip(a, -1.0, 1.0))
                done = bool(term or trunc)
                observations.append(prev)
                actions.append(np.asarray(a, np.float32))
                next_observations.append(np.asarray(ob, np.float32))
                next_mjstate.append(_capture(env))
                successes.append(bool(info.get('success', False)))
                steps += 1
        episode_ends.append(len(actions))

    return _pack_rollout(
        observations, actions, next_observations, successes,
        next_mjstate, episode_initial_mjstate, episode_ends, chunk_size,
    )


def collect_episodes_gaussian(
    env,
    actor_ckpt: dict,
    task_id: int,
    num_episodes: int,
    max_steps: int,
    seed: int,
    temperature: float = 1.0,
) -> tuple[dict, float]:
    import jax

    from awr.policy import sample_action_chunk

    chunk_size = int(actor_ckpt['chunk_size'])
    key = jax.random.PRNGKey(seed)

    observations, actions, next_observations, successes = [], [], [], []
    next_mjstate, episode_initial_mjstate, episode_ends = [], [], []

    for _ in range(num_episodes):
        ob, _ = env.reset(options=dict(task_id=task_id))
        episode_initial_mjstate.append(_capture(env))
        steps = 0
        done = False
        while steps < max_steps and not done:
            key, sample_key = jax.random.split(key)
            chunk = sample_action_chunk(actor_ckpt, ob, sample_key, temperature=temperature)
            for a in chunk:
                if steps >= max_steps or done:
                    break
                prev = np.asarray(ob, np.float32)
                ob, _, term, trunc, info = env.step(np.clip(a, -1.0, 1.0))
                done = bool(term or trunc)
                observations.append(prev)
                actions.append(np.asarray(a, np.float32))
                next_observations.append(np.asarray(ob, np.float32))
                next_mjstate.append(_capture(env))
                successes.append(bool(info.get('success', False)))
                steps += 1
        episode_ends.append(len(actions))

    return _pack_rollout(
        observations, actions, next_observations, successes,
        next_mjstate, episode_initial_mjstate, episode_ends, chunk_size,
    )


def _pack_rollout(observations, actions, next_observations, successes,
                  next_mjstate, episode_initial_mjstate, episode_ends, chunk_size):
    ends = np.asarray(episode_ends, np.int32)
    succ = np.asarray(successes, np.bool_)
    success_rate = float(succ[ends - 1].mean()) if len(ends) else 0.0
    rollout = dict(
        observations=np.asarray(observations, np.float32),
        actions=np.asarray(actions, np.float32),
        next_observations=np.asarray(next_observations, np.float32),
        successes=succ,
        next_mjstate=np.asarray(next_mjstate, np.float64),
        episode_initial_mjstate=np.asarray(episode_initial_mjstate, np.float64),
        episode_ends=ends,
        chunk_size=np.array(chunk_size),
    )
    return rollout, success_rate


def evaluate_policy(
    env,
    policy,  # ('flow', params, apply_fn, meta) or ('gaussian', actor_ckpt)
    task_id: int,
    num_episodes: int,
    max_steps: int,
    seed: int,
    n_flow_steps: int = 10,
) -> float:
    if policy[0] == 'flow':
        _, params, apply_fn, meta = policy
        _, sr = collect_episodes_flow(
            env, params, apply_fn, meta, task_id, num_episodes, n_flow_steps, max_steps, seed,
        )
        return sr
    _, actor_ckpt = policy
    _, sr = collect_episodes_gaussian(
        env, actor_ckpt, task_id, num_episodes, max_steps, seed, temperature=1.0,
    )
    return sr


# ---------------------------------------------------------------------------
# Oracle labeling
# ---------------------------------------------------------------------------


def _episode_ranges(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    starts = [0] + episode_ends[:-1].tolist()
    return list(zip(starts, episode_ends.tolist()))


def _chunk_boundary_indices(episode_ends: np.ndarray, chunk_size: int) -> np.ndarray:
    indices = []
    for start, end in _episode_ranges(episode_ends):
        t = start
        while t < end:
            indices.append(t)
            t += chunk_size
    return np.asarray(indices, np.int64)


def _mjstate_at(rollout: dict, t: int, ep_start: int, ep_idx: int) -> np.ndarray:
    if t == ep_start:
        return np.asarray(rollout['episode_initial_mjstate'][ep_idx], np.float64)
    return np.asarray(rollout['next_mjstate'][t - 1], np.float64)


def _annotate_worker(worker_id, input_path, indices, goal_xyz, max_oracle_steps, warmup_steps):
    import gymnasium

    import ogbench.manipspace  # noqa: F401
    from awr.oracle_utils import oracle_distance
    from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle

    data = dict(np.load(input_path, allow_pickle=False))
    episode_ends = data['episode_ends']
    starts = [0] + episode_ends[:-1].tolist()
    # map global t -> (ep_start, ep_idx)
    t_to_ep = {}
    for ep_idx, (start, end) in enumerate(zip(starts, episode_ends.tolist())):
        for t in range(start, end):
            t_to_ep[t] = (start, ep_idx)

    oracle_env = gymnasium.make('cube-single-v0', mode='data_collection', terminate_at_goal=True)
    oracle_env.reset()
    oracle = CubeMarkovOracle(env=oracle_env)
    goal = np.asarray(goal_xyz, np.float64)
    distances = np.zeros(len(indices), np.int32)

    for j, t in enumerate(indices):
        t = int(t)
        ep_start, ep_idx = t_to_ep[t]
        mjstate = _mjstate_at(data, t, ep_start, ep_idx)
        distances[j] = oracle_distance(
            oracle_env, oracle, mjstate, max_oracle_steps,
            warmup_steps=warmup_steps, goal_xyz=goal,
        )
        if (j + 1) % 50 == 0:
            print(f'  oracle worker {worker_id}: {j + 1}/{len(indices)}', flush=True)

    oracle_env.close()
    return indices, distances


def label_rollout(
    rollout: dict,
    goal_xyz: np.ndarray,
    chunk_size: int,
    num_workers: int,
    max_oracle_steps: int,
    warmup_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return chunk-level (obs, flat_actions, improve_label)."""
    from awr.classifier.ogbench_dataset import build_chunk_progress

    n = len(rollout['observations'])
    terminals = np.zeros(n, np.float32)
    terminals[np.asarray(rollout['episode_ends'], np.int64) - 1] = 1.0
    indices = _chunk_boundary_indices(rollout['episode_ends'], chunk_size)

    fd, tmp = tempfile.mkstemp(suffix='.npz')
    os.close(fd)
    try:
        np.savez_compressed(
            tmp,
            next_mjstate=rollout['next_mjstate'],
            episode_initial_mjstate=rollout['episode_initial_mjstate'],
            episode_ends=rollout['episode_ends'],
        )
        dist_map = {}
        splits = np.array_split(indices, num_workers)
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futs = [
                pool.submit(
                    _annotate_worker, wid, tmp, split, goal_xyz, max_oracle_steps, warmup_steps,
                )
                for wid, split in enumerate(splits)
                if len(split) > 0
            ]
            for fut in as_completed(futs):
                idx_out, dist = fut.result()
                dist_map.update(zip(idx_out.tolist(), dist.tolist()))
        distance = np.full(n, -1, np.int32)
        for t, d in dist_map.items():
            distance[int(t)] = int(d)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    return build_chunk_progress(
        rollout['observations'], rollout['actions'], terminals, distance, chunk_size,
    )


# ---------------------------------------------------------------------------
# AWR with binary advantage
# ---------------------------------------------------------------------------


def train_awr_binary(
    observations: np.ndarray,
    actions: np.ndarray,
    labels: np.ndarray,
    *,
    train_steps: int,
    batch_size: int,
    alpha: float,
    lr: float,
    seed: int,
    chunk_size: int,
    hidden_dims=(512, 512, 512, 512),
) -> dict:
    """Fit a fresh Gaussian actor with AWR; advantage = binary improve label."""
    import jax
    import jax.numpy as jnp
    from tqdm import trange

    from awr.policy import _awr_loss, create_actor, weight_stats

    model, state = create_actor(
        observations.shape[1], actions.shape[1],
        hidden_dims=hidden_dims, lr=lr, seed=seed,
    )
    n = len(observations)
    rng = np.random.default_rng(seed)
    info = {}

    @jax.jit
    def train_step(state, obs, acts, adv):
        def loss_fn(params):
            dist = state.apply_fn(params, obs)
            return _awr_loss(dist, acts, adv, alpha)

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), metrics

    for _ in trange(train_steps, desc='awr', leave=False):
        idx = rng.integers(0, n, size=min(batch_size, n))
        state, info = train_step(
            state,
            jnp.asarray(observations[idx]),
            jnp.asarray(actions[idx]),
            jnp.asarray(labels[idx]),
        )

    metrics = {k: float(v) for k, v in info.items()}
    metrics.update(weight_stats(labels, alpha))
    metrics['label_mean'] = float(labels.mean())
    return {
        'mode': 'awr_actor',
        'params': state.params,
        'apply_fn': model.apply,
        'obs_dim': int(observations.shape[1]),
        'act_dim': int(actions.shape[1]),
        'chunk_size': int(chunk_size),
        'hidden_dims': list(hidden_dims),
        'metrics': metrics,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--policy_ckpt', default='flow_bc/checkpoints/cube_single_gcbc/best.pkl',
                   help='pi_0 GCBC (flow-BC) checkpoint')
    p.add_argument('--env_name', default='cube-single-v0')
    p.add_argument('--task_id', type=int, default=1)
    p.add_argument('--rounds', type=int, default=50)
    p.add_argument('--episodes_per_round', type=int, default=100)
    p.add_argument('--eval_episodes', type=int, default=50)
    p.add_argument('--n_flow_steps', type=int, default=10)
    p.add_argument('--max_steps', type=int, default=None)
    p.add_argument('--train_steps', type=int, default=5000, help='AWR gradient steps per round')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--alpha', type=float, default=10.0, help='AWR temperature on binary labels')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--num_workers', type=int, default=10, help='Parallel oracle-label workers')
    p.add_argument('--max_oracle_steps', type=int, default=200)
    p.add_argument('--warmup_steps', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no_accumulate', action='store_true',
                   help='Train each round only on that round\'s data (default: accumulate)')
    p.add_argument('--output_dir', default='sft/checkpoints')
    p.add_argument('--device', choices=['cpu', 'auto'], default='cpu')
    p.add_argument('--wandb_project', default='sft-online')
    p.add_argument('--wandb_name', default=None)
    p.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
    args = p.parse_args()

    if args.device == 'cpu':
        os.environ['JAX_PLATFORMS'] = 'cpu'

    import gymnasium
    import wandb

    import ogbench.manipspace  # noqa: F401
    from awr.policy import save_actor
    from flow_bc.checkpoint import load_flow_bc

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config=vars(args),
    )

    flow_model, flow_params, meta = load_flow_bc(args.policy_ckpt)
    env_name = meta.get('eval_env_name', args.env_name)
    chunk_size = int(meta['chunk_size'])

    env = gymnasium.make(env_name)
    max_steps = args.max_steps or env.spec.max_episode_steps
    goal_xyz = env.unwrapped.task_infos[args.task_id - 1]['goal_xyzs'][0].copy()
    task_name = env.unwrapped.task_infos[args.task_id - 1]['task_name']
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f'SFT online | env={env_name} task={args.task_id} ({task_name}) '
        f'chunk={chunk_size} rounds={args.rounds} eps/round={args.episodes_per_round}'
    )

    # pi_0 = GCBC
    policy = ('flow', flow_params, flow_model.apply, meta)
    buf_obs, buf_act, buf_lab = [], [], []

    # Eval pi_0
    sr0 = evaluate_policy(
        env, policy, args.task_id, args.eval_episodes, max_steps, args.seed, args.n_flow_steps,
    )
    print(f'round 0 (GCBC) eval success={sr0:.3f}')
    wandb.log({'round': 0, 'eval/success_rate': sr0, 'policy': 'gcbc'}, step=0)

    for k in range(1, args.rounds + 1):
        # 1. Collect
        if policy[0] == 'flow':
            _, params, apply_fn, meta_k = policy
            rollout, collect_sr = collect_episodes_flow(
                env, params, apply_fn, meta_k, args.task_id,
                args.episodes_per_round, args.n_flow_steps, max_steps, args.seed + k,
            )
        else:
            rollout, collect_sr = collect_episodes_gaussian(
                env, policy[1], args.task_id,
                args.episodes_per_round, max_steps, args.seed + k,
            )
        print(f'round {k}: collected {args.episodes_per_round} eps, success={collect_sr:.3f}')

        # 2. Oracle label
        obs, act, labels = label_rollout(
            rollout, goal_xyz, chunk_size, args.num_workers,
            args.max_oracle_steps, args.warmup_steps,
        )
        pos_frac = float(labels.mean()) if len(labels) else 0.0
        print(f'round {k}: labeled {len(labels)} chunks, improve_frac={pos_frac:.3f}')

        if args.no_accumulate:
            buf_obs, buf_act, buf_lab = [obs], [act], [labels]
        else:
            buf_obs.append(obs)
            buf_act.append(act)
            buf_lab.append(labels)

        all_obs = np.concatenate(buf_obs, axis=0)
        all_act = np.concatenate(buf_act, axis=0)
        all_lab = np.concatenate(buf_lab, axis=0)

        # 3. AWR → pi_{k+1}  (advantage = binary label)
        actor = train_awr_binary(
            all_obs, all_act, all_lab,
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            alpha=args.alpha,
            lr=args.lr,
            seed=args.seed + k,
            chunk_size=chunk_size,
        )
        ckpt_path = out_dir / f'round{k:03d}.pkl'
        save_actor(str(ckpt_path), actor)
        policy = ('gaussian', actor)

        # Eval
        eval_sr = evaluate_policy(
            env, policy, args.task_id, args.eval_episodes, max_steps, args.seed + 10_000 + k,
        )
        log = {
            'round': k,
            'collect/success_rate': collect_sr,
            'label/improve_frac': pos_frac,
            'label/num_chunks': len(labels),
            'dataset/size': len(all_obs),
            'dataset/improve_frac': float(all_lab.mean()),
            'eval/success_rate': eval_sr,
            **{f'train/{kk}': vv for kk, vv in actor['metrics'].items()},
        }
        wandb.log(log, step=k)
        print(
            f'round {k}: buffer={len(all_obs)} eval_success={eval_sr:.3f} '
            f'-> {ckpt_path}',
            flush=True,
        )

    env.close()
    wandb.finish()


if __name__ == '__main__':
    main()
