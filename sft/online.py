"""Filtered BC online loop with oracle or learned success-classifier feedback.

Round k:
  1. Collect episodes with the current flow policy pi_k (pi_0 = pretrained GCBC).
  2. Label each action chunk using oracle progress, or threshold a classifier trained
     to distinguish chunks from successful vs failed episodes.
  3. Keep only y=1 chunks and continue flow-BC training → pi_{k+1}.

Positive chunks accumulate across rounds by default (true filtered BC, not AWR).

Usage (from ogbench/):
  python sft/online.py \\
    --policy_ckpt flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl \\
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


def collect_episodes(
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

    observations, actions, goals, successes = [], [], [], []
    next_mjstate, episode_initial_mjstate, episode_ends = [], [], []

    for _ in range(num_episodes):
        ob, info = env.reset(options=dict(task_id=task_id))
        goal = np.asarray(info['goal'], np.float32) if goal_condition else None
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
                if goal_condition:
                    goals.append(goal.copy())
                next_mjstate.append(_capture(env))
                successes.append(bool(info.get('success', False)))
                steps += 1
        episode_ends.append(len(actions))

    ends = np.asarray(episode_ends, np.int32)
    succ = np.asarray(successes, np.bool_)
    success_rate = float(succ[ends - 1].mean()) if len(ends) else 0.0
    rollout = dict(
        observations=np.asarray(observations, np.float32),
        actions=np.asarray(actions, np.float32),
        successes=succ,
        next_mjstate=np.asarray(next_mjstate, np.float64),
        episode_initial_mjstate=np.asarray(episode_initial_mjstate, np.float64),
        episode_ends=ends,
        chunk_size=np.array(chunk_size),
    )
    if goal_condition:
        rollout['goals'] = np.asarray(goals, np.float32)
    return rollout, success_rate


def evaluate_policy(
    env, params, apply_fn, meta, task_id, num_episodes, max_steps, seed, n_flow_steps,
) -> float:
    _, sr = collect_episodes(
        env, params, apply_fn, meta, task_id, num_episodes, n_flow_steps, max_steps, seed,
    )
    return sr


# ---------------------------------------------------------------------------
# Oracle labeling → filtered chunks
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


def label_and_filter(
    rollout: dict,
    goal_xyz: np.ndarray,
    chunk_size: int,
    act_dim: int,
    num_workers: int,
    max_oracle_steps: int,
    warmup_steps: int,
    goal_condition: bool,
    tau: int | None = None,
) -> tuple[dict, dict]:
    """Oracle-label chunks; return (stats, filtered flow-BC batch dict of positives only).

    Positive if d(s) - d(s') >= H - tau (default tau=H-1 ⇒ need improve by >= 1).
    """
    from awr.classifier.ogbench_dataset import episode_ends as ends_from_terminals
    from awr.oracle_utils import progress_label

    if tau is None:
        tau = chunk_size - 1
    threshold = chunk_size - tau

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

    obs_list, goal_list, chunk_list, mask_list = [], [], [], []
    n_total, n_pos = 0, 0
    start = 0
    for end in ends_from_terminals(terminals, n):
        t = start
        while t + chunk_size < end:
            n_total += 1
            if progress_label(distance[t], distance[t + chunk_size], chunk_size, tau) > 0.5:
                n_pos += 1
                chunk = rollout['actions'][t : t + chunk_size]
                obs_list.append(rollout['observations'][t])
                chunk_list.append(chunk)
                mask_list.append(np.ones(chunk_size, np.float32))
                if goal_condition:
                    goal_list.append(rollout['goals'][t])
            t += chunk_size
        start = end

    stats = {
        'num_chunks': n_total,
        'num_positive': n_pos,
        'improve_frac': float(n_pos / max(n_total, 1)),
        'tau': int(tau),
        'threshold': int(threshold),
    }
    if n_pos == 0:
        empty = {
            'observations': np.zeros((0, rollout['observations'].shape[1]), np.float32),
            'action_chunks': np.zeros((0, chunk_size, act_dim), np.float32),
            'chunk_masks': np.zeros((0, chunk_size), np.float32),
        }
        if goal_condition:
            empty['goals'] = np.zeros((0, rollout['goals'].shape[1]), np.float32)
        return stats, empty

    filtered = {
        'observations': np.asarray(obs_list, np.float32),
        'action_chunks': np.asarray(chunk_list, np.float32),
        'chunk_masks': np.asarray(mask_list, np.float32),
    }
    if goal_condition:
        filtered['goals'] = np.asarray(goal_list, np.float32)
    return stats, filtered


def merge_filtered(buffers: list[dict]) -> dict:
    keys = buffers[0].keys()
    return {k: np.concatenate([b[k] for b in buffers], axis=0) for k in keys}


def filter_classifier_chunks(
    chunks: dict[str, np.ndarray],
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict, dict]:
    """Keep chunks whose predicted episode-success probability clears threshold."""
    keep = np.asarray(probabilities) >= threshold
    keys = ('observations', 'action_chunks', 'chunk_masks', 'goals')
    filtered = {key: chunks[key][keep] for key in keys if key in chunks}
    stats = {
        'num_chunks': int(len(keep)),
        'num_positive': int(keep.sum()),
        'improve_frac': float(keep.mean()) if len(keep) else 0.0,
        'tau': float(threshold),
        'threshold': float(threshold),
        'mean_probability': float(np.mean(probabilities)) if len(probabilities) else 0.0,
    }
    return stats, filtered


# ---------------------------------------------------------------------------
# Filtered flow-BC fine-tune
# ---------------------------------------------------------------------------


def continue_flow_bc(
    params,
    apply_fn,
    data: dict,
    *,
    train_steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    goal_condition: bool,
) -> tuple[object, float]:
    """Continue Adam updates on filtered positive chunks; return (new_params, last_loss)."""
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state
    from tqdm import trange

    from flow_bc.model import train_step

    n = len(data['observations'])
    if n == 0:
        raise ValueError('no positive chunks to train on')

    tx = optax.adam(lr)
    state = train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=tx)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    last_loss = 0.0

    for _ in trange(train_steps, desc='filtered-bc', leave=False):
        idx = rng.integers(0, n, size=min(batch_size, n))
        batch = {k: jnp.asarray(v[idx]) for k, v in data.items()}
        key, step_key = jax.random.split(key)
        state, loss = train_step(state, batch, step_key, goal_condition=goal_condition)
        last_loss = float(loss)

    return state.params, last_loss


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
    p.add_argument('--train_steps', type=int, default=5000, help='Flow-BC gradient steps per round')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--num_workers', type=int, default=10, help='Parallel oracle-label workers')
    p.add_argument('--max_oracle_steps', type=int, default=200)
    p.add_argument('--warmup_steps', type=int, default=2)
    p.add_argument('--tau', type=int, default=None,
                   help='Progress slack: y=1 iff d(s)-d(s\') >= H-tau. Default tau=H-1 (threshold 1)')
    p.add_argument('--feedback', choices=['oracle', 'success_classifier'], default='oracle')
    p.add_argument('--classifier_tau', type=float, default=0.5,
                   help='Success-classifier probability threshold (default: 0.5)')
    p.add_argument('--classifier_steps', type=int, default=2000)
    p.add_argument('--classifier_batch_size', type=int, default=256)
    p.add_argument('--classifier_lr', type=float, default=3e-4)
    p.add_argument('--classifier_hidden', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no_accumulate', action='store_true',
                   help='Train each round only on that round\'s positives (default: accumulate)')
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
    from flow_bc.checkpoint import load_flow_bc

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config=vars(args),
    )

    model, params, meta = load_flow_bc(args.policy_ckpt)
    env_name = meta.get('eval_env_name', args.env_name)
    chunk_size = int(meta['chunk_size'])
    act_dim = int(meta['act_dim'])
    goal_condition = bool(meta['goal_condition'])
    apply_fn = model.apply
    tau = chunk_size - 1 if args.tau is None else int(args.tau)

    env = gymnasium.make(env_name)
    max_steps = args.max_steps or env.spec.max_episode_steps
    goal_xyz = env.unwrapped.task_infos[args.task_id - 1]['goal_xyzs'][0].copy()
    task_name = env.unwrapped.task_infos[args.task_id - 1]['task_name']

    print(
        f'Filtered BC | env={env_name} task={args.task_id} ({task_name}) '
        f'chunk={chunk_size} tau={tau} (threshold={chunk_size - tau}) '
        f'rounds={args.rounds} eps/round={args.episodes_per_round}'
    )
    wandb.config.update({
        'tau': tau,
        'progress_threshold': chunk_size - tau,
        'classifier_tau': args.classifier_tau,
    }, allow_val_change=True)

    buffers: list[dict] = []
    classifier_buffers: list[dict] = []
    classifier_params = None

    sr0 = evaluate_policy(
        env, params, apply_fn, meta, args.task_id, args.eval_episodes,
        max_steps, args.seed, args.n_flow_steps,
    )
    print(f'round 0 (GCBC) eval success={sr0:.3f}')
    wandb.log({'round': 0, 'eval/success_rate': sr0, 'policy': 'gcbc'}, step=0)

    for k in range(1, args.rounds + 1):
        rollout, collect_sr = collect_episodes(
            env, params, apply_fn, meta, args.task_id,
            args.episodes_per_round, args.n_flow_steps, max_steps, args.seed + k,
        )
        print(f'round {k}: collected {args.episodes_per_round} eps, success={collect_sr:.3f}')

        classifier_metrics = {}
        if args.feedback == 'oracle':
            stats, filtered = label_and_filter(
                rollout, goal_xyz, chunk_size, act_dim, args.num_workers,
                args.max_oracle_steps, args.warmup_steps, goal_condition, tau=tau,
            )
        else:
            from bon_sampling.advantage.success_feedback import (
                build_episode_success_chunks,
                merge_success_chunks,
                predict_success_probabilities,
                train_success_classifier,
            )

            current_chunks = build_episode_success_chunks(rollout, chunk_size)
            classifier_buffers.append(current_chunks)
            classifier_data = merge_success_chunks(classifier_buffers)
            classifier_params, classifier_metrics = train_success_classifier(
                classifier_params,
                classifier_data,
                hidden=args.classifier_hidden,
                train_steps=args.classifier_steps,
                batch_size=args.classifier_batch_size,
                lr=args.classifier_lr,
                seed=args.seed + k,
            )
            if classifier_params is None:
                stats = {
                    'num_chunks': len(current_chunks['observations']),
                    'num_positive': 0,
                    'improve_frac': 0.0,
                    'tau': args.classifier_tau,
                    'threshold': args.classifier_tau,
                    'mean_probability': 0.0,
                }
                filtered = {
                    key: value[:0]
                    for key, value in current_chunks.items()
                    if key not in ('success_labels',)
                }
            else:
                probabilities = predict_success_probabilities(
                    classifier_params,
                    current_chunks,
                    hidden=args.classifier_hidden,
                    batch_size=args.classifier_batch_size,
                )
                stats, filtered = filter_classifier_chunks(
                    current_chunks, probabilities, args.classifier_tau
                )
        print(
            f'round {k}: chunks={stats["num_chunks"]} '
            f'positives={stats["num_positive"]} improve_frac={stats["improve_frac"]:.3f} '
            f'(tau={stats["tau"]}, thr={stats["threshold"]})'
        )

        if stats['num_positive'] == 0:
            print(f'round {k}: no positive chunks; skipping train', flush=True)
            wandb.log({
                'round': k,
                'collect/success_rate': collect_sr,
                'label/improve_frac': stats['improve_frac'],
                'label/num_chunks': stats['num_chunks'],
                'label/num_positive': 0,
                'label/tau': stats['tau'],
                'label/threshold': stats['threshold'],
                'label/mean_probability': stats.get('mean_probability', 0.0),
                **{f'classifier/{key}': value for key, value in classifier_metrics.items()},
                'dataset/size': sum(len(b['observations']) for b in buffers),
                'eval/success_rate': collect_sr,
            }, step=k)
            continue

        if args.no_accumulate:
            buffers = [filtered]
        else:
            buffers.append(filtered)
        data = merge_filtered(buffers)

        params, last_loss = continue_flow_bc(
            params, apply_fn, data,
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + k,
            goal_condition=goal_condition,
        )

        eval_sr = evaluate_policy(
            env, params, apply_fn, meta, args.task_id, args.eval_episodes,
            max_steps, args.seed + 10_000 + k, args.n_flow_steps,
        )
        log = {
            'round': k,
            'collect/success_rate': collect_sr,
            'label/improve_frac': stats['improve_frac'],
            'label/num_chunks': stats['num_chunks'],
            'label/num_positive': stats['num_positive'],
            'label/tau': stats['tau'],
            'label/threshold': stats['threshold'],
            'label/mean_probability': stats.get('mean_probability', 0.0),
            **{f'classifier/{key}': value for key, value in classifier_metrics.items()},
            'dataset/size': len(data['observations']),
            'train/loss': last_loss,
            'eval/success_rate': eval_sr,
        }
        wandb.log(log, step=k)
        print(
            f'round {k}: buffer={len(data["observations"])} loss={last_loss:.4f} '
            f'eval_success={eval_sr:.3f}',
            flush=True,
        )

    env.close()
    wandb.finish()


if __name__ == '__main__':
    main()
