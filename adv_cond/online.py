"""Online advantage-conditioned flow-BC with oracle or learned classifier labels.

Single-task: goal_condition=False, advantage_condition=True (cond = [obs, A]).

Round k (default 30):
  1. Collect N episodes.
       k=0: frozen GCBC (goal-cond) for data only.
       k>0: current adv-cond policy with CFG requesting A=1.
  2. Set binary A using oracle progress, or threshold a classifier trained to
     distinguish chunks from successful vs failed episodes.
  3. Accumulate (s, chunk, A) over all rounds; take a few flow grad steps with CFG dropout.
  4. Eval success at guidance scales {0, 0.25, 0.5, 0.75, 1, 2, 3} (request A=1).

No persistent checkpoints — wandb only.

Usage (from ogbench/):
  python adv_cond/online.py \\
    --policy_ckpt flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl \\
    --task_id 1 --rounds 30 --episodes_per_round 100 --tau 5
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

CFG_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0)


def _capture(env) -> np.ndarray:
    from awr.oracle_utils import capture_sim_state

    return capture_sim_state(env)


def _chunk_boundary_indices(episode_ends: np.ndarray, chunk_size: int) -> np.ndarray:
    starts = [0] + episode_ends[:-1].tolist()
    out = []
    for start, end in zip(starts, episode_ends.tolist()):
        out.extend(range(start, end, chunk_size))
    return np.asarray(out, np.int64)


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


def label_chunks(
    rollout: dict,
    goal_xyz: np.ndarray,
    chunk_size: int,
    num_workers: int,
    max_oracle_steps: int,
    warmup_steps: int,
    tau: int | None,
) -> tuple[dict, dict]:
    """Oracle-label all full chunks; return (stats, batch dict with advantages)."""
    from awr.classifier.ogbench_dataset import episode_ends as ends_from_terminals
    from awr.oracle_utils import progress_label

    if tau is None:
        tau = chunk_size - 1
    threshold = chunk_size - int(tau)

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

    obs_list, chunk_list, mask_list, adv_list = [], [], [], []
    n_total, n_pos = 0, 0
    start = 0
    for end in ends_from_terminals(terminals, n):
        t = start
        while t + chunk_size < end:
            n_total += 1
            y = progress_label(distance[t], distance[t + chunk_size], chunk_size, tau)
            if y > 0.5:
                n_pos += 1
            obs_list.append(rollout['observations'][t])
            chunk_list.append(rollout['actions'][t : t + chunk_size])
            mask_list.append(np.ones(chunk_size, np.float32))
            adv_list.append(float(y))
            t += chunk_size
        start = end

    stats = {
        'num_chunks': n_total,
        'num_positive': n_pos,
        'improve_frac': float(n_pos / max(n_total, 1)),
        'tau': int(tau),
        'threshold': int(threshold),
    }
    data = {
        'observations': np.asarray(obs_list, np.float32),
        'action_chunks': np.asarray(chunk_list, np.float32),
        'chunk_masks': np.asarray(mask_list, np.float32),
        'advantages': np.asarray(adv_list, np.float32),
    }
    return stats, data


def merge_data(buffers: list[dict]) -> dict:
    keys = buffers[0].keys()
    return {k: np.concatenate([b[k] for b in buffers], axis=0) for k in keys}


def collect_gcbc_round(
    policy_ckpt: str,
    env_name: str,
    task_id: int,
    n_episodes: int,
    num_workers: int,
    n_flow_steps: int,
) -> tuple[dict, float]:
    """Collect with frozen GCBC via bon_sampling parallel collector."""
    import gymnasium

    import ogbench.manipspace  # noqa: F401
    import bon_sampling.collect_transitions as ct
    from flow_bc.checkpoint import read_ckpt_meta

    ct.NUM_VIDEO_EPISODES = 0
    meta = read_ckpt_meta(policy_ckpt)
    tmp_env = gymnasium.make(env_name)
    max_steps = tmp_env.spec.max_episode_steps
    tmp_env.close()

    obs, act, next_obs, state, step_succ, ends, successes, _, init_state = ct.parallel_collect(
        policy_ckpt, env_name, task_id, n_episodes, num_workers, max_steps, n_flow_steps,
        None, 'cpu', [], None, 'auto', 8,
    )
    rollout = dict(
        observations=np.asarray(obs, np.float32),
        actions=np.asarray(act, np.float32),
        next_observations=np.asarray(next_obs, np.float32),
        next_mjstate=np.asarray(state, np.float64),
        successes=np.asarray(step_succ, np.bool_),
        episode_initial_mjstate=np.asarray(init_state, np.float64),
        episode_ends=np.asarray(ends, np.int32),
        chunk_size=np.array(meta['chunk_size']),
    )
    return rollout, float(np.mean(successes))


def collect_adv_cond(
    env,
    params,
    apply_fn,
    meta: dict,
    task_id: int,
    num_episodes: int,
    n_flow_steps: int,
    max_steps: int,
    seed: int,
    cfg_weight: float,
    cfg_cond_advantage: float,
) -> tuple[dict, float]:
    """Collect with advantage-conditioned policy + CFG (request high A)."""
    import jax
    import jax.numpy as jnp

    from flow_bc.model import sample_action_chunk

    chunk_size = meta['chunk_size']
    act_dim = meta['act_dim']
    key = jax.random.PRNGKey(seed)

    observations, actions, successes = [], [], []
    next_mjstate, episode_initial_mjstate, episode_ends = [], [], []

    for _ in range(num_episodes):
        ob, _info = env.reset(options=dict(task_id=task_id))
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
                    None,
                    sample_key,
                    chunk_size=chunk_size,
                    act_dim=act_dim,
                    n_flow_steps=n_flow_steps,
                    advantage=cfg_cond_advantage,
                    goal_condition=False,
                    advantage_condition=True,
                    cfg_weight=cfg_weight,
                    cfg_cond_advantage=cfg_cond_advantage,
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
    return rollout, success_rate


def evaluate_cfg(
    env,
    params,
    apply_fn,
    meta: dict,
    task_id: int,
    num_episodes: int,
    max_steps: int,
    seed: int,
    n_flow_steps: int,
    cfg_weight: float,
    cfg_cond_advantage: float,
) -> float:
    _, sr = collect_adv_cond(
        env, params, apply_fn, meta, task_id, num_episodes, n_flow_steps, max_steps, seed,
        cfg_weight=cfg_weight, cfg_cond_advantage=cfg_cond_advantage,
    )
    return sr


def init_adv_cond_policy(obs_dim: int, act_dim: int, chunk_size: int, hidden_dims, seed: int):
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state

    from flow_bc.model import VelocityNet, cond_dim

    out_dim = chunk_size * act_dim
    cdim = cond_dim(obs_dim, goal_condition=False, advantage_condition=True)
    model = VelocityNet(hidden_dims=tuple(hidden_dims), out_dim=out_dim)
    key = jax.random.PRNGKey(seed)
    params = model.init(
        key,
        jnp.zeros((1, out_dim)),
        jnp.zeros((1,)),
        jnp.zeros((1, cdim)),
    )
    return model, params


def train_adv_cond(
    params,
    apply_fn,
    data: dict,
    *,
    train_steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    cfg_dropout: float,
) -> tuple[object, float]:
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state
    from tqdm import trange

    from flow_bc.model import train_step

    n = len(data['observations'])
    if n == 0:
        raise ValueError('empty training buffer')

    state = train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=optax.adam(lr))
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    last_loss = 0.0

    for _ in trange(train_steps, desc='adv-cond', leave=False):
        idx = rng.integers(0, n, size=min(batch_size, n))
        batch = {k: jnp.asarray(v[idx]) for k, v in data.items()}
        key, step_key = jax.random.split(key)
        state, loss = train_step(
            state,
            batch,
            step_key,
            goal_condition=False,
            advantage_condition=True,
            cfg_dropout=cfg_dropout,
        )
        last_loss = float(loss)
    return state.params, last_loss


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--policy_ckpt', default='flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl',
                   help='Frozen GCBC used only for round-0 collection / arch dims')
    p.add_argument('--env_name', default='cube-single-v0')
    p.add_argument('--task_id', type=int, default=1)
    p.add_argument('--rounds', type=int, default=30)
    p.add_argument('--episodes_per_round', type=int, default=100)
    p.add_argument('--eval_episodes', type=int, default=50)
    p.add_argument('--n_flow_steps', type=int, default=10)
    p.add_argument('--max_steps', type=int, default=None)
    p.add_argument('--train_steps', type=int, default=5000)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--cfg_dropout', type=float, default=0.1,
                   help='Train-time prob of replacing A with null (-1) for CFG')
    p.add_argument('--collect_cfg_weight', type=float, default=1.0,
                   help='CFG weight when collecting with adv-cond policy (rounds > 0)')
    p.add_argument('--cfg_cond_advantage', type=float, default=1.0,
                   help='Requested advantage at collect/eval (binary positive = 1)')
    p.add_argument('--num_workers', type=int, default=10)
    p.add_argument('--max_oracle_steps', type=int, default=200)
    p.add_argument('--warmup_steps', type=int, default=2)
    p.add_argument('--tau', type=int, default=None,
                   help='A=1 iff d(s)-d(s\') >= H-tau. Default tau=H-1')
    p.add_argument('--feedback', choices=['oracle', 'success_classifier'], default='oracle')
    p.add_argument('--classifier_tau', type=float, default=0.5,
                   help='Success-classifier probability threshold (default: 0.5)')
    p.add_argument('--classifier_steps', type=int, default=2000)
    p.add_argument('--classifier_batch_size', type=int, default=256)
    p.add_argument('--classifier_lr', type=float, default=3e-4)
    p.add_argument('--classifier_hidden', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=['cpu', 'auto'], default='cpu')
    p.add_argument('--wandb_project', default='adv-cond-online')
    p.add_argument('--wandb_name', default=None)
    p.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
    args = p.parse_args()

    if args.device == 'cpu':
        os.environ['JAX_PLATFORMS'] = 'cpu'

    import gymnasium
    import wandb

    import ogbench.manipspace  # noqa: F401
    from flow_bc.checkpoint import read_ckpt_meta

    gcbc_meta = read_ckpt_meta(args.policy_ckpt)
    env_name = gcbc_meta.get('eval_env_name', args.env_name)
    chunk_size = int(gcbc_meta['chunk_size'])
    act_dim = int(gcbc_meta['act_dim'])
    obs_dim = int(gcbc_meta['obs_dim'])
    hidden_dims = tuple(gcbc_meta['hidden_dims'])
    tau = chunk_size - 1 if args.tau is None else int(args.tau)

    env = gymnasium.make(env_name)
    max_steps = args.max_steps or env.spec.max_episode_steps
    goal_xyz = env.unwrapped.task_infos[args.task_id - 1]['goal_xyzs'][0].copy()
    task_name = env.unwrapped.task_infos[args.task_id - 1]['task_name']

    model, params = init_adv_cond_policy(obs_dim, act_dim, chunk_size, hidden_dims, args.seed)
    apply_fn = model.apply
    meta = {
        'obs_dim': obs_dim,
        'act_dim': act_dim,
        'chunk_size': chunk_size,
        'hidden_dims': hidden_dims,
        'goal_condition': False,
        'advantage_condition': True,
    }

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config={
            **vars(args),
            'chunk_size': chunk_size,
            'tau': tau,
            'progress_threshold': chunk_size - tau,
            'classifier_tau': args.classifier_tau,
            'cfg_scales': list(CFG_SCALES),
            'task_name': task_name,
            'env_name': env_name,
        },
    )

    print(
        f'AdvCond | env={env_name} task={args.task_id} ({task_name}) '
        f'chunk={chunk_size} tau={tau} thr={chunk_size - tau} '
        f'rounds={args.rounds} eps/round={args.episodes_per_round} '
        f'cfg_dropout={args.cfg_dropout}'
    )

    buffers: list[dict] = []
    classifier_buffers: list[dict] = []
    classifier_params = None
    adv_policy_trained = False

    for k in range(args.rounds):
        if not adv_policy_trained:
            rollout, collect_sr = collect_gcbc_round(
                args.policy_ckpt, env_name, args.task_id, args.episodes_per_round,
                args.num_workers, args.n_flow_steps,
            )
            collect_mode = 'gcbc'
        else:
            rollout, collect_sr = collect_adv_cond(
                env, params, apply_fn, meta, args.task_id, args.episodes_per_round,
                args.n_flow_steps, max_steps, args.seed + 1000 + k,
                cfg_weight=args.collect_cfg_weight,
                cfg_cond_advantage=args.cfg_cond_advantage,
            )
            collect_mode = f'adv_cond_cfg{args.collect_cfg_weight}'

        print(f'round {k}: collect={collect_mode} success={collect_sr:.3f}', flush=True)

        classifier_metrics = {}
        if args.feedback == 'oracle':
            stats, labeled = label_chunks(
                rollout, goal_xyz, chunk_size, args.num_workers,
                args.max_oracle_steps, args.warmup_steps, tau,
            )
        else:
            from bon_sampling.advantage.success_feedback import (
                build_episode_success_chunks,
                merge_success_chunks,
                predict_success_probabilities,
                threshold_success_feedback,
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
                print(
                    f'round {k}: success classifier needs both successful and failed episodes; '
                    'skipping adv-cond update',
                    flush=True,
                )
                wandb.log({
                    'round': k,
                    'collect/success_rate': collect_sr,
                    **{f'classifier/{key}': value for key, value in classifier_metrics.items()},
                }, step=k)
                continue
            probabilities = predict_success_probabilities(
                classifier_params,
                current_chunks,
                hidden=args.classifier_hidden,
                batch_size=args.classifier_batch_size,
            )
            labeled, predicted_stats = threshold_success_feedback(
                current_chunks, probabilities, args.classifier_tau
            )
            stats = {
                'num_chunks': int(predicted_stats['num_chunks']),
                'num_positive': int(predicted_stats['num_positive']),
                'improve_frac': predicted_stats['positive_frac'],
                'tau': predicted_stats['threshold'],
                'threshold': predicted_stats['threshold'],
                'mean_probability': predicted_stats['mean_probability'],
            }
        print(
            f'round {k}: chunks={stats["num_chunks"]} pos={stats["num_positive"]} '
            f'improve_frac={stats["improve_frac"]:.3f}',
            flush=True,
        )
        if stats['num_chunks'] == 0:
            raise SystemExit(f'round {k}: no labeled chunks')

        buffers.append(labeled)
        data = merge_data(buffers)

        params, last_loss = train_adv_cond(
            params, apply_fn, data,
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + k,
            cfg_dropout=args.cfg_dropout,
        )
        adv_policy_trained = True

        log = {
            'round': k,
            'collect/mode': collect_mode,
            'collect/success_rate': collect_sr,
            'label/num_chunks': stats['num_chunks'],
            'label/num_positive': stats['num_positive'],
            'label/improve_frac': stats['improve_frac'],
            'label/tau': stats['tau'],
            'label/threshold': stats['threshold'],
            'label/mean_probability': stats.get('mean_probability', 0.0),
            **{f'classifier/{key}': value for key, value in classifier_metrics.items()},
            'dataset/size': len(data['observations']),
            'dataset/pos_rate': float(np.mean(data['advantages'])),
            'train/loss': last_loss,
        }

        for w in CFG_SCALES:
            # Distinct seeds per scale so episodes differ but are comparable across rounds.
            sr = evaluate_cfg(
                env, params, apply_fn, meta, args.task_id, args.eval_episodes,
                max_steps, args.seed + 10_000 + k * 100 + int(w * 100),
                args.n_flow_steps, cfg_weight=w, cfg_cond_advantage=args.cfg_cond_advantage,
            )
            key = f'eval/cfg_{w:g}'
            log[key] = sr
            print(f'round {k}: {key}={sr:.3f}', flush=True)

        wandb.log(log, step=k)
        print(
            f'round {k}: buffer={len(data["observations"])} loss={last_loss:.4f}',
            flush=True,
        )

    env.close()
    wandb.finish()


if __name__ == '__main__':
    main()
