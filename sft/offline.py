"""Offline filtered BC: collect once, oracle-filter, then train with periodic eval.

  1. Collect --num_episodes with the frozen GCBC flow policy.
  2. Oracle-label chunks (y=1 if d(s)-d(s') >= H-tau; default tau=H-1 ⇒ threshold 1);
     keep only positives.
  3. Fine-tune flow-BC for --train_steps, evaluating every --eval_interval steps.

No checkpoints are saved (metrics via wandb).

Usage (from ogbench/):
  python sft/offline.py \\
    --policy_ckpt flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl \\
    --task_id 1 --num_episodes 10000 --train_steps 100000 --eval_interval 5000
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')


def _collect_worker(
    worker_id: int,
    num_episodes: int,
    policy_ckpt: str,
    env_name: str,
    task_id: int,
    n_flow_steps: int,
    max_steps: int,
    seed: int,
    egl_device: int | None,
) -> tuple[dict, float]:
    if egl_device is not None:
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(egl_device + worker_id)
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')

    import gymnasium

    import ogbench.manipspace  # noqa: F401
    from flow_bc.checkpoint import load_flow_bc
    from sft.online import collect_episodes

    model, params, meta = load_flow_bc(policy_ckpt)
    env = gymnasium.make(env_name)
    rollout, sr = collect_episodes(
        env, params, model.apply, meta, task_id, num_episodes,
        n_flow_steps, max_steps, seed,
    )
    env.close()
    print(f'collect worker {worker_id}: {num_episodes} eps, success={sr:.3f}', flush=True)
    return rollout, sr


def parallel_collect(
    policy_ckpt: str,
    env_name: str,
    task_id: int,
    num_episodes: int,
    num_workers: int,
    n_flow_steps: int,
    max_steps: int,
    seed: int,
    egl_device: int | None,
) -> tuple[dict, float]:
    splits = np.array_split(np.arange(num_episodes), num_workers)
    counts = [len(s) for s in splits if len(s) > 0]
    import multiprocessing as mp

    ctx = mp.get_context('spawn')
    rollouts = []
    srs = []
    with ProcessPoolExecutor(max_workers=len(counts), mp_context=ctx) as pool:
        futs = [
            pool.submit(
                _collect_worker,
                wid,
                n,
                str(Path(policy_ckpt).resolve()),
                env_name,
                task_id,
                n_flow_steps,
                max_steps,
                seed + wid * 100_003,
                egl_device,
            )
            for wid, n in enumerate(counts)
        ]
        for fut in futs:
            r, sr = fut.result()
            rollouts.append(r)
            srs.append(sr)

    # Weighted mean success by episode count
    total_eps = sum(counts)
    collect_sr = float(sum(sr * n for sr, n in zip(srs, counts)) / max(total_eps, 1))
    return _concat_rollouts(rollouts), collect_sr


def _concat_rollouts(rollouts: list[dict]) -> dict:
    keys = [k for k in rollouts[0] if k != 'episode_ends' and k != 'chunk_size']
    out = {k: np.concatenate([r[k] for r in rollouts], axis=0) for k in keys}
    ends = []
    offset = 0
    for r in rollouts:
        ends.append(np.asarray(r['episode_ends'], np.int32) + offset)
        offset = int(ends[-1][-1]) if len(ends[-1]) else offset
    out['episode_ends'] = np.concatenate(ends, axis=0) if ends else np.zeros(0, np.int32)
    out['chunk_size'] = rollouts[0]['chunk_size']
    return out


def train_with_periodic_eval(
    env,
    params,
    apply_fn,
    meta,
    data: dict,
    *,
    train_steps: int,
    eval_interval: int,
    eval_episodes: int,
    batch_size: int,
    lr: float,
    seed: int,
    task_id: int,
    max_steps: int,
    n_flow_steps: int,
    goal_condition: bool,
    wandb_run,
) -> object:
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state
    from tqdm import trange

    from flow_bc.model import train_step
    from sft.online import evaluate_policy

    n = len(data['observations'])
    if n == 0:
        raise ValueError('no positive chunks to train on')

    tx = optax.adam(lr)
    state = train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=tx)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    # Eval at step 0 (pretrained GCBC)
    sr0 = evaluate_policy(
        env, state.params, apply_fn, meta, task_id, eval_episodes,
        max_steps, seed, n_flow_steps,
    )
    print(f'step 0 eval success={sr0:.3f}')
    if wandb_run is not None:
        wandb_run.log({'eval/success_rate': sr0, 'train/loss': 0.0}, step=0)

    last_loss = 0.0
    for step in trange(1, train_steps + 1, desc='filtered-bc'):
        idx = rng.integers(0, n, size=min(batch_size, n))
        batch = {k: jnp.asarray(v[idx]) for k, v in data.items()}
        key, step_key = jax.random.split(key)
        state, loss = train_step(state, batch, step_key, goal_condition=goal_condition)
        last_loss = float(loss)

        if step % eval_interval == 0 or step == train_steps:
            sr = evaluate_policy(
                env, state.params, apply_fn, meta, task_id, eval_episodes,
                max_steps, seed + step, n_flow_steps,
            )
            print(f'step {step}: loss={last_loss:.4f} eval_success={sr:.3f}', flush=True)
            if wandb_run is not None:
                wandb_run.log(
                    {'eval/success_rate': sr, 'train/loss': last_loss, 'dataset/size': n},
                    step=step,
                )

    return state.params


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--policy_ckpt', default='flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl')
    p.add_argument('--env_name', default='cube-single-v0')
    p.add_argument('--task_id', type=int, default=1)
    p.add_argument('--num_episodes', type=int, default=10000)
    p.add_argument('--collect_workers', type=int, default=10, help='Parallel envs for collection')
    p.add_argument('--eval_episodes', type=int, default=50)
    p.add_argument('--eval_interval', type=int, default=5000, help='Eval every K gradient steps')
    p.add_argument('--train_steps', type=int, default=100000)
    p.add_argument('--n_flow_steps', type=int, default=10)
    p.add_argument('--max_steps', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--num_workers', type=int, default=10, help='Parallel oracle-label workers')
    p.add_argument('--max_oracle_steps', type=int, default=200)
    p.add_argument('--warmup_steps', type=int, default=2)
    p.add_argument('--tau', type=int, default=None,
                   help='Progress slack: y=1 iff d(s)-d(s\') >= H-tau. Default tau=H-1 (threshold 1)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--egl_device', type=int, default=None)
    p.add_argument('--device', choices=['cpu', 'auto'], default='cpu')
    p.add_argument('--wandb_project', default='sft-filtered-bc-offline')
    p.add_argument('--wandb_name', default=None)
    p.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
    args = p.parse_args()

    if args.device == 'cpu':
        os.environ['JAX_PLATFORMS'] = 'cpu'
    if args.egl_device is not None:
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.egl_device)

    import gymnasium
    import wandb

    import ogbench.manipspace  # noqa: F401
    from flow_bc.checkpoint import load_flow_bc
    from sft.online import label_and_filter

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

    tmp_env = gymnasium.make(env_name)
    max_steps = args.max_steps or tmp_env.spec.max_episode_steps
    goal_xyz = tmp_env.unwrapped.task_infos[args.task_id - 1]['goal_xyzs'][0].copy()
    task_name = tmp_env.unwrapped.task_infos[args.task_id - 1]['task_name']
    tmp_env.close()

    print(
        f'Offline filtered BC | env={env_name} task={args.task_id} ({task_name}) '
        f'chunk={chunk_size} tau={tau} (threshold={chunk_size - tau}) '
        f'episodes={args.num_episodes} train_steps={args.train_steps} eval_every={args.eval_interval}'
    )
    wandb.config.update({'tau': tau, 'progress_threshold': chunk_size - tau}, allow_val_change=True)

    # 1. Collect
    print(f'collecting {args.num_episodes} episodes with {args.collect_workers} workers...')
    rollout, collect_sr = parallel_collect(
        args.policy_ckpt, env_name, args.task_id, args.num_episodes,
        args.collect_workers, args.n_flow_steps, max_steps, args.seed, args.egl_device,
    )
    print(f'collect done: success={collect_sr:.3f} transitions={len(rollout["observations"])}')
    wandb.log({'collect/success_rate': collect_sr, 'collect/num_transitions': len(rollout['observations'])}, step=0)

    # 2. Label + filter
    print('oracle labeling...')
    stats, filtered = label_and_filter(
        rollout, goal_xyz, chunk_size, act_dim, args.num_workers,
        args.max_oracle_steps, args.warmup_steps, goal_condition, tau=tau,
    )
    print(
        f'labeled: chunks={stats["num_chunks"]} positives={stats["num_positive"]} '
        f'improve_frac={stats["improve_frac"]:.3f} (tau={stats["tau"]}, thr={stats["threshold"]})'
    )
    wandb.log({
        'label/num_chunks': stats['num_chunks'],
        'label/num_positive': stats['num_positive'],
        'label/improve_frac': stats['improve_frac'],
        'label/tau': stats['tau'],
        'label/threshold': stats['threshold'],
        'dataset/size': stats['num_positive'],
    }, step=0)

    if stats['num_positive'] == 0:
        raise SystemExit('no positive chunks; nothing to train on')

    # Free raw rollout memory before training
    del rollout

    # 3. Train with periodic eval
    env = gymnasium.make(env_name)
    train_with_periodic_eval(
        env, params, apply_fn, meta, filtered,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        task_id=args.task_id,
        max_steps=max_steps,
        n_flow_steps=args.n_flow_steps,
        goal_condition=goal_condition,
        wandb_run=wandb,
    )
    env.close()
    wandb.finish()


if __name__ == '__main__':
    main()
