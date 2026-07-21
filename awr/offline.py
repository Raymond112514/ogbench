"""Offline IQL + AWR on OGBench singletask datasets (same data / loop as FQL).

No GCBC collection and no online fine-tuning — pure offline RL like FQL's IQL baseline:

  python awr/offline.py --env_name=cube-single-play-singletask-task1-v0 --alpha 10
  python awr/offline.py --env_name=cube-single-play-singletask-task2-v0 --alpha 10 --data_percent 10

Uses ogbench.make_env_and_datasets (downloads cube-single-play-v0 once; rewards are
task-relabeled). Jointly trains V, Q, and the AWR actor every step.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')


def evaluate_iql(agent, env, num_episodes: int, seed: int = 0) -> float:
    """Eval success rate with the jointly trained IQL actor (single-step actions)."""
    import jax

    key = jax.random.PRNGKey(seed)
    successes = []
    for _ in range(num_episodes):
        ob, info = env.reset()
        done = False
        success = float(info.get('success', 0.0))
        while not done:
            key, sample_key = jax.random.split(key)
            action = np.array(agent.sample_actions(observations=ob, seed=sample_key, temperature=1.0))
            ob, _, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated
            success = float(info.get('success', 0.0))
        successes.append(success)
    return float(np.mean(successes))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--env_name',
        default='cube-single-play-singletask-task1-v0',
        help='OGBench singletask env/dataset name (same as FQL)',
    )
    p.add_argument('--train_steps', type=int, default=1_000_000, help='FQL offline_steps default: 1e6')
    p.add_argument('--log_interval', type=int, default=5000)
    p.add_argument('--eval_interval', type=int, default=100_000)
    p.add_argument('--eval_episodes', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--alpha', type=float, default=10.0, help='AWR temperature (FQL IQL default: 10)')
    p.add_argument('--expectile', type=float, default=0.9)
    p.add_argument('--discount', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument(
        '--data_percent',
        type=float,
        default=100.0,
        help='Percent of OGBench training transitions to use (1–100; subsampled with --seed)',
    )
    p.add_argument('--device', choices=['cpu', 'auto'], default='cpu')
    p.add_argument('--dataset_dir', default=None, help='Override OGBench dataset dir (default ~/.ogbench/data)')
    p.add_argument('--wandb_project', default='awr-offline')
    p.add_argument('--wandb_name', default=None)
    p.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
    args = p.parse_args()

    if not (0 < args.data_percent <= 100):
        p.error(f'--data_percent must be in (0, 100], got {args.data_percent}')

    if args.device == 'cpu':
        os.environ['JAX_PLATFORMS'] = 'cpu'

    import ogbench
    import wandb

    from awr.iql.dataset import IQLDataset
    from awr.iql.train import train_iql

    wandb.init(project=args.wandb_project, name=args.wandb_name, mode=args.wandb_mode, config=vars(args))

    kwargs = {}
    if args.dataset_dir is not None:
        kwargs['dataset_dir'] = args.dataset_dir

    print(f'loading OGBench dataset: {args.env_name}')
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(args.env_name, **kwargs)
    dataset = IQLDataset.from_ogbench(train_dataset)
    full_size = dataset.size
    dataset = dataset.subsample(args.data_percent, seed=args.seed)
    print(
        f'dataset size={dataset.size}/{full_size} ({args.data_percent:g}%)  '
        f'obs={dataset.data["observations"].shape}  act={dataset.data["actions"].shape}'
    )
    wandb.log(
        {
            'dataset/size': dataset.size,
            'dataset/full_size': full_size,
            'dataset/data_percent': args.data_percent,
        },
        step=0,
    )

    def eval_fn(agent, step):
        sr = evaluate_iql(agent, env, args.eval_episodes, seed=args.seed)
        wandb.log({'evaluation/success_rate': sr}, step=step)
        print(f'step {step}: eval success_rate={sr:.3f}', flush=True)

    agent, fit_metrics = train_iql(
        args.train_steps,
        dataset=dataset,
        seed=args.seed,
        batch_size=args.batch_size,
        expectile=args.expectile,
        alpha=args.alpha,
        lr=args.lr,
        discount=args.discount,
        tau=args.tau,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval if args.eval_episodes > 0 else 0,
        eval_fn=eval_fn if args.eval_episodes > 0 else None,
        wandb_run=wandb,
    )

    log = {f'train/{k}': v for k, v in fit_metrics.items()}
    if args.eval_episodes > 0:
        sr = evaluate_iql(agent, env, args.eval_episodes, seed=args.seed)
        log['evaluation/success_rate'] = sr
        print(f'done offline iql: eval success_rate={sr:.3f}')
    wandb.log(log)
    wandb.finish()


if __name__ == '__main__':
    main()
