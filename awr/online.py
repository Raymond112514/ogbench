"""Online AWR: collect -> fit -> collect with updated policy; repeat.

IQL path matches FQL: joint V + Q + AWR actor (no separate extraction stage).
Classifier path: fit classifier then AWR-extract for `awr_epochs`.

Round 0 uses the GCBC base policy.

python awr/online.py \
  --advantage iql --rounds 30 --episodes_per_round 100 --num_workers 10 \
  --train_steps 2000 --device cpu
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='flow_bc/checkpoints/cube_single_gcbc/best.pkl')
    p.add_argument('--advantage', choices=['iql', 'classifier'], default='iql')
    p.add_argument('--env_name', default='cube-single-v0')
    p.add_argument('--task_id', type=int, default=1)
    p.add_argument('--rounds', type=int, default=30)
    p.add_argument('--episodes_per_round', type=int, default=100)
    p.add_argument('--num_workers', type=int, default=10)
    p.add_argument('--n_flow_steps', type=int, default=10)
    p.add_argument('--train_steps', type=int, default=2000, help='IQL joint / classifier gradient steps')
    p.add_argument('--log_interval', type=int, default=500, help='IQL wandb log interval within a round')
    p.add_argument('--awr_epochs', type=int, default=10, help='Classifier-only: AWR extraction epochs')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--alpha', type=float, default=10.0)
    p.add_argument('--expectile', type=float, default=0.9)
    p.add_argument('--discount', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=['cpu', 'auto'], default='cpu')
    p.add_argument('--wandb_project', default='awr-online')
    p.add_argument('--wandb_name', default=None)
    p.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
    p.add_argument('--max_oracle_steps', type=int, default=200)
    args = p.parse_args()

    if args.device == 'cpu':
        os.environ['JAX_PLATFORMS'] = 'cpu'

    import wandb

    from awr.collect import parallel_collect
    from awr.policy import actor_ckpt_from_iql, extract_awr, make_classifier_advantage_fn

    wandb.init(project=args.wandb_project, name=args.wandb_name, mode=args.wandb_mode, config=vars(args))

    actor_ckpt = None
    rollouts: list[dict] = []
    annotated: list[dict] = []

    for r in range(args.rounds):
        data = parallel_collect(
            args.checkpoint, args.env_name, args.task_id, args.episodes_per_round, args.num_workers,
            args.n_flow_steps, actor_ckpt=actor_ckpt,
        )
        rollouts.append(data)
        success_rate = data['success_rate']
        chunk_size = int(data['chunk_size'])

        log = {
            'round': r,
            'collect/mode': 1 if actor_ckpt is not None else 0,
            'collect/success_rate': success_rate,
            'collect/num_transitions': len(data['actions']),
            'train/num_datasets': len(rollouts),
        }

        if args.advantage == 'classifier':
            from awr.annotate import annotate_rollouts
            from awr.classifier.dataset import merge_annotated
            from awr.classifier.train import train_classifier

            ann, mean_d = annotate_rollouts(
                data, num_workers=args.num_workers, max_oracle_steps=args.max_oracle_steps,
            )
            annotated.append(ann)
            merged = merge_annotated(annotated)
            clf = train_classifier(merged, args.train_steps, args.batch_size, args.lr, args.hidden, args.seed)
            observations = np.asarray(merged['observations'], np.float32)
            actions = np.asarray(merged['action_chunks'], np.float32).reshape(len(observations), -1)
            adv_fn = make_classifier_advantage_fn(clf)
            actor_ckpt = extract_awr(
                observations, actions, adv_fn, epochs=args.awr_epochs, batch_size=args.batch_size,
                alpha=args.alpha, lr=args.lr, seed=args.seed + r, chunk_size=chunk_size,
            )
            log['collect/mean_oracle_distance'] = mean_d
            log['train/classifier_loss'] = clf['loss']
            log.update({f'awr/{k}': v for k, v in actor_ckpt['metrics'].items()})
        else:
            from awr.iql.train import train_iql

            # Joint V/Q/actor like FQL; deploy the jointly trained actor (no second AWR stage).
            agent, fit_metrics = train_iql(
                args.train_steps, rollouts=rollouts, seed=args.seed + r, batch_size=args.batch_size,
                expectile=args.expectile, alpha=args.alpha, lr=args.lr,
                discount=args.discount, tau=args.tau,
                log_interval=args.log_interval, wandb_run=None,
            )
            actor_ckpt = actor_ckpt_from_iql(agent, chunk_size)
            log.update({f'train/{k}': v for k, v in fit_metrics.items()})

        wandb.log(log)
        print(f'round {r} [{args.advantage}]: success_rate={success_rate:.3f}')

    wandb.finish()


if __name__ == '__main__':
    main()
