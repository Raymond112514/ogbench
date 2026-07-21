"""Offline AWR: collect N episodes -> fit IQL or classifier -> AWR policy extraction.

python awr/offline.py \
  --advantage iql --num_episodes 1000 --num_workers 10 --train_steps 2000 --awr_epochs 10

python awr/offline.py \
  --advantage classifier --num_episodes 1000 --num_workers 10 --train_steps 2000 --awr_epochs 10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
    p.add_argument('--num_episodes', type=int, default=1000)
    p.add_argument('--num_workers', type=int, default=10)
    p.add_argument('--n_flow_steps', type=int, default=10)
    p.add_argument('--train_steps', type=int, default=2000, help='IQL / classifier gradient steps')
    p.add_argument('--awr_epochs', type=int, default=10, help='AWR policy-extraction epochs')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--alpha', type=float, default=10.0, help='AWR temperature')
    p.add_argument('--expectile', type=float, default=0.9)
    p.add_argument('--hidden', type=int, default=256, help='Classifier hidden size')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=['cpu', 'auto'], default='cpu')
    p.add_argument('--output_dir', default='awr/data/offline')
    p.add_argument('--wandb_project', default='awr-offline')
    p.add_argument('--wandb_name', default=None)
    p.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
    p.add_argument('--max_oracle_steps', type=int, default=200)
    args = p.parse_args()

    if args.device == 'cpu':
        os.environ['JAX_PLATFORMS'] = 'cpu'

    import wandb

    from awr.collect import parallel_collect, save_rollouts
    from awr.iql.dataset import IQLDataset
    from awr.policy import extract_awr, make_classifier_advantage_fn, make_iql_advantage_fn

    wandb.init(project=args.wandb_project, name=args.wandb_name, mode=args.wandb_mode, config=vars(args))
    out = Path(args.output_dir) / args.advantage
    raw_path = out / 'rollouts.npz'

    data = parallel_collect(
        args.checkpoint, args.env_name, args.task_id, args.num_episodes, args.num_workers, args.n_flow_steps,
    )
    save_rollouts(raw_path, data)
    success_rate = data['success_rate']
    chunk_size = int(data['chunk_size'])
    print(f'collected success_rate={success_rate:.3f}')

    if args.advantage == 'classifier':
        from awr.annotate import annotate
        from awr.classifier.train import train_classifier

        ann_path = out / 'annotated.npz'
        mean_d = annotate(str(raw_path), str(ann_path), num_workers=args.num_workers, max_oracle_steps=args.max_oracle_steps)
        clf = train_classifier(ann_path, args.train_steps, args.batch_size, args.lr, args.hidden, args.seed)
        # AWR data = annotated chunk (obs, flat action_chunks)
        import numpy as np

        ann = np.load(ann_path)
        observations = np.asarray(ann['observations'], np.float32)
        actions = np.asarray(ann['action_chunks'], np.float32).reshape(len(observations), -1)
        adv_fn = make_classifier_advantage_fn(clf)
        fit_metrics = {'classifier/loss': clf['loss'], 'oracle/mean_distance': mean_d}
    else:
        from awr.iql.train import train_iql

        agent, fit_metrics = train_iql(
            [raw_path], args.train_steps, seed=args.seed, batch_size=args.batch_size,
            expectile=args.expectile, alpha=args.alpha, lr=args.lr,
        )
        ds = IQLDataset.from_paths([raw_path])
        observations, actions = ds.data['observations'], ds.data['actions']
        adv_fn = make_iql_advantage_fn(agent)

    actor = extract_awr(
        observations, actions, adv_fn, epochs=args.awr_epochs, batch_size=args.batch_size,
        alpha=args.alpha, lr=args.lr, seed=args.seed, chunk_size=chunk_size,
    )

    log = {
        'collect/success_rate': success_rate,
        'collect/num_transitions': len(data['actions']),
        **{f'train/{k}': v for k, v in fit_metrics.items()},
        **{f'awr/{k}': v for k, v in actor['metrics'].items()},
    }
    wandb.log(log)
    print(f'done offline {args.advantage}: success_rate={success_rate:.3f} awr_loss={actor["metrics"].get("actor_loss", 0):.4f}')
    wandb.finish()


if __name__ == '__main__':
    main()
