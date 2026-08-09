"""Online AWR with GCBC bootstrap (same schedule as bon_sampling online).

  1. Collect --initial_episodes (default 1000) with the GCBC policy.
  2. Annotate if needed (classifier / oracle_delta / oracle_binned).
  3. Fit advantage (IQL V/Q, classifier, or direct oracle Δ / signed bins).
  4. Extract an AWR policy.
  5. For each of --rounds refinement steps:
       collect --episodes_per_round (default 100) with the extracted policy,
       annotate, fit advantage on all data so far, extract policy.

python awr/online.py \
  --advantage iql --rounds 30 --initial_episodes 1000 --episodes_per_round 100 \
  --num_workers 10 --train_steps 2000 --awr_epochs 10 --device cpu

python awr/online.py \
  --advantage classifier --rounds 30 --initial_episodes 1000 --episodes_per_round 100 \
  --num_workers 10 --train_steps 2000 --awr_epochs 10 --device cpu

python awr/online.py \
  --advantage oracle_delta --rounds 30 --initial_episodes 1000 --episodes_per_round 100 \
  --checkpoint flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl

python awr/online.py \
  --advantage oracle_binned --num_bins 5 --rounds 30 --initial_episodes 1000 \
  --episodes_per_round 100 \
  --checkpoint flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default='flow_bc/checkpoints/cube_single_gcbc/best.pkl')
    p.add_argument(
        '--advantage',
        choices=['iql', 'classifier', 'oracle_delta', 'oracle_binned'],
        default='iql',
        help=(
            'iql: learn V/Q; classifier: binary progress net; '
            'oracle_delta: A=d(s)-d(s\'); oracle_binned: signed Δ bins'
        ),
    )
    p.add_argument(
        '--num_bins',
        type=int,
        default=5,
        help='(--advantage oracle_binned) odd # of equal pieces of [-H,H]',
    )
    p.add_argument('--env_name', default='cube-single-v0')
    p.add_argument('--task_id', type=int, default=1)
    p.add_argument('--rounds', type=int, default=30, help='Number of refinement rounds after the initial GCBC collect')
    p.add_argument('--initial_episodes', type=int, default=1000, help='Episodes collected with GCBC before round 1')
    p.add_argument('--episodes_per_round', type=int, default=100, help='Episodes collected each refinement round')
    p.add_argument('--num_workers', type=int, default=10)
    p.add_argument('--n_flow_steps', type=int, default=10)
    p.add_argument('--train_steps', type=int, default=2000, help='Advantage fit steps (IQL joint / classifier) each round')
    p.add_argument('--log_interval', type=int, default=500, help='IQL wandb log interval within a fit')
    p.add_argument('--awr_epochs', type=int, default=10, help='AWR policy extraction epochs after each advantage fit')
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

    if args.advantage == 'oracle_binned' and (args.num_bins < 1 or args.num_bins % 2 == 0):
        p.error(f'--num_bins must be odd and >= 1, got {args.num_bins}')

    import wandb

    from awr.collect import parallel_collect
    from awr.policy import extract_awr, make_classifier_advantage_fn, make_iql_advantage_fn

    wandb.init(project=args.wandb_project, name=args.wandb_name, mode=args.wandb_mode, config=vars(args))

    actor_ckpt = None
    rollouts: list[dict] = []
    annotated: list[dict] = []
    chunk_size = None
    use_oracle_adv = args.advantage in ('oracle_delta', 'oracle_binned')
    oracle_mode = 'delta' if args.advantage == 'oracle_delta' else 'binned'

    # t=0: GCBC collect; t=1..rounds: collect with extracted policy.
    n_iters = args.rounds + 1
    for t in range(n_iters):
        n_eps = args.initial_episodes if t == 0 else args.episodes_per_round
        data = parallel_collect(
            args.checkpoint, args.env_name, args.task_id, n_eps, args.num_workers,
            args.n_flow_steps, actor_ckpt=actor_ckpt,
        )
        rollouts.append(data)
        success_rate = data['success_rate']
        chunk_size = int(data['chunk_size'])

        log = {
            'round': t,
            'collect/mode': 1 if actor_ckpt is not None else 0,
            'collect/success_rate': success_rate,
            'collect/num_episodes': n_eps,
            'collect/num_transitions': len(data['actions']),
            'train/num_datasets': len(rollouts),
        }

        if args.advantage == 'classifier' or use_oracle_adv:
            from awr.annotate import annotate_rollouts
            from awr.classifier.dataset import build_oracle_advantages, merge_annotated

            ann, mean_d = annotate_rollouts(
                data, num_workers=args.num_workers, max_oracle_steps=args.max_oracle_steps,
            )
            annotated.append(ann)
            merged = merge_annotated(annotated)
            log['collect/mean_oracle_distance'] = mean_d

            if use_oracle_adv:
                observations, actions, advantages, adv_stats = build_oracle_advantages(
                    merged, mode=oracle_mode, num_bins=args.num_bins,
                )
                if len(advantages) == 0:
                    raise RuntimeError(f'round {t}: no oracle-labeled chunks for AWR')
                actor_ckpt = extract_awr(
                    observations,
                    actions,
                    advantages=advantages,
                    epochs=args.awr_epochs,
                    batch_size=args.batch_size,
                    alpha=args.alpha,
                    lr=args.lr,
                    seed=args.seed + t,
                    chunk_size=chunk_size,
                )
                log.update({f'label/{k}': v for k, v in adv_stats.items() if v is not None})
                log.update({f'awr/{k}': v for k, v in actor_ckpt['metrics'].items()})
            else:
                from awr.classifier.train import train_classifier

                clf = train_classifier(
                    merged, args.train_steps, args.batch_size, args.lr, args.hidden, args.seed + t,
                )
                observations = np.asarray(merged['observations'], np.float32)
                actions = np.asarray(merged['action_chunks'], np.float32).reshape(len(observations), -1)
                adv_fn = make_classifier_advantage_fn(clf)
                actor_ckpt = extract_awr(
                    observations, actions, adv_fn, epochs=args.awr_epochs, batch_size=args.batch_size,
                    alpha=args.alpha, lr=args.lr, seed=args.seed + t, chunk_size=chunk_size,
                )
                log['train/classifier_loss'] = clf['loss']
                log.update({f'awr/{k}': v for k, v in actor_ckpt['metrics'].items()})
        else:
            from awr.iql.dataset import IQLDataset
            from awr.iql.train import train_iql

            # Fit IQL advantage (V/Q; joint actor is trained but we re-extract AWR for deploy).
            agent, fit_metrics = train_iql(
                args.train_steps, rollouts=rollouts, seed=args.seed + t, batch_size=args.batch_size,
                expectile=args.expectile, alpha=args.alpha, lr=args.lr,
                discount=args.discount, tau=args.tau,
                log_interval=args.log_interval, wandb_run=None,
            )
            ds = IQLDataset.from_rollouts(rollouts)
            adv_fn = make_iql_advantage_fn(agent)
            actor_ckpt = extract_awr(
                ds.data['observations'], ds.data['actions'], adv_fn, epochs=args.awr_epochs,
                batch_size=args.batch_size, alpha=args.alpha, lr=args.lr, seed=args.seed + t,
                chunk_size=chunk_size,
            )
            log.update({f'train/{k}': v for k, v in fit_metrics.items()})
            log.update({f'awr/{k}': v for k, v in actor_ckpt['metrics'].items()})

        wandb.log(log)
        print(
            f'round {t} [{args.advantage}]: episodes={n_eps} success_rate={success_rate:.3f} '
            f'policy={"gcbc" if t == 0 and log["collect/mode"] == 0 else "awr"}',
            flush=True,
        )

    wandb.finish()


if __name__ == '__main__':
    main()
