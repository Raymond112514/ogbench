"""Shared success/failure warm-start, then fork finetune labels for BoN.

Experiment:
  Round 0 (warm-start, shared):
    GCBC collect → train classifier with episode success vs failure labels → C0.
    Warm-start data is discarded for later finetuning.

  Round 1 (optional shared collect from C0 for fairness):
    BoN(C0) collect once; both forks reuse these rollouts.

  Rounds 1..R (per fork):
    Label with either:
      (1) success vs failure  (--finetune_label success)
      (2) oracle improvement  (--finetune_label oracle)
    Finetune from C0 / previous fork ckpt on *online* data only (no round-0 data).
    Collect with BoN under that fork's classifier.

This isolates whether a finer-grained oracle-progress objective beats continuing
with coarse success/failure after the same initialization.

Usage:
  # Run both forks in one job (warmstart once, shared round-1 collect):
  python bon_sampling/online/online_fork_labels.py \\
    --checkpoint flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl \\
    --finetune_label both --rounds 10 --episodes_per_round 100 --tau 5

  # Or warmstart once, then launch forks separately with the same C0:
  python ... --phase warmstart --warmstart_out /tmp/c0.pkl
  python ... --phase finetune --finetune_label success --warmstart_ckpt /tmp/c0.pkl
  python ... --phase finetune --finetune_label oracle  --warmstart_ckpt /tmp/c0.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')


def _load_clf_params(ckpt_path: str):
    with open(ckpt_path, 'rb') as f:
        return pickle.load(f)


def _train_classifier(
    data_path: str,
    ckpt_path: str,
    steps: int,
    batch_size: int,
    lr: float,
    hidden: int,
    val_ratio: float,
    seed: int,
    eval_interval: int,
    *,
    label_mode: str,
    tau: int | None = None,
    init_ckpt: str | None = None,
) -> tuple[str, dict]:
    """Train/finetune AdvantageClassifier; optionally warm-start from init_ckpt."""
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state
    from tqdm import trange

    from bon_sampling.advantage.model import AdvantageClassifier
    from bon_sampling.advantage.train import eval_dataset, sample_batch, train_step

    if label_mode == 'success':
        from bon_sampling.advantage.train_success import make_train_val_success

        train_data, val_data, train_succ, val_succ = make_train_val_success(
            str(data_path), val_ratio, seed
        )
        if len(train_data.pos_sel) == 0 or len(train_data.neg_sel) == 0:
            raise ValueError(
                f'success train needs both classes; '
                f'succ={int(train_succ.sum())} fail={int((~train_succ).sum())}'
            )

        def sample_fn(rng):
            return train_data.sample_balanced_batch(batch_size, rng)

        extra = {
            'train_succ_eps': float(train_succ.sum()),
            'train_fail_eps': float((~train_succ).sum()),
            'val_succ_eps': float(val_succ.sum()),
            'val_fail_eps': float((~val_succ).sum()),
        }
        desc = 'finetune success-clf'
    else:
        from bon_sampling.advantage.dataset import make_train_val

        train_data, val_data = make_train_val(str(data_path), val_ratio, seed, tau=tau)

        def sample_fn(rng):
            return sample_batch(train_data, batch_size, rng)

        extra = {}
        desc = 'finetune oracle-clf'

    chunk_size = train_data.chunk_size if train_data.chunk_mode else 1
    obs_dim = train_data.observations.shape[1]
    act_dim = train_data.act_dim
    model = AdvantageClassifier(hidden=hidden)

    if init_ckpt is not None:
        ckpt = _load_clf_params(init_ckpt)
        params = ckpt['params']
        hidden = int(ckpt.get('hidden', hidden))
        chunk_size = int(ckpt.get('chunk_size', chunk_size))
    else:
        key = jax.random.PRNGKey(seed)
        params = model.init(key, jnp.zeros((1, obs_dim)), jnp.zeros((1, act_dim)))

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(lr))
    rng = np.random.default_rng(seed)
    best_val_acc, best_val_loss = -1.0, 0.0
    best_params = state.params
    best_step = 0
    for step in trange(1, steps + 1, desc=desc):
        batch = sample_fn(rng)
        state, _ = train_step(state, {k: jnp.asarray(v) for k, v in batch.items()})
        if step % eval_interval == 0 or step == steps:
            val_m = eval_dataset(state, val_data, batch_size, rng)
            if val_m.get('accuracy', 0.0) > best_val_acc:
                best_val_acc, best_val_loss = val_m['accuracy'], val_m['loss']
                best_params, best_step = state.params, step

    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(ckpt_path, 'wb') as f:
        pickle.dump(
            {
                'mode': 'classifier',
                'step': best_step,
                'params': best_params,
                'obs_dim': obs_dim,
                'act_dim': act_dim,
                'chunk_size': chunk_size,
                'hidden': hidden,
                'label': 'episode_success' if label_mode == 'success' else 'oracle_progress',
            },
            f,
        )
    return str(ckpt_path), {
        'val_acc': best_val_acc,
        'val_loss': best_val_loss,
        **extra,
    }


def run_warmstart(args, tmp_dir: Path) -> tuple[str, dict]:
    from bon_sampling.online.online_success_bon import collect_round, merge_success_data, prepare_chunk_labeled

    round_dir = tmp_dir / 'warmstart'
    round_dir.mkdir(parents=True, exist_ok=True)
    raw_path = round_dir / 'rollouts.npz'
    labeled_path = round_dir / 'labeled.npz'
    ckpt_path = Path(args.warmstart_out) if args.warmstart_out else (tmp_dir / 'warmstart.pkl')

    success_rate, num_transitions, chunk_size, n_succ, n_fail = collect_round(
        args.checkpoint, args.env_name, args.task_id, args.episodes_per_round, args.num_workers,
        args.n_flow_steps, None, args.bon_n, raw_path,
    )
    label_stats = prepare_chunk_labeled(raw_path, labeled_path)
    merge_success_data([str(labeled_path)], str(round_dir / 'merged.npz'))
    ckpt, metrics = _train_classifier(
        str(round_dir / 'merged.npz'),
        str(ckpt_path),
        args.train_steps,
        args.batch_size,
        args.lr,
        args.hidden,
        args.val_ratio,
        args.seed,
        args.eval_interval,
        label_mode='success',
        init_ckpt=None,
    )
    info = {
        'collect/success_rate': success_rate,
        'collect/num_transitions': num_transitions,
        'collect/num_success_eps': float(n_succ),
        'collect/num_fail_eps': float(n_fail),
        'label/pos_rate': label_stats['pos_rate'],
        'chunk_size': chunk_size,
        **{f'train/{k}': v for k, v in metrics.items()},
    }
    print(
        f'warmstart: success={success_rate:.3f} succ_eps={n_succ} fail_eps={n_fail} '
        f'val_acc={metrics["val_acc"]:.3f} -> {ckpt}',
        flush=True,
    )
    return ckpt, info


def _label_success(raw_path: Path, labeled_path: Path) -> dict:
    from bon_sampling.online.online_success_bon import prepare_chunk_labeled

    return prepare_chunk_labeled(raw_path, labeled_path)


def _label_oracle(raw_path: Path, annotated_path: Path, args) -> dict:
    from bon_sampling.online.online_bon import annotate_round

    mean_d = annotate_round(
        raw_path, annotated_path, args.num_workers, args.max_oracle_steps, args.warmup_steps,
    )
    return {'mean_oracle_distance': mean_d}


def run_fork(args, warmstart_ckpt: str, finetune_label: str, tmp_dir: Path, shared_r1_raw: Path | None):
    import wandb

    from bon_sampling.advantage.dataset import merge_annotated
    from bon_sampling.online.online_success_bon import collect_round, merge_success_data

    fork_dir = tmp_dir / f'fork_{finetune_label}'
    fork_dir.mkdir(parents=True, exist_ok=True)
    reranker_path = fork_dir / 'reranker.pkl'
    shutil.copy(warmstart_ckpt, reranker_path)
    reranker_ckpt = str(reranker_path)

    # Online-only buffers (warmstart / round-0 data intentionally excluded).
    train_paths: list[Path] = []
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f'fork_{finetune_label}_task{args.task_id}_seed{args.seed}',
        mode=args.wandb_mode,
        config={**vars(args), 'finetune_label': finetune_label, 'warmstart_ckpt': warmstart_ckpt},
        reinit=True,
    )
    wandb.log({'round': 0, 'phase': 'warmstart_ref', 'fork': finetune_label})

    # Finetune rounds are 1..rounds (rounds counts finetune rounds after warmstart).
    for r in range(1, args.rounds + 1):
        round_dir = fork_dir / f'round{r}'
        round_dir.mkdir(parents=True, exist_ok=True)
        raw_path = round_dir / 'rollouts.npz'
        labeled_path = round_dir / ('labeled.npz' if finetune_label == 'success' else 'annotated.npz')

        if r == 1 and shared_r1_raw is not None and shared_r1_raw.exists():
            shutil.copy(shared_r1_raw, raw_path)
            raw = np.load(raw_path, allow_pickle=False)
            ends = np.asarray(raw['episode_ends'], np.int32)
            if 'episode_success' in raw:
                ep_succ = np.asarray(raw['episode_success']).astype(bool)
            else:
                ep_succ = np.asarray(raw['successes'])[ends - 1].astype(bool)
            success_rate = float(ep_succ.mean()) if len(ep_succ) else 0.0
            num_transitions = len(raw['actions'])
            collect_mode = 2  # shared BoN collect from C0
            print(f'[{finetune_label}] round {r}: reuse shared C0 BoN collect', flush=True)
        else:
            success_rate, num_transitions, _chunk, n_succ, n_fail = collect_round(
                args.checkpoint, args.env_name, args.task_id, args.episodes_per_round,
                args.num_workers, args.n_flow_steps, reranker_ckpt, args.bon_n, raw_path,
            )
            collect_mode = 1
            print(
                f'[{finetune_label}] round {r}: collect success={success_rate:.3f} '
                f'(succ_eps={n_succ} fail_eps={n_fail})',
                flush=True,
            )

        log = {
            'round': r,
            'fork': finetune_label,
            'phase': 'finetune',
            'collect/mode': collect_mode,
            'collect/success_rate': success_rate,
            'collect/num_transitions': num_transitions,
            'label/mode': finetune_label,
        }

        if finetune_label == 'success':
            stats = _label_success(raw_path, labeled_path)
            train_paths.append(labeled_path)
            merged = fork_dir / 'online_all.npz'
            merge_success_data([str(p) for p in train_paths], str(merged))
            log['label/pos_rate'] = stats['pos_rate']
            ep = np.asarray(np.load(merged)['episode_success']).astype(bool)
            if ep.sum() == 0 or (~ep).sum() == 0:
                log['train/skipped'] = 1
                wandb.log(log)
                print(f'[{finetune_label}] round {r}: skip train (need both classes)', flush=True)
                continue
            ckpt, metrics = _train_classifier(
                str(merged), str(reranker_path), args.finetune_steps, args.batch_size, args.lr,
                args.hidden, args.val_ratio, args.seed + r, args.eval_interval,
                label_mode='success', init_ckpt=reranker_ckpt,
            )
        else:
            stats = _label_oracle(raw_path, labeled_path, args)
            train_paths.append(labeled_path)
            merged = fork_dir / 'online_all.npz'
            merge_annotated([str(p) for p in train_paths], str(merged))
            log['collect/mean_oracle_distance'] = stats['mean_oracle_distance']
            log['label/tau'] = args.tau if args.tau is not None else -1
            ckpt, metrics = _train_classifier(
                str(merged), str(reranker_path), args.finetune_steps, args.batch_size, args.lr,
                args.hidden, args.val_ratio, args.seed + r, args.eval_interval,
                label_mode='oracle', tau=args.tau, init_ckpt=reranker_ckpt,
            )

        reranker_ckpt = ckpt
        log['train/skipped'] = 0
        log['train/num_online_datasets'] = len(train_paths)
        log.update({f'train/{k}': v for k, v in metrics.items()})
        wandb.log(log)
        print(
            f'[{finetune_label}] round {r}: success={success_rate:.3f} '
            f'val_acc={metrics.get("val_acc", float("nan")):.3f}',
            flush=True,
        )

    run.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default='flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl')
    p.add_argument('--phase', choices=['all', 'warmstart', 'finetune'], default='all')
    p.add_argument(
        '--finetune_label',
        choices=['success', 'oracle', 'both'],
        default='both',
        help='Finetune objective after shared success/failure warm-start',
    )
    p.add_argument('--warmstart_ckpt', default=None, help='Existing C0 for --phase finetune')
    p.add_argument('--warmstart_out', default=None, help='Where to write C0 in warmstart/all')
    p.add_argument('--env_name', default='cube-single-v0')
    p.add_argument('--task_id', type=int, default=1)
    p.add_argument('--n_flow_steps', type=int, default=10)
    p.add_argument(
        '--rounds',
        type=int,
        default=10,
        help='Number of *finetune* rounds after warm-start (not counting round 0)',
    )
    p.add_argument('--episodes_per_round', type=int, default=100)
    p.add_argument('--num_workers', type=int, default=10)
    p.add_argument('--bon_n', type=int, default=8)
    p.add_argument('--train_steps', type=int, default=5000, help='Warm-start train steps')
    p.add_argument('--finetune_steps', type=int, default=2000, help='Per-round finetune steps')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--eval_interval', type=int, default=500)
    p.add_argument('--tau', type=int, default=5, help='Oracle progress slack (finetune_label=oracle)')
    p.add_argument('--max_oracle_steps', type=int, default=200)
    p.add_argument('--warmup_steps', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=['cpu', 'auto'], default='cpu')
    p.add_argument('--share_round1_collect', action='store_true', default=True,
                   help='Both forks reuse the same BoN(C0) round-1 rollouts (default on)')
    p.add_argument('--no_share_round1_collect', action='store_false', dest='share_round1_collect')
    p.add_argument('--wandb_project', default='bon-fork-labels')
    p.add_argument('--wandb_name', default=None)
    p.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
    args = p.parse_args()

    if args.device == 'cpu':
        os.environ['JAX_PLATFORMS'] = 'cpu'
    if args.phase == 'finetune' and not args.warmstart_ckpt:
        p.error('--phase finetune requires --warmstart_ckpt')
    if args.phase == 'finetune' and args.finetune_label == 'both':
        p.error('--phase finetune needs --finetune_label success|oracle (use --phase all for both)')

    import wandb

    from bon_sampling.online.online_success_bon import collect_round

    tmp_dir = Path(tempfile.mkdtemp(prefix='bon_fork_'))
    try:
        warmstart_ckpt = args.warmstart_ckpt
        if args.phase in ('all', 'warmstart'):
            ws_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or f'warmstart_task{args.task_id}_seed{args.seed}',
                mode=args.wandb_mode,
                config=vars(args),
                reinit=True,
            )
            warmstart_ckpt, info = run_warmstart(args, tmp_dir)
            if args.warmstart_out:
                Path(args.warmstart_out).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(warmstart_ckpt, args.warmstart_out)
                warmstart_ckpt = str(Path(args.warmstart_out).resolve())
            wandb.log({'round': 0, 'phase': 'warmstart', **info})
            ws_run.finish()
            if args.phase == 'warmstart':
                print(f'Wrote warmstart ckpt: {warmstart_ckpt}')
                return

        assert warmstart_ckpt is not None

        shared_r1 = None
        if args.share_round1_collect and args.phase == 'all' and args.finetune_label == 'both':
            shared_r1 = tmp_dir / 'shared_round1_rollouts.npz'
            sr, n_tr, _, n_s, n_f = collect_round(
                args.checkpoint, args.env_name, args.task_id, args.episodes_per_round,
                args.num_workers, args.n_flow_steps, warmstart_ckpt, args.bon_n, shared_r1,
            )
            print(
                f'shared round-1 BoN(C0) collect: success={sr:.3f} '
                f'succ_eps={n_s} fail_eps={n_f} transitions={n_tr}',
                flush=True,
            )

        labels = ['success', 'oracle'] if args.finetune_label == 'both' else [args.finetune_label]
        for lab in labels:
            run_fork(args, warmstart_ckpt, lab, tmp_dir, shared_r1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
