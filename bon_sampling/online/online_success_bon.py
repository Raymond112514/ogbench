"""Online BoN with a success/failure episode classifier (no oracle).

Round k:
  1. Collect episodes (round 0: GCBC; later: BoN with previous classifier).
  2. Label every chunk by whether its episode succeeded (y=1) or failed (y=0).
  3. Fit classifier on all data so far with 1/2–1/2 success/failure batches.
  4. Log collect success + held-out classifier metrics on fresh GCBC episodes.

No persistent checkpoints — wandb only (temp files cleaned up).

Usage (from ogbench/):
  python bon_sampling/online/online_success_bon.py \\
    --checkpoint flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl \\
    --task_id 1 --rounds 30 --episodes_per_round 100
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')


def collect_round(checkpoint, env_name, task_id, n_episodes, num_workers, n_flow_steps, reranker_ckpt, bon_n, out_path):
    import gymnasium

    import ogbench.manipspace  # noqa: F401
    import bon_sampling.collect_transitions as ct
    from flow_bc.checkpoint import read_ckpt_meta

    ct.NUM_VIDEO_EPISODES = 0
    meta = read_ckpt_meta(checkpoint)
    tmp_env = gymnasium.make(env_name)
    max_steps = tmp_env.spec.max_episode_steps
    goal_xyz = tmp_env.unwrapped.task_infos[task_id - 1]['goal_xyzs'][0].copy()
    tmp_env.close()

    obs, act, next_obs, state, step_succ, ends, successes, _, init_state = ct.parallel_collect(
        checkpoint, env_name, task_id, n_episodes, num_workers, max_steps, n_flow_steps,
        None, 'cpu', [], reranker_ckpt, 'auto', bon_n,
    )

    ends = np.asarray(ends, np.int32)
    step_succ = np.asarray(step_succ, np.bool_)
    episode_success = np.asarray(successes, np.bool_)
    # Prefer explicit episode outcomes from the collector; fall back to final-step flags.
    if len(episode_success) != len(ends):
        episode_success = step_succ[ends - 1]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        observations=np.asarray(obs, np.float32),
        actions=np.asarray(act, np.float32),
        next_observations=np.asarray(next_obs, np.float32),
        next_mjstate=np.asarray(state, np.float64),
        successes=step_succ,
        episode_success=episode_success,
        episode_initial_mjstate=np.asarray(init_state, np.float64),
        goal_xyz=goal_xyz,
        task_id=np.array(task_id),
        episode_ends=ends,
        chunk_size=np.array(meta['chunk_size']),
        policy='bon' if reranker_ckpt else 'flow_bc',
    )
    return float(np.mean(episode_success)), len(act), int(meta['chunk_size']), int(episode_success.sum()), int((~episode_success).sum())


def prepare_chunk_labeled(raw_path: Path, out_path: Path) -> dict[str, float]:
    """Subsample to chunk boundaries; label rows by episode success (for BoN chunk scoring)."""
    from bon_sampling.annotate_oracle_distance import chunk_boundary_indices, subsample_arrays

    data = dict(np.load(raw_path, allow_pickle=False))
    chunk_size = int(data.get('chunk_size', 1))
    ends = np.asarray(data['episode_ends'], np.int32)
    if 'episode_success' in data:
        ep_succ = np.asarray(data['episode_success']).astype(bool)
    else:
        ep_succ = np.asarray(data['successes'])[ends - 1].astype(bool)

    indices = chunk_boundary_indices(ends, chunk_size)
    out = subsample_arrays(data, indices, chunk_size)
    new_ends = np.asarray(out['episode_ends'], np.int32)
    row_succ = np.zeros(len(out['observations']), dtype=bool)
    start = 0
    for i, end in enumerate(new_ends.tolist()):
        row_succ[start:end] = bool(ep_succ[i])
        start = end
    out['successes'] = row_succ
    out['episode_success'] = ep_succ.astype(bool)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    return {
        'num_episodes': float(len(ep_succ)),
        'num_success_eps': float(ep_succ.sum()),
        'num_fail_eps': float((~ep_succ).sum()),
        'pos_rate': float(ep_succ.mean()) if len(ep_succ) else 0.0,
    }


def merge_success_data(paths: list[str], out_path: str) -> str:
    datas = [dict(np.load(p, allow_pickle=False)) for p in paths]
    keys = [
        'observations', 'actions', 'next_observations', 'next_mjstate',
        'successes', 'action_chunks', 'chunk_masks', 'chunk_boundary_indices',
    ]
    out = {}
    for k in keys:
        if k in datas[0]:
            out[k] = np.concatenate([d[k] for d in datas], axis=0)

    ends, ep_succ = [], []
    offset = 0
    for d in datas:
        for e in d['episode_ends']:
            ends.append(int(e) + offset)
        offset = ends[-1]
        if 'episode_success' in d:
            ep_succ.append(np.asarray(d['episode_success']).astype(bool))
        else:
            ep = np.asarray(d['episode_ends'], np.int32)
            ep_succ.append(np.asarray(d['successes'])[ep - 1].astype(bool))
    out['episode_ends'] = np.asarray(ends, np.int32)
    out['episode_success'] = np.concatenate(ep_succ, axis=0)

    for k in ('goal_xyz', 'task_id', 'chunk_size', 'policy'):
        if k in datas[0]:
            out[k] = datas[0][k]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    return out_path


def train_success_classifier(data_path, ckpt_path, steps, batch_size, lr, hidden, val_ratio, seed, eval_interval):
    """Fit success/failure classifier with balanced batches; write temp pickle for BoN."""
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state
    from tqdm import trange

    from bon_sampling.advantage.model import AdvantageClassifier
    from bon_sampling.advantage.train import eval_dataset, train_step
    from bon_sampling.advantage.train_success import make_train_val_success

    train_data, val_data, train_succ, val_succ = make_train_val_success(str(data_path), val_ratio, seed)
    if len(train_data.pos_sel) == 0 or len(train_data.neg_sel) == 0:
        raise ValueError(
            f'train split needs both classes; '
            f'succ_eps={int(train_succ.sum())} fail_eps={int((~train_succ).sum())}'
        )

    chunk_size = train_data.chunk_size if train_data.chunk_mode else 1
    obs_dim = train_data.observations.shape[1]
    act_dim = train_data.act_dim

    model = AdvantageClassifier(hidden=hidden)
    key = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.zeros((1, obs_dim)), jnp.zeros((1, act_dim)))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(lr))

    rng = np.random.default_rng(seed)
    best_val_acc, best_val_loss = -1.0, 0.0
    best_params = state.params
    best_step = 0
    for step in trange(1, steps + 1, desc='train success-clf'):
        batch = train_data.sample_balanced_batch(batch_size, rng)
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
                'label': 'episode_success',
            },
            f,
        )
    return str(ckpt_path), {
        'val_acc': best_val_acc,
        'val_loss': best_val_loss,
        'train_succ_eps': float(train_succ.sum()),
        'train_fail_eps': float((~train_succ).sum()),
        'val_succ_eps': float(val_succ.sum()),
        'val_fail_eps': float((~val_succ).sum()),
    }


def _prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(np.float32).reshape(-1)
    y_pred = np.asarray(y_pred).astype(np.float32).reshape(-1)
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))
    acc = float(np.mean(y_pred == y_true)) if len(y_true) else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'num_samples': float(len(y_true)),
        'label_pos_rate': float(np.mean(y_true)) if len(y_true) else 0.0,
    }


def eval_classifier_fresh(
    policy_ckpt: str,
    env_name: str,
    task_id: int,
    n_episodes: int,
    num_workers: int,
    n_flow_steps: int,
    bon_n: int,
    classifier_ckpt: str,
    batch_size: int,
    tmp_dir: Path,
) -> dict[str, float]:
    """Collect fresh GCBC episodes; score classifier vs episode-success labels."""
    import jax.numpy as jnp

    from bon_sampling.advantage.model import AdvantageClassifier
    from bon_sampling.advantage.train_success import SuccessEpisodeDataset

    raw_path = tmp_dir / 'eval_fresh_raw.npz'
    labeled_path = tmp_dir / 'eval_fresh_labeled.npz'

    collect_round(
        policy_ckpt, env_name, task_id, n_episodes, num_workers, n_flow_steps,
        None, bon_n, raw_path,
    )
    prepare_chunk_labeled(raw_path, labeled_path)

    data = SuccessEpisodeDataset(str(labeled_path), episode_ids=None)
    if len(data) == 0:
        return {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
            'num_samples': 0.0, 'label_pos_rate': 0.0,
        }

    with open(classifier_ckpt, 'rb') as f:
        ckpt = pickle.load(f)
    model = AdvantageClassifier(hidden=ckpt['hidden'])
    params = ckpt['params']

    preds, labels = [], []
    for start in range(0, len(data), batch_size):
        sel = np.arange(start, min(start + batch_size, len(data)))
        batch = data.get_batch(sel)
        logits = model.apply(
            params,
            jnp.asarray(batch['observations']),
            jnp.asarray(batch['actions']),
        )
        pred = (np.asarray(logits) >= 0.0).astype(np.float32)
        preds.append(pred)
        labels.append(batch['labels'])

    return _prf(np.concatenate(labels, axis=0), np.concatenate(preds, axis=0))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default='flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl')
    p.add_argument('--env_name', default='cube-single-v0')
    p.add_argument('--task_id', type=int, default=1)
    p.add_argument('--n_flow_steps', type=int, default=10)
    p.add_argument('--rounds', type=int, default=30)
    p.add_argument('--episodes_per_round', type=int, default=100)
    p.add_argument('--eval_clf_episodes', type=int, default=20)
    p.add_argument('--num_workers', type=int, default=10)
    p.add_argument('--bon_n', type=int, default=8)
    p.add_argument('--train_steps', type=int, default=5000)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--eval_interval', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=['cpu', 'auto'], default='cpu')
    p.add_argument('--wandb_project', default='bon-online-success')
    p.add_argument('--wandb_name', default=None)
    p.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
    args = p.parse_args()

    if args.device == 'cpu':
        os.environ['JAX_PLATFORMS'] = 'cpu'

    import shutil

    import wandb

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config=vars(args),
    )

    reranker_ckpt = None
    train_paths: list[Path] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix='bon_success_'))
    reranker_path = tmp_dir / 'reranker.pkl'

    try:
        for r in range(args.rounds):
            round_dir = tmp_dir / f'round{r}'
            round_dir.mkdir(parents=True, exist_ok=True)
            raw_path = round_dir / 'rollouts.npz'
            labeled_path = round_dir / 'labeled.npz'

            success_rate, num_transitions, chunk_size, n_succ, n_fail = collect_round(
                args.checkpoint, args.env_name, args.task_id, args.episodes_per_round, args.num_workers,
                args.n_flow_steps, reranker_ckpt, args.bon_n, raw_path,
            )
            label_stats = prepare_chunk_labeled(raw_path, labeled_path)
            train_paths.append(labeled_path)

            log = {
                'round': r,
                'collect/mode': 1 if reranker_ckpt else 0,
                'collect/success_rate': success_rate,
                'collect/num_transitions': num_transitions,
                'collect/num_success_eps': float(n_succ),
                'collect/num_fail_eps': float(n_fail),
                'label/pos_rate': label_stats['pos_rate'],
                'train/num_datasets': len(train_paths),
            }

            merged = tmp_dir / 'labeled_all.npz'
            merge_success_data([str(p) for p in train_paths], str(merged))
            merged_ep = np.asarray(np.load(merged)['episode_success']).astype(bool)
            if merged_ep.sum() == 0 or (~merged_ep).sum() == 0:
                print(
                    f'round {r}: skip train (need both success and failure in buffer; '
                    f'succ={int(merged_ep.sum())} fail={int((~merged_ep).sum())})',
                    flush=True,
                )
                log['train/skipped'] = 1
                wandb.log(log)
                continue

            ckpt_path, metrics = train_success_classifier(
                merged, reranker_path, args.train_steps, args.batch_size, args.lr, args.hidden,
                args.val_ratio, args.seed, args.eval_interval,
            )
            log.update({f'train/{k}': v for k, v in metrics.items()})
            log['train/skipped'] = 0

            clf_m = eval_classifier_fresh(
                args.checkpoint, args.env_name, args.task_id, args.eval_clf_episodes,
                args.num_workers, args.n_flow_steps, args.bon_n, ckpt_path,
                args.batch_size, tmp_dir,
            )
            log.update({f'eval_clf/{k}': v for k, v in clf_m.items()})
            print(
                f'round {r}: success={success_rate:.3f} '
                f'clf_acc={clf_m["accuracy"]:.3f} prec={clf_m["precision"]:.3f} '
                f'recall={clf_m["recall"]:.3f} (n={int(clf_m["num_samples"])})',
                flush=True,
            )

            wandb.log(log)
            reranker_ckpt = ckpt_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    wandb.finish()


if __name__ == '__main__':
    main()
