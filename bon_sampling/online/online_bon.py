"""Online BoN loop: collect -> fit reranker, logging to wandb.

--method classifier: oracle-label, then fit classifier on all annotated data so far.
--method iql: no oracle; fit FQL IQL on all rollouts so far using env success (-1/0).

Round 0 always collects with the plain GCBC policy. Later rounds use BoN with the previous
round's reranker. Data collection always runs on CPU (--num_workers parallel envs).

No persistent checkpoints or rollout dumps — only temp files for BoN workers / labeling.
Each round also logs classifier accuracy / precision / recall on ~20 fresh oracle-labeled
episodes (held-out from training).

python bon_sampling/online/online_bon.py \\
  --checkpoint flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl \\
  --method classifier --rounds 30 --episodes_per_round 100 --num_workers 10 --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
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

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        observations=np.asarray(obs, np.float32),
        actions=np.asarray(act, np.float32),
        next_observations=np.asarray(next_obs, np.float32),
        next_mjstate=np.asarray(state, np.float64),
        successes=np.asarray(step_succ, np.bool_),
        episode_initial_mjstate=np.asarray(init_state, np.float64),
        goal_xyz=goal_xyz,
        task_id=np.array(task_id),
        episode_ends=np.asarray(ends, np.int32),
        chunk_size=np.array(meta['chunk_size']),
        policy='bon' if reranker_ckpt else 'flow_bc',
    )
    return float(np.mean(successes)), len(act), int(meta['chunk_size'])


def annotate_round(raw_path, annotated_path, num_workers, max_oracle_steps, warmup_steps):
    from bon_sampling.annotate_oracle_distance import annotate_indices, chunk_boundary_indices, subsample_arrays

    data = dict(np.load(raw_path, allow_pickle=False))
    chunk_size = int(data.get('chunk_size', 1))
    task_id = int(data.get('task_id', 1))
    indices = chunk_boundary_indices(data['episode_ends'], chunk_size)

    dist_map = {}
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(annotate_indices, wid, str(raw_path), chunk, task_id, 0, max_oracle_steps, warmup_steps, 0)
            for wid, chunk in enumerate(np.array_split(indices, num_workers))
            if len(chunk) > 0
        ]
        for fut in as_completed(futures):
            idx_out, dist = fut.result()
            dist_map.update(zip(idx_out.tolist(), dist.tolist()))
    distance = np.asarray([dist_map[int(t)] for t in indices], np.int32)

    out = subsample_arrays(data, indices, chunk_size)
    out['distance'] = distance
    Path(annotated_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(annotated_path, **out)
    return float(distance.mean())


def train_classifier(data_path, ckpt_path, steps, batch_size, lr, hidden, val_ratio, seed, eval_interval, tau=None):
    """Fit classifier; write a temp pickle for BoN workers only (not archived)."""
    import pickle

    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state
    from tqdm import trange

    from bon_sampling.advantage.dataset import make_train_val
    from bon_sampling.advantage.model import AdvantageClassifier
    from bon_sampling.advantage.train import eval_dataset, sample_batch, train_step

    train_data, val_data = make_train_val(str(data_path), val_ratio, seed, tau=tau)
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
    for step in trange(1, steps + 1, desc='train classifier'):
        batch = sample_batch(train_data, batch_size, rng)
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
            },
            f,
        )
    return str(ckpt_path), {'val_acc': best_val_acc, 'val_loss': best_val_loss}


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
    max_oracle_steps: int,
    warmup_steps: int,
    batch_size: int,
    tmp_dir: Path,
    tau: int | None = None,
) -> dict[str, float]:
    """Collect fresh GCBC episodes, oracle-label, score classifier accuracy/precision/recall."""
    import pickle

    import jax.numpy as jnp

    from bon_sampling.advantage.dataset import AdvantageDataset
    from bon_sampling.advantage.model import AdvantageClassifier

    raw_path = tmp_dir / 'eval_fresh_raw.npz'
    ann_path = tmp_dir / 'eval_fresh_annotated.npz'

    collect_round(
        policy_ckpt, env_name, task_id, n_episodes, num_workers, n_flow_steps,
        None, bon_n, raw_path,  # plain GCBC: held-out (s,a), not BoN-selected
    )
    annotate_round(raw_path, ann_path, num_workers, max_oracle_steps, warmup_steps)

    data = AdvantageDataset(str(ann_path), episode_ids=None, task='classifier', tau=tau)
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

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(labels, axis=0)
    return _prf(y_true, y_pred)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default='flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl')
    p.add_argument('--method', choices=['classifier', 'iql'], default='classifier')
    p.add_argument('--env_name', default='cube-single-v0')
    p.add_argument('--task_id', type=int, default=1)
    p.add_argument('--n_flow_steps', type=int, default=10)
    p.add_argument('--rounds', type=int, default=30)
    p.add_argument('--episodes_per_round', type=int, default=100)
    p.add_argument('--eval_clf_episodes', type=int, default=20,
                   help='Fresh episodes for held-out classifier metrics each round')
    p.add_argument('--num_workers', type=int, default=10)
    p.add_argument('--bon_n', type=int, default=8)
    p.add_argument('--max_oracle_steps', type=int, default=200)
    p.add_argument('--warmup_steps', type=int, default=2)
    p.add_argument('--tau', type=int, default=None,
                   help='Progress slack: y=1 iff d(s)-d(s\') >= H-tau. Default tau=H-1 (threshold 1)')
    p.add_argument('--train_steps', type=int, default=5000)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--eval_interval', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--expectile', type=float, default=0.9, help='IQL expectile (ignored for classifier)')
    p.add_argument('--device', choices=['cpu', 'auto'], default='cpu',
                   help='JAX backend for training; data collection is always CPU')
    p.add_argument('--wandb_project', default='bon-online')
    p.add_argument('--wandb_name', default=None, help='Wandb run name')
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
    tmp_dir = Path(tempfile.mkdtemp(prefix='bon_online_'))
    reranker_path = tmp_dir / 'reranker.pkl'

    try:
        for r in range(args.rounds):
            round_dir = tmp_dir / f'round{r}'
            round_dir.mkdir(parents=True, exist_ok=True)
            raw_path = round_dir / 'rollouts.npz'
            annotated_path = round_dir / 'annotated.npz'

            success_rate, num_transitions, chunk_size = collect_round(
                args.checkpoint, args.env_name, args.task_id, args.episodes_per_round, args.num_workers,
                args.n_flow_steps, reranker_ckpt, args.bon_n, raw_path,
            )

            log = {
                'round': r,
                'method': args.method,
                'collect/mode': 1 if reranker_ckpt else 0,
                'collect/success_rate': success_rate,
                'collect/num_transitions': num_transitions,
            }

            if args.method == 'classifier':
                mean_distance = annotate_round(
                    raw_path, annotated_path, args.num_workers, args.max_oracle_steps, args.warmup_steps
                )
                train_paths.append(annotated_path)
                from bon_sampling.advantage.dataset import merge_annotated

                merged = tmp_dir / 'annotated_all.npz'
                merge_annotated([str(p) for p in train_paths], str(merged))
                ckpt_path, metrics = train_classifier(
                    merged, reranker_path, args.train_steps, args.batch_size, args.lr, args.hidden,
                    args.val_ratio, args.seed, args.eval_interval, tau=args.tau,
                )
                log['collect/mean_oracle_distance'] = mean_distance
                log['label/tau'] = chunk_size - 1 if args.tau is None else int(args.tau)
                log['label/threshold'] = chunk_size - log['label/tau']

                clf_m = eval_classifier_fresh(
                    args.checkpoint, args.env_name, args.task_id, args.eval_clf_episodes,
                    args.num_workers, args.n_flow_steps, args.bon_n, ckpt_path,
                    args.max_oracle_steps, args.warmup_steps, args.batch_size, tmp_dir,
                    tau=args.tau,
                )
                log.update({f'eval_clf/{k}': v for k, v in clf_m.items()})
                print(
                    f'round {r} [classifier]: success={success_rate:.3f} '
                    f'clf_acc={clf_m["accuracy"]:.3f} prec={clf_m["precision"]:.3f} '
                    f'recall={clf_m["recall"]:.3f} (n={int(clf_m["num_samples"])})',
                    flush=True,
                )
            else:
                train_paths.append(raw_path)
                from bon_sampling.iql.train import train_iql

                ckpt_path, metrics = train_iql(
                    train_paths, reranker_path, args.train_steps, seed=args.seed,
                    batch_size=args.batch_size, chunk_size=chunk_size, expectile=args.expectile,
                )
                print(f'round {r} [iql]: success_rate={success_rate:.3f}', flush=True)

            log['train/num_datasets'] = len(train_paths)
            log.update({f'train/{k}': v for k, v in metrics.items()})
            wandb.log(log)
            reranker_ckpt = ckpt_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    wandb.finish()


if __name__ == '__main__':
    main()
