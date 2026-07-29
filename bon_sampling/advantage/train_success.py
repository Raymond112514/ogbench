"""Train a (s,a) classifier for episode success vs failure (no oracle).

Label: y=1 if (s,a) comes from a successful episode, else 0.
Each train batch is mixed 1/2 from success episodes and 1/2 from failure episodes.

Expects an npz from bon_sampling/collect_transitions.py (or compatible) with:
  observations, actions (or action_chunks), episode_ends, successes (per-step)

Usage (from ogbench/):
  python bon_sampling/advantage/train_success.py \\
    --data bon_sampling/data/gcbc/flowbc_rollouts.npz \\
    --checkpoint_dir bon_sampling/advantage/checkpoints_success
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm import trange

from bon_sampling.advantage.dataset import episode_ranges, is_chunk_data, split_episodes
from bon_sampling.advantage.model import AdvantageClassifier
from bon_sampling.advantage.train import eval_dataset, save_checkpoint, train_step


def episode_success_flags(episode_ends: np.ndarray, successes: np.ndarray) -> np.ndarray:
    """Per-episode success from saved arrays.

    Supports:
      - successes length == num_episodes (already episode-level)
      - successes length == num_transitions (per-step; use final step of each episode)
    """
    ends = np.asarray(episode_ends, np.int64)
    succ = np.asarray(successes)
    n_ep = len(ends)
    if len(succ) == n_ep:
        return succ.astype(bool)
    if len(succ) == int(ends[-1]):
        return succ[ends - 1].astype(bool)
    raise ValueError(
        f'cannot derive episode success: len(successes)={len(succ)} '
        f'num_episodes={n_ep} num_transitions={int(ends[-1])}'
    )


def build_success_samples(
    episode_ends: np.ndarray,
    episode_success: np.ndarray,
    chunk_mode: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Index every stored (s,a) / chunk row; label = episode success."""
    del chunk_mode  # every stored row in [start, end) gets the episode label
    indices, labels = [], []
    for ep, (start, end) in enumerate(episode_ranges(episode_ends)):
        y = 1.0 if episode_success[ep] else 0.0
        for t in range(start, end):
            indices.append(t)
            labels.append(y)
    return np.asarray(indices, np.int64), np.asarray(labels, np.float32)


class SuccessEpisodeDataset:
    """(s, a) samples labeled by whether their episode succeeded."""

    def __init__(self, path: str, episode_ids: np.ndarray | None = None):
        self.data = np.load(path, mmap_mode='r')
        self.observations = self.data['observations']
        self.episode_ends = self.data['episode_ends']
        self.chunk_mode = is_chunk_data(self.data)
        self.chunk_size = int(self.data['chunk_size']) if 'chunk_size' in self.data else 1
        if 'episode_success' in self.data:
            self.episode_success = np.asarray(self.data['episode_success']).astype(bool)
        elif 'successes' in self.data:
            self.episode_success = episode_success_flags(self.episode_ends, self.data['successes'])
        else:
            raise ValueError(f'{path} missing successes / episode_success')

        if self.chunk_mode:
            chunks = self.data['action_chunks']
            self.act_dim = chunks.shape[1] * chunks.shape[2]
        else:
            self.act_dim = self.data['actions'].shape[1]

        all_indices, all_labels = build_success_samples(
            self.episode_ends, self.episode_success, self.chunk_mode,
        )

        if episode_ids is not None:
            ranges = episode_ranges(self.episode_ends)
            allowed = set()
            for ep in episode_ids:
                start, end = ranges[int(ep)]
                allowed.update(range(start, end))
            keep = np.array([i in allowed for i in all_indices])
            self.indices = all_indices[keep]
            self.labels = all_labels[keep]
        else:
            self.indices = all_indices
            self.labels = all_labels

        self.pos_sel = np.flatnonzero(self.labels > 0.5)
        self.neg_sel = np.flatnonzero(self.labels <= 0.5)

    def __len__(self) -> int:
        return len(self.indices)

    def get_batch(self, sel: np.ndarray) -> dict[str, np.ndarray]:
        idx = self.indices[sel]
        if self.chunk_mode:
            chunks = np.asarray(self.data['action_chunks'][idx], dtype=np.float32)
            actions = chunks.reshape(len(idx), -1)
        else:
            actions = np.asarray(self.data['actions'][idx], dtype=np.float32)
        return {
            'observations': np.asarray(self.observations[idx], dtype=np.float32),
            'actions': actions,
            'labels': self.labels[sel],
        }

    def sample_balanced_batch(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Half batch from success episodes, half from failure episodes."""
        if len(self.pos_sel) == 0 or len(self.neg_sel) == 0:
            raise ValueError(
                f'need both success and failure samples; '
                f'pos={len(self.pos_sel)} neg={len(self.neg_sel)}'
            )
        n_pos = batch_size // 2
        n_neg = batch_size - n_pos
        pos = rng.choice(self.pos_sel, size=n_pos, replace=True)
        neg = rng.choice(self.neg_sel, size=n_neg, replace=True)
        sel = np.concatenate([pos, neg])
        rng.shuffle(sel)
        return self.get_batch(sel)

    def label_balance(self) -> tuple[float, float]:
        if len(self.labels) == 0:
            return 0.0, 0.0
        return float(np.mean(self.labels == 0)), float(np.mean(self.labels == 1))


def make_train_val_success(path: str, val_ratio: float, seed: int):
    data = np.load(path, allow_pickle=False)
    num_episodes = len(data['episode_ends'])
    train_eps, val_eps = split_episodes(num_episodes, val_ratio, seed)
    train = SuccessEpisodeDataset(path, train_eps)
    val = SuccessEpisodeDataset(path, val_eps)
    train_succ = train.episode_success[train_eps]
    val_succ = val.episode_success[val_eps]
    return train, val, train_succ, val_succ


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, help='npz with observations/actions/episode_ends/successes')
    p.add_argument('--checkpoint_dir', default='bon_sampling/advantage/checkpoints_success')
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log_interval', type=int, default=500)
    p.add_argument('--eval_interval', type=int, default=2500)
    p.add_argument('--save_interval', type=int, default=5000)
    p.add_argument('--wandb_project', default=None, help='If set, log metrics to wandb')
    p.add_argument('--wandb_name', default=None)
    p.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
    args = p.parse_args()

    train_data, val_data, train_succ, val_succ = make_train_val_success(
        args.data, args.val_ratio, args.seed,
    )
    chunk_size = train_data.chunk_size if train_data.chunk_mode else 1
    mode_str = f'chunk (H={chunk_size})' if train_data.chunk_mode else 'per-step'
    print(
        f'mode={mode_str} train={len(train_data)} val={len(val_data)}\n'
        f'  train episodes succ/fail={int(train_succ.sum())}/{int((~train_succ).sum())} '
        f'sample y=0/1={train_data.label_balance()}\n'
        f'  val   episodes succ/fail={int(val_succ.sum())}/{int((~val_succ).sum())} '
        f'sample y=0/1={val_data.label_balance()}'
    )
    if len(train_data.pos_sel) == 0 or len(train_data.neg_sel) == 0:
        raise SystemExit('train split needs both successful and failed episodes')

    wandb = None
    if args.wandb_project:
        import wandb as _wandb

        wandb = _wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config=vars(args),
        )

    obs_dim = train_data.observations.shape[1]
    act_dim = train_data.act_dim
    model = AdvantageClassifier(hidden=args.hidden)
    key = jax.random.PRNGKey(args.seed)
    params = model.init(key, jnp.zeros((1, obs_dim)), jnp.zeros((1, act_dim)))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(args.lr))

    rng = np.random.default_rng(args.seed)
    best_val_acc = -1.0
    for step in trange(1, args.steps + 1):
        batch = train_data.sample_balanced_batch(args.batch_size, rng)
        state, loss = train_step(
            state,
            {
                'observations': jnp.asarray(batch['observations']),
                'actions': jnp.asarray(batch['actions']),
                'labels': jnp.asarray(batch['labels']),
            },
        )

        if step % args.log_interval == 0 or step == 1:
            print(f'step {step}: train_loss={float(loss):.4f}')
            if wandb is not None:
                wandb.log({'train/loss': float(loss)}, step=step)

        if step % args.eval_interval == 0 or step == 1:
            train_m = eval_dataset(state, train_data, args.batch_size, rng)
            val_m = eval_dataset(state, val_data, args.batch_size, rng)
            print(
                f'  eval step {step}: train_acc={train_m.get("accuracy", 0):.3f} '
                f'val_acc={val_m.get("accuracy", 0):.3f} val_loss={val_m.get("loss", 0):.4f}'
            )
            if wandb is not None:
                wandb.log({
                    'train/accuracy': train_m.get('accuracy', 0),
                    'val/accuracy': val_m.get('accuracy', 0),
                    'val/loss': val_m.get('loss', 0),
                }, step=step)
            if val_m.get('accuracy', 0) > best_val_acc:
                best_val_acc = val_m['accuracy']
                save_checkpoint(
                    os.path.join(args.checkpoint_dir, 'best.pkl'),
                    state, obs_dim, act_dim, args.hidden, step, chunk_size,
                )

        if step % args.save_interval == 0:
            save_checkpoint(
                os.path.join(args.checkpoint_dir, f'step_{step:06d}.pkl'),
                state, obs_dim, act_dim, args.hidden, step, chunk_size,
            )

    save_checkpoint(
        os.path.join(args.checkpoint_dir, 'final.pkl'),
        state, obs_dim, act_dim, args.hidden, args.steps, chunk_size,
    )
    print(f'done. best val acc={best_val_acc:.3f}')
    if wandb is not None:
        wandb.summary['best_val_acc'] = best_val_acc
        wandb.finish()


if __name__ == '__main__':
    main()
