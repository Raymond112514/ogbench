"""Train A(s, a) regressor for oracle Q targets
Target: Q(s, a) = (gamma^{d'} - gamma^{d-1}) / (1 - gamma)

Usage (from ogbench/):
  python bon_sampling/advantage/train_regression.py \\
    --data bon_sampling/data/gcbc/flowbc_rollouts_100_annotated.npz \\
    --checkpoint_dir bon_sampling/advantage/checkpoints_flowbc100_regression \\
    --gamma 0.99
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm import trange

from bon_sampling.advantage.dataset import make_train_val
from bon_sampling.advantage.model import AdvantageRegressor


def mse_loss(preds, targets):
    return jnp.mean((preds - targets) ** 2)


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        preds = state.apply_fn(params, batch['observations'], batch['actions'])
        return mse_loss(preds, batch['targets'])

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


@jax.jit
def eval_metrics(state, observations, actions, targets):
    preds = state.apply_fn(state.params, observations, actions)
    loss = mse_loss(preds, targets)
    mae = jnp.mean(jnp.abs(preds - targets))
    return {'loss': loss, 'mae': mae}


def sample_batch(dataset, batch_size, rng):
    n = len(dataset)
    sel = rng.integers(0, n, size=batch_size)
    batch = dataset.get_batch(sel)
    batch['targets'] = batch.pop('labels')
    return batch


def eval_dataset(state, dataset, batch_size):
    if len(dataset) == 0:
        return {}
    n = len(dataset)
    losses, maes = [], []
    for start in range(0, n, batch_size):
        sel = np.arange(start, min(start + batch_size, n))
        batch = dataset.get_batch(sel)
        m = eval_metrics(
            state,
            jnp.asarray(batch['observations']),
            jnp.asarray(batch['actions']),
            jnp.asarray(batch['labels']),
        )
        losses.append(float(m['loss']))
        maes.append(float(m['mae']))
    return {'loss': np.mean(losses), 'mae': np.mean(maes)}


def target_stats(dataset) -> dict[str, float]:
    labels = np.asarray(dataset.labels, dtype=np.float64)
    return {
        'min': float(labels.min()),
        'max': float(labels.max()),
        'mean': float(labels.mean()),
        'std': float(labels.std()),
    }


def save_checkpoint(path, state, obs_dim, act_dim, hidden, step, chunk_size, gamma):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(
            {
                'mode': 'regression',
                'step': step,
                'params': state.params,
                'obs_dim': obs_dim,
                'act_dim': act_dim,
                'chunk_size': chunk_size,
                'hidden': hidden,
                'gamma': gamma,
            },
            f,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='bon_sampling/data/bc_transitions_annotated.npz')
    p.add_argument('--checkpoint_dir', default='bon_sampling/advantage/checkpoints_regression')
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log_interval', type=int, default=500)
    p.add_argument('--eval_interval', type=int, default=2500)
    p.add_argument('--save_interval', type=int, default=5000)
    args = p.parse_args()

    train_data, val_data = make_train_val(
        args.data, args.val_ratio, args.seed, task='regression', gamma=args.gamma
    )
    chunk_size = train_data.chunk_size if train_data.chunk_mode else 1
    mode_str = f'chunk (H={chunk_size})' if train_data.chunk_mode else 'per-step'
    train_stats = target_stats(train_data)
    print(
        f'mode={mode_str} gamma={args.gamma} train={len(train_data)} val={len(val_data)} '
        f'target mean={train_stats["mean"]:.3f} std={train_stats["std"]:.3f} '
        f'range=[{train_stats["min"]:.2f}, {train_stats["max"]:.2f}]'
    )

    obs_dim = train_data.observations.shape[1]
    act_dim = train_data.act_dim
    model = AdvantageRegressor(hidden=args.hidden)

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = model.init(
        init_key,
        jnp.zeros((1, obs_dim)),
        jnp.zeros((1, act_dim)),
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(args.lr)
    )

    rng = np.random.default_rng(args.seed)
    best_val_loss = float('inf')
    for step in trange(1, args.steps + 1):
        batch = sample_batch(train_data, args.batch_size, rng)
        state, loss = train_step(
            state,
            {
                'observations': jnp.asarray(batch['observations']),
                'actions': jnp.asarray(batch['actions']),
                'targets': jnp.asarray(batch['targets']),
            },
        )

        if step % args.log_interval == 0 or step == 1:
            print(f'step {step}: train_loss={float(loss):.4f}')

        if step % args.eval_interval == 0 or step == 1:
            train_m = eval_dataset(state, train_data, args.batch_size)
            val_m = eval_dataset(state, val_data, args.batch_size)
            print(
                f'  eval step {step}: train_mae={train_m.get("mae", 0):.3f} '
                f'val_mae={val_m.get("mae", 0):.3f} val_loss={val_m.get("loss", 0):.4f}'
            )
            if val_m.get('loss', float('inf')) < best_val_loss:
                best_val_loss = val_m['loss']
                save_checkpoint(
                    os.path.join(args.checkpoint_dir, 'best.pkl'),
                    state,
                    obs_dim,
                    act_dim,
                    args.hidden,
                    step,
                    chunk_size,
                    args.gamma,
                )

        if step % args.save_interval == 0:
            save_checkpoint(
                os.path.join(args.checkpoint_dir, f'step_{step:06d}.pkl'),
                state,
                obs_dim,
                act_dim,
                args.hidden,
                step,
                chunk_size,
                args.gamma,
            )

    save_checkpoint(
        os.path.join(args.checkpoint_dir, 'final.pkl'),
        state,
        obs_dim,
        act_dim,
        args.hidden,
        args.steps,
        chunk_size,
        args.gamma,
    )
    print(f'done. best val loss={best_val_loss:.4f}')


if __name__ == '__main__':
    main()
