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
from bon_sampling.advantage.model import AdvantageClassifier, expected_bin_advantage


def bce_loss(logits, labels):
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(params, batch['observations'], batch['actions'])
        loss = bce_loss(logits, batch['labels'])
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


@jax.jit
def eval_metrics(state, observations, actions, labels):
    logits = state.apply_fn(state.params, observations, actions)
    probs = jax.nn.sigmoid(logits)
    preds = (probs >= 0.5).astype(jnp.float32)
    acc = jnp.mean((preds == labels).astype(jnp.float32))
    loss = bce_loss(logits, labels)
    pos_rate = jnp.mean(labels)
    pred_pos = jnp.mean(preds)
    return {'loss': loss, 'accuracy': acc, 'label_pos_rate': pos_rate, 'pred_pos_rate': pred_pos}


def sample_batch(dataset, batch_size, rng):
    n = len(dataset)
    sel = rng.integers(0, n, size=batch_size)
    return dataset.get_batch(sel)


def eval_dataset(state, dataset, batch_size, rng):
    if len(dataset) == 0:
        return {}
    n = len(dataset)
    losses, accs = [], []
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
        accs.append(float(m['accuracy']))
    return {'loss': np.mean(losses), 'accuracy': np.mean(accs)}


def softmax_ce_loss(logits, labels):
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))


@jax.jit
def train_step_bins(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(params, batch['observations'], batch['actions'])
        return softmax_ce_loss(logits, batch['labels'])

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


@jax.jit
def eval_metrics_bins(state, observations, actions, labels):
    logits = state.apply_fn(state.params, observations, actions)
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean((preds == labels).astype(jnp.float32))
    loss = softmax_ce_loss(logits, labels)
    ell_true = labels.astype(jnp.float32) - (logits.shape[-1] - 1) / 2.0
    ell_hat = expected_bin_advantage(logits)
    mae = jnp.mean(jnp.abs(ell_hat - ell_true))
    return {'loss': loss, 'accuracy': acc, 'expected_mae': mae}


def eval_dataset_bins(state, dataset, batch_size):
    if len(dataset) == 0:
        return {}
    n = len(dataset)
    losses, accs, maes = [], [], []
    for start in range(0, n, batch_size):
        sel = np.arange(start, min(start + batch_size, n))
        batch = dataset.get_batch(sel)
        m = eval_metrics_bins(
            state,
            jnp.asarray(batch['observations']),
            jnp.asarray(batch['actions']),
            jnp.asarray(batch['labels'], dtype=np.int32),
        )
        losses.append(float(m['loss']))
        accs.append(float(m['accuracy']))
        maes.append(float(m['expected_mae']))
    return {
        'loss': float(np.mean(losses)),
        'accuracy': float(np.mean(accs)),
        'expected_mae': float(np.mean(maes)),
    }


def save_checkpoint(path, state, obs_dim, act_dim, hidden, step, chunk_size=1):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(
            {
                'mode': 'classifier',
                'step': step,
                'params': state.params,
                'obs_dim': obs_dim,
                'act_dim': act_dim,
                'chunk_size': chunk_size,
                'hidden': hidden,
            },
            f,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='bon_sampling/data/bc_transitions_annotated.npz')
    p.add_argument('--checkpoint_dir', default='bon_sampling/advantage/checkpoints')
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

    train_data, val_data = make_train_val(args.data, args.val_ratio, args.seed)
    chunk_size = train_data.chunk_size if train_data.chunk_mode else 1
    mode_str = f'chunk (H={chunk_size})' if train_data.chunk_mode else 'per-step'
    print(
        f'mode={mode_str} train={len(train_data)} val={len(val_data)} '
        f'train y=0/1={train_data.label_balance()} val y=0/1={val_data.label_balance()}'
    )

    obs_dim = train_data.observations.shape[1]
    act_dim = train_data.act_dim
    model = AdvantageClassifier(hidden=args.hidden)

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
    best_val_acc = -1.0
    for step in trange(1, args.steps + 1):
        batch = sample_batch(train_data, args.batch_size, rng)
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

        if step % args.eval_interval == 0 or step == 1:
            train_m = eval_dataset(state, train_data, args.batch_size, rng)
            val_m = eval_dataset(state, val_data, args.batch_size, rng)
            print(
                f'  eval step {step}: train_acc={train_m.get("accuracy", 0):.3f} '
                f'val_acc={val_m.get("accuracy", 0):.3f} val_loss={val_m.get("loss", 0):.4f}'
            )
            if val_m.get('accuracy', 0) > best_val_acc:
                best_val_acc = val_m['accuracy']
                save_checkpoint(
                    os.path.join(args.checkpoint_dir, 'best.pkl'),
                    state,
                    obs_dim,
                    act_dim,
                    args.hidden,
                    step,
                    chunk_size,
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
            )

    save_checkpoint(
        os.path.join(args.checkpoint_dir, 'final.pkl'),
        state,
        obs_dim,
        act_dim,
        args.hidden,
        args.steps,
        chunk_size,
    )
    print(f'done. best val acc={best_val_acc:.3f}')


if __name__ == '__main__':
    main()
