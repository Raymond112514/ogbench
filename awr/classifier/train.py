"""Train advantage classifier."""

from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm import trange

from awr.classifier.dataset import ClassifierDataset
from awr.classifier.model import AdvantageClassifier


@jax.jit
def _train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(params, batch['observations'], batch['actions'])
        return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, batch['labels']))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


def train_classifier(
    data: str | Path | dict,
    steps: int,
    batch_size: int = 256,
    lr: float = 3e-4,
    hidden: int = 256,
    seed: int = 0,
) -> dict:
    dataset = ClassifierDataset(data if isinstance(data, dict) else str(data))
    model = AdvantageClassifier(hidden=hidden)
    key = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.zeros((1, dataset.obs_dim)), jnp.zeros((1, dataset.act_dim)))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(lr))
    rng = np.random.default_rng(seed)
    loss = 0.0
    for _ in trange(1, steps + 1, desc='train classifier'):
        batch = {k: jnp.asarray(v) for k, v in dataset.sample(batch_size, rng).items()}
        state, loss = _train_step(state, batch)
    return {
        'mode': 'classifier',
        'params': state.params,
        'apply_fn': model.apply,
        'obs_dim': dataset.obs_dim,
        'act_dim': dataset.act_dim,
        'chunk_size': dataset.chunk_size,
        'hidden': hidden,
        'loss': float(loss),
    }


def save_classifier(path: str, ckpt: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({k: v for k, v in ckpt.items() if k != 'apply_fn'}, f)


def load_classifier(path: str) -> dict:
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)
    model = AdvantageClassifier(hidden=ckpt['hidden'])
    ckpt['apply_fn'] = model.apply
    return ckpt
