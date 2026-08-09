"""AWR policy extraction — FQL Actor + advantage-weighted regression (exact AWR loss)."""

from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm import trange

from awr.iql.networks import Actor


def _awr_loss(dist, actions, adv, alpha):
    """FQL AWR: -(min(exp(alpha * adv), 100) * log_prob).mean()."""
    exp_a = jnp.minimum(jnp.exp(adv * alpha), 100.0)
    log_prob = dist.log_prob(actions)
    loss = -(exp_a * log_prob).mean()
    return loss, {
        'actor_loss': loss,
        'adv_mean': adv.mean(),
        'adv_std': adv.std(),
        'bc_log_prob': log_prob.mean(),
        'mse': jnp.mean((dist.mode() - actions) ** 2),
        'weight_mean': exp_a.mean(),
        'weight_std': exp_a.std(),
        'weight_min': exp_a.min(),
        'weight_max': exp_a.max(),
        'weight_clip_frac': jnp.mean(exp_a >= 100.0 - 1e-6),
    }


def weight_stats(advantages: np.ndarray, alpha: float) -> dict[str, float]:
    """Dataset-level effective AWR weight statistics: w = min(exp(alpha * A), 100)."""
    adv = np.asarray(advantages, dtype=np.float64)
    w = np.minimum(np.exp(adv * alpha), 100.0)
    sum_w = float(w.sum())
    sum_w2 = float((w ** 2).sum())
    ess = (sum_w ** 2) / (sum_w2 + 1e-12)
    return {
        'weight_mean': float(w.mean()),
        'weight_std': float(w.std()),
        'weight_min': float(w.min()),
        'weight_max': float(w.max()),
        'weight_median': float(np.median(w)),
        'weight_p90': float(np.percentile(w, 90)),
        'weight_p99': float(np.percentile(w, 99)),
        'weight_clip_frac': float(np.mean(w >= 100.0 - 1e-6)),
        'weight_ess': ess,
        'weight_ess_frac': ess / max(len(w), 1),
        'adv_mean': float(adv.mean()),
        'adv_std': float(adv.std()),
    }


def create_actor(obs_dim, act_dim, hidden_dims=(512, 512, 512, 512), lr=3e-4, seed=0, const_std=True):
    model = Actor(
        hidden_dims=hidden_dims,
        action_dim=act_dim,
        layer_norm=False,
        state_dependent_std=False,
        const_std=const_std,
    )
    key = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.zeros((1, obs_dim)))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(lr))
    return model, state


def extract_awr(
    observations: np.ndarray,
    actions: np.ndarray,
    advantage_fn=None,
    epochs: int = 10,
    batch_size: int = 256,
    alpha: float = 10.0,
    lr: float = 3e-4,
    hidden_dims=(512, 512, 512, 512),
    seed: int = 0,
    chunk_size: int = 4,
    advantages: np.ndarray | None = None,
) -> dict:
    """Train Gaussian actor with AWR for `epochs` passes over the dataset.

    Provide either ``advantage_fn(obs, acts) -> adv`` or a precomputed
    ``advantages`` array aligned with ``observations`` / ``actions``.
    """
    if advantages is None and advantage_fn is None:
        raise ValueError('provide advantage_fn or advantages')
    if advantages is not None:
        advantages = np.asarray(advantages, np.float32)
        if len(advantages) != len(observations):
            raise ValueError(
                f'advantages length {len(advantages)} != observations {len(observations)}'
            )

    obs_dim = observations.shape[1]
    act_dim = actions.shape[1]
    model, state = create_actor(obs_dim, act_dim, hidden_dims=hidden_dims, lr=lr, seed=seed)
    n = len(observations)
    rng = np.random.default_rng(seed)
    steps_per_epoch = max(1, n // batch_size)
    info = {}

    def _batch_adv(idx):
        if advantages is not None:
            return np.asarray(advantages[idx], np.float32)
        return np.asarray(
            advantage_fn(np.asarray(observations[idx]), np.asarray(actions[idx])),
            np.float32,
        )

    @jax.jit
    def train_step(state, obs, acts, adv):
        def loss_fn(params):
            dist = state.apply_fn(params, obs)
            loss, metrics = _awr_loss(dist, acts, adv, alpha)
            return loss, metrics

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), metrics

    for ep in trange(epochs, desc='awr epochs'):
        perm = rng.permutation(n)
        for s in range(steps_per_epoch):
            idx = perm[s * batch_size : (s + 1) * batch_size]
            if len(idx) == 0:
                continue
            obs = jnp.asarray(observations[idx])
            acts = jnp.asarray(actions[idx])
            adv = jnp.asarray(_batch_adv(idx))
            state, info = train_step(state, obs, acts, adv)

    # Full-dataset effective weight stats for wandb.
    all_adv = []
    for s in range(0, n, batch_size):
        idx = np.arange(s, min(s + batch_size, n))
        all_adv.append(np.asarray(_batch_adv(idx), dtype=np.float64))
    all_adv = np.concatenate(all_adv, axis=0)
    metrics = {k: float(v) for k, v in info.items()}
    metrics.update(weight_stats(all_adv, alpha))

    return {
        'mode': 'awr_actor',
        'params': state.params,
        'apply_fn': model.apply,
        'obs_dim': obs_dim,
        'act_dim': act_dim,
        'chunk_size': chunk_size,
        'hidden_dims': list(hidden_dims),
        'metrics': metrics,
    }


def make_iql_advantage_fn(iql_agent):
    def advantage_fn(obs, acts):
        obs_j, acts_j = jnp.asarray(obs), jnp.asarray(acts)
        q1, q2 = iql_agent.network.select('critic')(obs_j, actions=acts_j)
        v = iql_agent.network.select('value')(obs_j)
        return np.asarray(jnp.minimum(q1, q2) - v)

    return advantage_fn


def actor_ckpt_from_iql(iql_agent, chunk_size: int) -> dict:
    """Export the jointly trained IQL actor for collection / eval (no second AWR stage)."""
    actor_params = iql_agent.network.params['modules_actor']
    # Standalone Actor.apply expects variables={'params': ...}; match extract_awr / create_actor.
    if 'params' in actor_params:
        params = actor_params
    else:
        params = {'params': actor_params}

    mean_kernel = params['params']['mean_net']['kernel']
    act_dim = int(mean_kernel.shape[-1])
    hidden_dims = tuple(int(x) for x in iql_agent.config['actor_hidden_dims'])
    model = Actor(
        hidden_dims=hidden_dims,
        action_dim=act_dim,
        layer_norm=bool(iql_agent.config['actor_layer_norm']),
        state_dependent_std=False,
        const_std=bool(iql_agent.config['const_std']),
    )
    return {
        'mode': 'iql_actor',
        'params': params,
        'apply_fn': model.apply,
        'act_dim': act_dim,
        'chunk_size': int(chunk_size),
        'hidden_dims': list(hidden_dims),
        'metrics': {},
    }


def make_classifier_advantage_fn(clf_ckpt):
    apply_fn, params = clf_ckpt['apply_fn'], clf_ckpt['params']

    def advantage_fn(obs, acts):
        return np.asarray(apply_fn(params, jnp.asarray(obs), jnp.asarray(acts)))

    return advantage_fn


def sample_action_chunk(actor_ckpt, observation, key, temperature=1.0):
    """Sample one flattened action chunk, reshape to (H, act_dim)."""
    apply_fn, params = actor_ckpt['apply_fn'], actor_ckpt['params']
    chunk_size = actor_ckpt['chunk_size']
    act_dim = actor_ckpt['act_dim'] // chunk_size
    dist = apply_fn(params, jnp.asarray(observation)[None], temperature=temperature)
    flat = dist.sample(seed=key)[0]
    flat = jnp.clip(flat, -1.0, 1.0)
    return np.asarray(flat).reshape(chunk_size, act_dim)


def save_actor(path: str, ckpt: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({k: v for k, v in ckpt.items() if k != 'apply_fn'}, f)


def load_actor(path: str) -> dict:
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)
    model = Actor(
        hidden_dims=tuple(ckpt['hidden_dims']),
        action_dim=ckpt['act_dim'],
        layer_norm=False,
        state_dependent_std=False,
        const_std=True,
    )
    ckpt['apply_fn'] = model.apply
    return ckpt
