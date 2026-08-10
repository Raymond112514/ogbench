"""Rectified-flow BC with action chunking and optional goal/advantage conditioning."""

from __future__ import annotations
from functools import partial
from typing import Sequence
import flax.linen as nn
import jax
import jax.numpy as jnp

ADVANTAGE_NULL = -1.0

class VelocityNet(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int

    @nn.compact
    def __call__(self, x_tau, tau, cond):
        if tau.ndim == 0:
            tau = jnp.broadcast_to(tau[None, None], (x_tau.shape[0], 1))
        elif tau.ndim == 1:
            tau = tau[:, None]
        h = jnp.concatenate([x_tau, tau, cond], axis=-1)
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = nn.gelu(h)
        return nn.Dense(self.out_dim)(h)


def cond_dim(
    obs_dim: int,
    goal_dim: int | None = None,
    *,
    goal_condition: bool = True,
    advantage_condition: bool = False,
) -> int:
    dim = obs_dim
    if goal_condition:
        dim += goal_dim if goal_dim is not None else obs_dim
    if advantage_condition:
        dim += 1
    return dim


def make_cond(
    observations,
    goal=None,
    advantage=None,
    *,
    goal_condition: bool = True,
    advantage_condition: bool = False,
):
    parts = [observations]
    if goal_condition:
        if goal is None:
            raise ValueError('goal is required')
        parts.append(goal)
    if advantage_condition:
        if advantage is None:
            raise ValueError('advantage is required')
        adv = advantage
        if adv.ndim == 1:
            adv = adv[:, None]
        parts.append(adv.astype(observations.dtype))
    return jnp.concatenate(parts, axis=-1)


def apply_advantage_cfg_dropout(
    advantage,
    rng,
    dropout_prob: float,
    null_value: float = ADVANTAGE_NULL,
):
    if dropout_prob <= 0.0 or advantage is None:
        return advantage
    mask = jax.random.bernoulli(rng, dropout_prob, advantage.shape)
    null = jnp.full_like(advantage, null_value)
    return jnp.where(mask, null, advantage)


def flow_loss(
    params,
    apply_fn,
    observations,
    action_chunks,
    chunk_masks,
    rng,
    goal=None,
    advantage=None,
    *,
    goal_condition: bool = True,
    advantage_condition: bool = False,
    cfg_dropout: float = 0.0,
    advantage_null: float = ADVANTAGE_NULL,
):
    B, H, act_dim = action_chunks.shape
    x1 = action_chunks.reshape(B, H * act_dim)
    rng_x0, rng_tau, rng_cfg = jax.random.split(rng, 3)
    x0 = jax.random.normal(rng_x0, x1.shape)
    tau = jax.random.uniform(rng_tau, (B,))
    x_tau = (1.0 - tau[:, None]) * x0 + tau[:, None] * x1
    v_target = x1 - x0
    if advantage_condition and cfg_dropout > 0.0:
        advantage = apply_advantage_cfg_dropout(
            advantage, rng_cfg, cfg_dropout, advantage_null
        )
    cond = make_cond(
        observations,
        goal,
        advantage,
        goal_condition=goal_condition,
        advantage_condition=advantage_condition,
    )
    v_pred = apply_fn(params, x_tau, tau, cond)
    residual = (v_pred - v_target).reshape(B, H, act_dim)
    per_step = jnp.mean(residual ** 2, axis=-1)
    masked = per_step * chunk_masks
    loss = jnp.sum(masked) / (jnp.sum(chunk_masks) + 1e-6)
    return loss


@partial(
    jax.jit,
    static_argnames=('goal_condition', 'advantage_condition', 'cfg_dropout', 'advantage_null'),
)
def train_step(
    state,
    batch,
    rng,
    goal_condition=True,
    advantage_condition=False,
    cfg_dropout=0.0,
    advantage_null=ADVANTAGE_NULL,
):
    def loss_fn(params):
        goal = batch['goals'] if goal_condition else None
        advantage = batch['advantages'] if advantage_condition else None
        return flow_loss(
            params,
            state.apply_fn,
            batch['observations'],
            batch['action_chunks'],
            batch['chunk_masks'],
            rng,
            goal=goal,
            advantage=advantage,
            goal_condition=goal_condition,
            advantage_condition=advantage_condition,
            cfg_dropout=cfg_dropout,
            advantage_null=advantage_null,
        )

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


def _make_advantage_cond(obs, goal, advantage_value, *, goal_condition, advantage_condition):
    adv = None
    if advantage_condition:
        adv = jnp.full((obs.shape[0], 1), float(advantage_value), dtype=obs.dtype)
    g = jnp.atleast_2d(goal) if goal_condition else None
    return make_cond(
        obs,
        g,
        adv,
        goal_condition=goal_condition,
        advantage_condition=advantage_condition,
    )


def sample_action_chunks(
    params,
    apply_fn,
    observation,
    goal,
    rng,
    num_samples,
    chunk_size,
    act_dim,
    n_flow_steps=10,
    goal_condition=True,
):
    obs_vec = jnp.asarray(observation).reshape(-1)
    obs = jnp.broadcast_to(obs_vec[None, :], (num_samples, obs_vec.shape[0]))
    out_dim = chunk_size * act_dim
    x = jax.random.normal(rng, (num_samples, out_dim))
    dt = 1.0 / n_flow_steps

    goal_batched = None
    if goal_condition:
        goal_vec = jnp.asarray(goal).reshape(-1)
        goal_batched = jnp.broadcast_to(goal_vec[None, :], (num_samples, goal_vec.shape[0]))
    cond = make_cond(
        obs,
        goal_batched,
        None,
        goal_condition=goal_condition,
        advantage_condition=False,
    )
    for i in range(n_flow_steps):
        tau_val = float(i) / n_flow_steps
        tau = jnp.full((num_samples,), tau_val)
        v = apply_fn(params, x, tau, cond)
        x = x + dt * v

    return jnp.clip(x.reshape(num_samples, chunk_size, act_dim), -1.0, 1.0)


def _rms(v, eps: float = 1e-6):
    """Per-sample RMS over the last axis; keeps leading batch dims."""
    return jnp.sqrt(jnp.mean(jnp.square(v), axis=-1, keepdims=True) + eps)


def renorm_cfg_rms(v_cfg, v_ref, eps: float = 1e-6):
    """Rescale guided velocity so RMS(v_cfg) matches RMS(v_ref) (usually v_uncond)."""
    return v_cfg * (_rms(v_ref, eps) / _rms(v_cfg, eps))


def sample_action_chunk(
    params,
    apply_fn,
    observation,
    goal,
    rng,
    chunk_size,
    act_dim,
    n_flow_steps=10,
    advantage=None,
    goal_condition=True,
    advantage_condition=False,
    cfg_weight=None,
    cfg_cond_advantage=1.0,
    advantage_null=ADVANTAGE_NULL,
    cfg_renorm_rms: bool = False,
):
    use_cfg = cfg_weight is not None and advantage_condition
    if not use_cfg and not advantage_condition:
        return sample_action_chunks(
            params,
            apply_fn,
            observation,
            goal,
            rng,
            1,
            chunk_size,
            act_dim,
            n_flow_steps=n_flow_steps,
            goal_condition=goal_condition,
        )[0]

    obs = jnp.atleast_2d(observation)
    out_dim = chunk_size * act_dim
    x = jax.random.normal(rng, (1, out_dim))
    dt = 1.0 / n_flow_steps

    if use_cfg:
        w = float(cfg_weight)
        cond_value = float(cfg_cond_advantage if advantage is None else advantage)
        cond_uncond = _make_advantage_cond(
            obs, goal, advantage_null, goal_condition=goal_condition, advantage_condition=True
        )
        cond_cond = _make_advantage_cond(
            obs, goal, cond_value, goal_condition=goal_condition, advantage_condition=True
        )
        for i in range(n_flow_steps):
            tau_val = float(i) / n_flow_steps
            tau = jnp.full((1,), tau_val)
            v_uncond = apply_fn(params, x, tau, cond_uncond)
            v_cond = apply_fn(params, x, tau, cond_cond)
            v = v_uncond + w * (v_cond - v_uncond)
            if cfg_renorm_rms:
                v = renorm_cfg_rms(v, v_uncond)
            x = x + dt * v
    else:
        adv = None
        if advantage_condition:
            adv = jnp.full(
                (obs.shape[0], 1),
                float(advantage if advantage is not None else 1.0),
                dtype=obs.dtype,
            )
        cond = make_cond(
            obs,
            jnp.atleast_2d(goal) if goal_condition else None,
            adv,
            goal_condition=goal_condition,
            advantage_condition=advantage_condition,
        )
        for i in range(n_flow_steps):
            tau_val = float(i) / n_flow_steps
            tau = jnp.full((1,), tau_val)
            v = apply_fn(params, x, tau, cond)
            x = x + dt * v

    return jnp.clip(x.reshape(chunk_size, act_dim), -1.0, 1.0)
