"""Train IQL jointly (V + Q + AWR actor) — same loop as seohongpark/fql/main.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
from tqdm import trange

from awr.iql.agent import IQLAgent, get_config
from awr.iql.dataset import IQLDataset


def _to_float_metrics(info: dict) -> dict[str, float]:
    metrics = {}
    for k, v in info.items():
        try:
            metrics[k] = float(v)
        except (TypeError, ValueError):
            pass
    return metrics


def train_iql(
    steps: int,
    *,
    dataset: IQLDataset | None = None,
    rollouts: list[dict] | list[str | Path] | None = None,
    seed: int = 0,
    batch_size: int = 256,
    expectile: float = 0.9,
    alpha: float = 10.0,
    lr: float = 3e-4,
    discount: float = 0.99,
    tau: float = 0.005,
    log_interval: int = 5000,
    eval_interval: int = 0,
    eval_fn: Callable[[IQLAgent, int], None] | None = None,
    wandb_run: Any = None,
) -> tuple[IQLAgent, dict]:
    """Joint IQL update for `steps` gradient steps (value + critic + AWR actor).

    Matches FQL: each `agent.update(batch)` optimizes V, Q, and the actor together.
    Periodic wandb keys mirror FQL: `training/value/*`, `training/critic/*`, `training/actor/*`.
    """
    if dataset is None:
        if not rollouts:
            raise ValueError('Provide dataset= or rollouts=')
        if isinstance(rollouts[0], dict):
            dataset = IQLDataset.from_rollouts(rollouts)
        else:
            dataset = IQLDataset.from_paths(rollouts)
    if dataset.size == 0:
        raise ValueError('IQL dataset is empty')

    config = get_config()
    config.batch_size = batch_size
    config.expectile = expectile
    config.alpha = alpha
    config.lr = lr
    config.discount = discount
    config.tau = tau

    agent = IQLAgent.create(
        seed=seed,
        ex_observations=dataset.data['observations'][:1],
        ex_actions=dataset.data['actions'][:1],
        config=config,
    )
    rng = np.random.default_rng(seed)
    info = {}
    for i in trange(1, steps + 1, desc='train iql (joint V/Q/actor)'):
        batch = {k: jnp.asarray(v) for k, v in dataset.sample(config.batch_size, rng).items()}
        agent, info = agent.update(batch)

        if wandb_run is not None and log_interval > 0 and i % log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in _to_float_metrics(info).items()}
            wandb_run.log(train_metrics, step=i)

        if eval_fn is not None and eval_interval > 0 and (i == 1 or i % eval_interval == 0):
            eval_fn(agent, i)

    return agent, _to_float_metrics(info)
