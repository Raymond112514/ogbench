"""Train IQL (FQL) and return agent in memory."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from tqdm import trange

from awr.iql.agent import IQLAgent, get_config
from awr.iql.dataset import IQLDataset


def train_iql(
    rollouts: list[dict] | list[str | Path],
    steps: int,
    seed: int = 0,
    batch_size: int = 256,
    expectile: float = 0.9,
    alpha: float = 10.0,
    lr: float = 3e-4,
) -> tuple[IQLAgent, dict]:
    if rollouts and isinstance(rollouts[0], dict):
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

    agent = IQLAgent.create(
        seed=seed,
        ex_observations=dataset.data['observations'][:1],
        ex_actions=dataset.data['actions'][:1],
        config=config,
    )
    rng = np.random.default_rng(seed)
    info = {}
    for _ in trange(1, steps + 1, desc='train iql'):
        batch = {k: jnp.asarray(v) for k, v in dataset.sample(config.batch_size, rng).items()}
        agent, info = agent.update(batch)

    metrics = {}
    for k, v in info.items():
        try:
            metrics[k] = float(v)
        except (TypeError, ValueError):
            pass
    return agent, metrics
