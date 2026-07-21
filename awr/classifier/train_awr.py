"""Train the classifier-advantage AWR agent — same loop shape as awr.iql.train.train_iql."""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
from tqdm import trange

from awr.classifier.agent import ClassifierAWRAgent, get_config
from awr.classifier.ogbench_dataset import ClassifierOgbenchDataset


def _to_float_metrics(info: dict) -> dict[str, float]:
    metrics = {}
    for k, v in info.items():
        try:
            metrics[k] = float(v)
        except (TypeError, ValueError):
            pass
    return metrics


def train_classifier_awr(
    steps: int,
    *,
    dataset: ClassifierOgbenchDataset,
    agent: ClassifierAWRAgent | None = None,
    seed: int = 0,
    batch_size: int = 256,
    alpha: float = 10.0,
    lr: float = 3e-4,
    classifier_hidden: int = 256,
    log_interval: int = 5000,
    eval_interval: int = 0,
    eval_fn: Callable[[ClassifierAWRAgent, int], None] | None = None,
    wandb_run: Any = None,
    step_offset: int = 0,
) -> tuple[ClassifierAWRAgent, dict]:
    """Joint classifier + AWR-actor update for `steps` gradient steps.

    Mirrors train_iql: each `agent.update(batch)` jointly optimizes the classifier and the
    actor, at the same frequency (every step) with the same logging/eval cadence. Only the
    advantage source (classifier logit instead of Q - V) differs.

    Pass an existing `agent` to keep training it on a (e.g. grown) dataset instead of
    creating a fresh one — used by the online loop to carry the same agent across rounds.
    `step_offset` shifts the wandb/eval step counter so repeated calls (e.g. once per
    round) don't collide on the same logged steps.
    """
    if dataset.size == 0:
        raise ValueError('classifier dataset is empty')

    config = get_config()
    config.batch_size = batch_size
    config.alpha = alpha
    config.lr = lr
    config.classifier_hidden = classifier_hidden

    if agent is None:
        agent = ClassifierAWRAgent.create(
            seed=seed,
            ex_observations=dataset.data['observations'][:1],
            ex_actions=dataset.data['actions'][:1],
            config=config,
        )
    rng = np.random.default_rng(seed)
    info = {}
    for i in trange(1, steps + 1, desc='train classifier-awr (joint classifier/actor)'):
        batch = {k: jnp.asarray(v) for k, v in dataset.sample(config.batch_size, rng).items()}
        agent, info = agent.update(batch)
        step = i + step_offset

        if wandb_run is not None and log_interval > 0 and i % log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in _to_float_metrics(info).items()}
            wandb_run.log(train_metrics, step=step)

        if eval_fn is not None and eval_interval > 0 and (i == 1 or i % eval_interval == 0):
            eval_fn(agent, step)

    return agent, _to_float_metrics(info)
