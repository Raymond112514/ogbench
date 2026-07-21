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


def eval_classifier(
    agent: ClassifierAWRAgent,
    dataset: ClassifierOgbenchDataset,
    batch_size: int,
    rng: np.random.Generator,
    num_batches: int = 20,
) -> dict[str, float]:
    """Average classifier loss/accuracy over random val batches."""
    if dataset.size == 0:
        return {}
    losses, accs, logit_means, label_means = [], [], [], []
    n = min(num_batches, max(1, (dataset.size + batch_size - 1) // batch_size))
    for _ in range(n):
        batch = {k: jnp.asarray(v) for k, v in dataset.sample(min(batch_size, dataset.size), rng).items()}
        m = agent.classifier_metrics(batch)
        losses.append(float(m['classifier_loss']))
        accs.append(float(m['accuracy']))
        logit_means.append(float(m['logit_mean']))
        label_means.append(float(m['label_mean']))
    return {
        'classifier_loss': float(np.mean(losses)),
        'accuracy': float(np.mean(accs)),
        'logit_mean': float(np.mean(logit_means)),
        'label_mean': float(np.mean(label_means)),
    }


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
    val_ratio: float = 0.1,
    classifier_update_every: int = 10,
    log_interval: int = 5000,
    eval_interval: int = 0,
    eval_fn: Callable[[ClassifierAWRAgent, int], None] | None = None,
    wandb_run: Any = None,
    step_offset: int = 0,
) -> tuple[ClassifierAWRAgent, dict]:
    """Actor updated every step; classifier every `classifier_update_every` steps.

    Holds out `val_ratio` of chunks for classifier validation metrics (loss/accuracy).
    Actor AWR and classifier BCE both sample from the train split only.
    """
    if dataset.size == 0:
        raise ValueError('classifier dataset is empty')
    if classifier_update_every < 1:
        raise ValueError(f'classifier_update_every must be >= 1, got {classifier_update_every}')

    train_data, val_data = dataset.split_train_val(val_ratio, seed=seed)

    config = get_config()
    config.batch_size = batch_size
    config.alpha = alpha
    config.lr = lr
    config.classifier_hidden = classifier_hidden

    if agent is None:
        agent = ClassifierAWRAgent.create(
            seed=seed,
            ex_observations=train_data.data['observations'][:1],
            ex_actions=train_data.data['actions'][:1],
            config=config,
        )
    rng = np.random.default_rng(seed)
    info = {}
    best_val_acc = -1.0
    best_val_loss = float('inf')
    for i in trange(1, steps + 1, desc='train classifier-awr'):
        batch = {k: jnp.asarray(v) for k, v in train_data.sample(config.batch_size, rng).items()}
        update_clf = (i % classifier_update_every) == 0
        agent, info = agent.update(batch, update_classifier=update_clf)
        step = i + step_offset

        do_val = val_data.size > 0 and log_interval > 0 and i % log_interval == 0
        val_m = eval_classifier(agent, val_data, batch_size, rng) if do_val else {}
        if val_m and val_m.get('accuracy', -1.0) > best_val_acc:
            best_val_acc = val_m['accuracy']
            best_val_loss = val_m['classifier_loss']

        if wandb_run is not None and log_interval > 0 and i % log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in _to_float_metrics(info).items()}
            if val_m:
                train_metrics.update({f'validation/{k}': v for k, v in val_m.items()})
                train_metrics['validation/best_accuracy'] = best_val_acc
                train_metrics['validation/best_classifier_loss'] = best_val_loss
            train_metrics['training/classifier_update_every'] = float(classifier_update_every)
            train_metrics['dataset/train_size'] = float(train_data.size)
            train_metrics['dataset/val_size'] = float(val_data.size)
            wandb_run.log(train_metrics, step=step)

        if eval_fn is not None and eval_interval > 0 and (i == 1 or i % eval_interval == 0):
            eval_fn(agent, step)

    metrics = _to_float_metrics(info)
    metrics['train_size'] = float(train_data.size)
    metrics['val_size'] = float(val_data.size)
    if val_data.size > 0:
        val_m = eval_classifier(agent, val_data, batch_size, rng)
        metrics.update({f'val_{k}': v for k, v in val_m.items()})
        metrics['val_best_accuracy'] = best_val_acc
        metrics['val_best_classifier_loss'] = best_val_loss
    return agent, metrics
