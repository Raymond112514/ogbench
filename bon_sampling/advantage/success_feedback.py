"""In-memory success/failure classifier feedback for online policy training."""

from __future__ import annotations

import numpy as np


def build_episode_success_chunks(rollout: dict, chunk_size: int) -> dict[str, np.ndarray]:
    """Build full action chunks labeled by their episode's final success."""
    observations, chunks, masks, labels, goals = [], [], [], [], []
    has_goals = 'goals' in rollout
    starts = [0] + np.asarray(rollout['episode_ends'], np.int64)[:-1].tolist()
    ends = np.asarray(rollout['episode_ends'], np.int64)
    step_success = np.asarray(rollout['successes'], bool)

    for start, end in zip(starts, ends.tolist()):
        episode_success = float(step_success[end - 1])
        t = start
        # Match the oracle-feedback pipelines: require s_{t+H} in the episode.
        while t + chunk_size < end:
            observations.append(rollout['observations'][t])
            chunks.append(rollout['actions'][t : t + chunk_size])
            masks.append(np.ones(chunk_size, np.float32))
            labels.append(episode_success)
            if has_goals:
                goals.append(rollout['goals'][t])
            t += chunk_size

    act_dim = rollout['actions'].shape[-1]
    obs_dim = rollout['observations'].shape[-1]
    out = {
        'observations': np.asarray(observations, np.float32).reshape(-1, obs_dim),
        'action_chunks': np.asarray(chunks, np.float32).reshape(-1, chunk_size, act_dim),
        'chunk_masks': np.asarray(masks, np.float32).reshape(-1, chunk_size),
        'success_labels': np.asarray(labels, np.float32),
    }
    if has_goals:
        goal_dim = rollout['goals'].shape[-1]
        out['goals'] = np.asarray(goals, np.float32).reshape(-1, goal_dim)
    return out


def merge_success_chunks(buffers: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = buffers[0].keys()
    return {key: np.concatenate([buffer[key] for buffer in buffers], axis=0) for key in keys}


def train_success_classifier(
    params,
    data: dict[str, np.ndarray],
    *,
    hidden: int,
    train_steps: int,
    batch_size: int,
    lr: float,
    seed: int,
):
    """Continue classifier training with half-positive, half-negative batches."""
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state

    from bon_sampling.advantage.model import AdvantageClassifier
    from bon_sampling.advantage.train import train_step

    labels = np.asarray(data['success_labels'], np.float32)
    pos = np.flatnonzero(labels > 0.5)
    neg = np.flatnonzero(labels <= 0.5)
    if len(pos) == 0 or len(neg) == 0:
        return params, {
            'skipped': 1.0,
            'loss': 0.0,
            'num_positive': float(len(pos)),
            'num_negative': float(len(neg)),
        }

    obs_dim = data['observations'].shape[-1]
    action_chunks = data['action_chunks']
    act_dim = action_chunks.shape[1] * action_chunks.shape[2]
    model = AdvantageClassifier(hidden=hidden)
    key = jax.random.PRNGKey(seed)
    if params is None:
        params = model.init(key, jnp.zeros((1, obs_dim)), jnp.zeros((1, act_dim)))
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(lr)
    )

    rng = np.random.default_rng(seed)
    last_loss = 0.0
    for _ in range(train_steps):
        n_pos = batch_size // 2
        n_neg = batch_size - n_pos
        indices = np.concatenate([
            rng.choice(pos, size=n_pos, replace=True),
            rng.choice(neg, size=n_neg, replace=True),
        ])
        rng.shuffle(indices)
        batch = {
            'observations': jnp.asarray(data['observations'][indices]),
            'actions': jnp.asarray(action_chunks[indices].reshape(len(indices), -1)),
            'labels': jnp.asarray(labels[indices]),
        }
        state, loss = train_step(state, batch)
        last_loss = float(loss)

    return state.params, {
        'skipped': 0.0,
        'loss': last_loss,
        'num_positive': float(len(pos)),
        'num_negative': float(len(neg)),
    }


def predict_success_probabilities(
    params,
    data: dict[str, np.ndarray],
    *,
    hidden: int,
    batch_size: int,
) -> np.ndarray:
    """Predict P(episode success | s, action chunk)."""
    import jax
    import jax.numpy as jnp

    from bon_sampling.advantage.model import AdvantageClassifier

    if params is None:
        raise ValueError('success classifier has not been trained')
    model = AdvantageClassifier(hidden=hidden)
    chunks = data['action_chunks']
    probabilities = []
    for start in range(0, len(chunks), batch_size):
        stop = min(start + batch_size, len(chunks))
        logits = model.apply(
            params,
            jnp.asarray(data['observations'][start:stop]),
            jnp.asarray(chunks[start:stop].reshape(stop - start, -1)),
        )
        probabilities.append(np.asarray(jax.nn.sigmoid(logits), np.float32))
    return np.concatenate(probabilities) if probabilities else np.zeros(0, np.float32)


def threshold_success_feedback(
    data: dict[str, np.ndarray],
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Return flow-training data with binary classifier predictions as advantages."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f'classifier threshold must be in [0, 1], got {threshold}')
    advantages = (np.asarray(probabilities) >= threshold).astype(np.float32)
    out = {
        'observations': data['observations'],
        'action_chunks': data['action_chunks'],
        'chunk_masks': data['chunk_masks'],
        'advantages': advantages,
    }
    if 'goals' in data:
        out['goals'] = data['goals']
    return out, {
        'num_chunks': float(len(advantages)),
        'num_positive': float(advantages.sum()),
        'positive_frac': float(advantages.mean()) if len(advantages) else 0.0,
        'mean_probability': float(np.mean(probabilities)) if len(probabilities) else 0.0,
        'threshold': float(threshold),
    }
