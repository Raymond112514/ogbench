"""Classifier advantage model (oracle-distance improvement labels)."""

import flax.linen as nn
import jax
import jax.numpy as jnp


class AdvantageClassifier(nn.Module):
    hidden: int = 256

    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)
        for _ in range(2):
            x = nn.Dense(self.hidden)(x)
            x = nn.relu(x)
        return nn.Dense(1)(x).squeeze(-1)


def predict_logits(apply_fn, params, observations, actions):
    return apply_fn(params, observations, actions)


def predict_proba(apply_fn, params, observations, actions):
    return jax.nn.sigmoid(predict_logits(apply_fn, params, observations, actions))
