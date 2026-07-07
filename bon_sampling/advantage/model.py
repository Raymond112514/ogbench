import flax.linen as nn
import jax
import jax.numpy as jnp


class AdvantageMLP(nn.Module):
    """Shared MLP backbone for classifier and regression advantage models."""

    hidden: int = 256

    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)
        for _ in range(2):
            x = nn.Dense(self.hidden)(x)
            x = nn.relu(x)
        return nn.Dense(1)(x).squeeze(-1)


class AdvantageClassifier(AdvantageMLP):
    """Binary classifier: P(oracle distance improves)."""


class AdvantageRegressor(AdvantageMLP):
    """Regressor: Q(s, a) ≈ (gamma^{d'} - gamma^{d-1}) / (1 - gamma)."""


def predict_proba(apply_fn, params, observations, actions):
    logits = apply_fn(params, observations, actions)
    return jax.nn.sigmoid(logits)


def predict_delta(apply_fn, params, observations, actions):
    return apply_fn(params, observations, actions)
