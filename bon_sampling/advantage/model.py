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


class AdvantageBinClassifier(nn.Module):
    """Softmax over signed Δ bins; BoN score is E[ℓ] under p(ℓ | s, a)."""

    hidden: int = 256
    num_bins: int = 5

    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)
        for _ in range(2):
            x = nn.Dense(self.hidden)(x)
            x = nn.relu(x)
        return nn.Dense(self.num_bins)(x)


def expected_bin_advantage(logits):
    """A = Σ_i softmax(logits)_i * ℓ_i with ℓ_i = i - (w-1)/2."""
    w = logits.shape[-1]
    values = jnp.arange(w, dtype=logits.dtype) - (w - 1) / 2.0
    return jnp.sum(jax.nn.softmax(logits, axis=-1) * values, axis=-1)


def predict_proba(apply_fn, params, observations, actions):
    logits = apply_fn(params, observations, actions)
    return jax.nn.sigmoid(logits)


def predict_delta(apply_fn, params, observations, actions):
    return apply_fn(params, observations, actions)
