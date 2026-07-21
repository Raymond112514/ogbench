"""Classifier-advantage AWR agent — same joint-update recipe as awr.iql.agent.IQLAgent.

The only difference from IQL is the advantage source and its network: a progress
classifier (BCE on oracle-improvement labels) replaces the V/Q value networks, and its
logit (analogous to Q - V) is used as the AWR advantage. Everything else — one shared
TrainState/optimizer, one gradient step per `update(batch)` call jointly moving both
networks, the AWR actor loss/weighting, and `sample_actions` — is identical to IQL.
"""

from __future__ import annotations

from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from awr.classifier.model import AdvantageClassifier
from awr.iql.flax_utils import ModuleDict, TrainState, nonpytree_field
from awr.iql.networks import Actor


class ClassifierAWRAgent(flax.struct.PyTreeNode):
    """Progress-classifier advantage + AWR actor, trained jointly every step."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def classifier_loss(self, batch, grad_params):
        """BCE loss on progress labels (1 if s_{t+H} improved over s_t, else 0)."""
        logits = self.network.select('classifier')(
            batch['observations'], batch['actions'], params=grad_params
        )
        loss = optax.sigmoid_binary_cross_entropy(logits, batch['labels']).mean()
        pred = (logits > 0).astype(jnp.float32)

        return loss, {
            'classifier_loss': loss,
            'accuracy': jnp.mean(pred == batch['labels']),
            'logit_mean': logits.mean(),
            'label_mean': batch['labels'].mean(),
        }

    def actor_loss(self, batch, grad_params):
        """AWR actor loss; advantage = classifier logit (stop-gradient, like IQL's Q - V)."""
        adv = self.network.select('classifier')(batch['observations'], batch['actions'])
        exp_a = jnp.minimum(jnp.exp(adv * self.config['alpha']), 100.0)

        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])
        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params):
        """Sum of classifier + actor losses; each only back-props into its own network."""
        info = {}

        classifier_loss, classifier_info = self.classifier_loss(batch, grad_params)
        for k, v in classifier_info.items():
            info[f'classifier/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        return classifier_loss + actor_loss, info

    @jax.jit
    def update(self, batch):
        """Joint gradient step on classifier + actor (no target networks needed)."""

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]

        classifier_def = AdvantageClassifier(hidden=config['classifier_hidden'])
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            state_dependent_std=False,
            const_std=config['const_std'],
            encoder=None,
        )

        network_info = dict(
            classifier=(classifier_def, (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Default classifier-AWR config (actor part identical to IQL defaults)."""
    config = ml_collections.ConfigDict(
        dict(
            agent_name='classifier_awr',
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            const_std=True,
            classifier_hidden=256,
            alpha=10.0,
        )
    )
    return config
