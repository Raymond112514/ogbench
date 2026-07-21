"""Classifier-advantage AWR agent — same joint-update recipe as awr.iql.agent.IQLAgent.

The only difference from IQL is the advantage source and its network: a progress
classifier (BCE on oracle-progress labels) replaces the V/Q value networks, and its
logit (analogous to Q - V) is used as the AWR advantage. The actor is updated every
step; the classifier can be updated less often via `update(..., update_classifier=)`.
"""

from __future__ import annotations

from functools import partial
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
    """Progress-classifier advantage + AWR actor."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def classifier_loss(self, batch, grad_params):
        """BCE loss on progress labels."""
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

    def classifier_metrics(self, batch):
        """Classifier metrics with stop-gradient params (for actor-only steps / val)."""
        logits = self.network.select('classifier')(batch['observations'], batch['actions'])
        loss = optax.sigmoid_binary_cross_entropy(logits, batch['labels']).mean()
        pred = (logits > 0).astype(jnp.float32)
        return {
            'classifier_loss': loss,
            'accuracy': jnp.mean(pred == batch['labels']),
            'logit_mean': logits.mean(),
            'label_mean': batch['labels'].mean(),
        }

    def actor_loss(self, batch, grad_params):
        """AWR actor loss; advantage = classifier logit (stop-gradient, like IQL's Q - V)."""
        adv = jax.lax.stop_gradient(
            self.network.select('classifier')(batch['observations'], batch['actions'])
        )
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
            'weight_mean': exp_a.mean(),
        }

    @partial(jax.jit, static_argnames=('update_classifier',))
    def update(self, batch, update_classifier: bool = True):
        """Gradient step. Actor always; classifier only if update_classifier=True."""

        def loss_fn(grad_params):
            info = {}
            actor_loss, actor_info = self.actor_loss(batch, grad_params)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v

            if update_classifier:
                classifier_loss, classifier_info = self.classifier_loss(batch, grad_params)
                for k, v in classifier_info.items():
                    info[f'classifier/{k}'] = v
                return classifier_loss + actor_loss, info

            for k, v in self.classifier_metrics(batch).items():
                info[f'classifier/{k}'] = v
            return actor_loss, info

        grads, info = jax.grad(loss_fn, has_aux=True)(self.network.params)
        if not update_classifier:
            # Zero classifier grads so Adam does not coast on stale moments.
            grads = {
                **grads,
                'modules_classifier': jax.tree_util.tree_map(jnp.zeros_like, grads['modules_classifier']),
            }

        updates, new_opt_state = self.network.tx.update(grads, self.network.opt_state, self.network.params)
        if not update_classifier:
            updates = {
                **updates,
                'modules_classifier': jax.tree_util.tree_map(jnp.zeros_like, updates['modules_classifier']),
            }
        new_params = optax.apply_updates(self.network.params, updates)
        new_network = self.network.replace(
            step=self.network.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )
        info['classifier/updated'] = jnp.asarray(1.0 if update_classifier else 0.0)
        info['grad/norm'] = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
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
