"""Train / save / load IQL for BoN scoring."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import flax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

from bon_sampling.iql.agent import IQLAgent, get_config
from bon_sampling.iql.dataset import IQLDataset


def save_iql(path: str, agent: IQLAgent, obs_dim: int, act_dim: int, chunk_size: int, step: int):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(
            {
                'mode': 'iql',
                'step': step,
                'agent': flax.serialization.to_state_dict(agent),
                'config': dict(agent.config),
                'obs_dim': obs_dim,
                'act_dim': act_dim,
                'chunk_size': chunk_size,
            },
            f,
        )


def load_iql(path: str) -> tuple[IQLAgent, dict]:
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)
    config = get_config()
    config.update(ckpt['config'])
    agent = IQLAgent.create(
        seed=0,
        ex_observations=np.zeros((1, ckpt['obs_dim']), np.float32),
        ex_actions=np.zeros((1, ckpt['act_dim']), np.float32),
        config=config,
    )
    agent = flax.serialization.from_state_dict(agent, ckpt['agent'])
    return agent, ckpt


def make_select_fn(agent: IQLAgent):
    """BoN selection via min(Q1, Q2)."""
    import jax
    import jax.numpy as jnp

    apply_fn = agent.network.apply_fn

    @jax.jit
    def select_chunk(params, observation, chunks):
        flat = chunks.reshape(chunks.shape[0], -1)
        obs = jnp.broadcast_to(observation[None], (flat.shape[0], observation.shape[0]))
        q1, q2 = apply_fn({'params': params}, obs, actions=flat, name='critic')
        return jnp.argmax(jnp.minimum(q1, q2))

    return select_chunk


def train_iql(
    data_paths: list[str | Path],
    ckpt_path: str | Path,
    steps: int,
    seed: int = 0,
    batch_size: int | None = None,
    chunk_size: int = 4,
) -> tuple[str, dict]:
    dataset = IQLDataset.from_paths(data_paths)
    if dataset.size == 0:
        raise ValueError('IQL dataset is empty')

    config = get_config()
    if batch_size is not None:
        config.batch_size = batch_size

    obs_dim = dataset.data['observations'].shape[1]
    act_dim = dataset.data['actions'].shape[1]
    agent = IQLAgent.create(
        seed=seed,
        ex_observations=dataset.data['observations'][:1],
        ex_actions=dataset.data['actions'][:1],
        config=config,
    )

    rng = np.random.default_rng(seed)
    info = {}
    for step in trange(1, steps + 1, desc='train iql'):
        batch = {k: jnp.asarray(v) for k, v in dataset.sample(config.batch_size, rng).items()}
        agent, info = agent.update(batch)

    # Temp file for BoN workers only (overwritten each round; not archived).
    save_iql(str(ckpt_path), agent, obs_dim, act_dim, chunk_size, steps)
    metrics = {}
    for k, v in info.items():
        try:
            metrics[k] = float(v)
        except (TypeError, ValueError):
            pass
    return str(ckpt_path), metrics
