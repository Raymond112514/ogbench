"""Checkpoint save/load for flow-BC policies."""

from __future__ import annotations

import pickle
from typing import Any

from flow_bc.model import VelocityNet


def _meta_from_ckpt(ckpt: dict[str, Any]) -> dict[str, Any]:
    chunk_size = int(ckpt['chunk_size'])
    act_dim = int(ckpt['act_dim'])
    hidden_dims = tuple(ckpt['hidden_dims'])
    meta = {
        'obs_dim': int(ckpt['obs_dim']),
        'act_dim': act_dim,
        'chunk_size': chunk_size,
        'hidden_dims': hidden_dims,
        'step': ckpt.get('step'),
        'goal_condition': bool(ckpt.get('goal_condition', True)),
        'advantage_condition': bool(ckpt.get('advantage_condition', False)),
    }
    for key in (
        'env_name',
        'eval_env_name',
        'task_id',
        'task_name',
        'rollout_data',
        'cfg_dropout',
        'advantage_null',
    ):
        if key in ckpt:
            meta[key] = ckpt[key]
    return meta


def read_ckpt_meta(path: str) -> dict[str, Any]:
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)
    return _meta_from_ckpt(ckpt)


def save_ckpt(
    path: str,
    params,
    obs_dim: int,
    act_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] | list[int],
    step: int,
    *,
    goal_condition: bool = True,
    advantage_condition: bool = False,
    extra_meta: dict | None = None,
) -> None:
    import os

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    payload = {
        'params': params,
        'obs_dim': obs_dim,
        'act_dim': act_dim,
        'chunk_size': chunk_size,
        'hidden_dims': list(hidden_dims),
        'step': step,
        'goal_condition': goal_condition,
        'advantage_condition': advantage_condition,
    }
    if extra_meta:
        payload.update(extra_meta)
    with open(path, 'wb') as f:
        pickle.dump(payload, f)


def load_flow_bc(path: str) -> tuple[VelocityNet, Any, dict[str, Any]]:
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)

    meta = _meta_from_ckpt(ckpt)
    model = VelocityNet(
        hidden_dims=meta['hidden_dims'],
        out_dim=meta['chunk_size'] * meta['act_dim'],
    )
    return model, ckpt['params'], meta
