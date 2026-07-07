"""Subprocess eval workers: env stepping only (JAX inference stays in parent)."""

from __future__ import annotations

import os
import subprocess

import numpy as np


def count_gpus() -> int:
    try:
        out = subprocess.check_output(['nvidia-smi', '-L'], text=True, stderr=subprocess.DEVNULL)
        return sum(1 for line in out.splitlines() if line.strip().startswith('GPU '))
    except (OSError, subprocess.CalledProcessError):
        return 0


def assign_egl_gpus(
    num_workers: int,
    egl_device: int | None,
    exclude_gpus: tuple[int, ...] | list[int] = (0,),
) -> list[int | None]:
    n = count_gpus()
    if n == 0:
        return [egl_device] * num_workers
    excluded = set(exclude_gpus)
    candidates = [gpu_id for gpu_id in range(n) if gpu_id not in excluded]
    if not candidates:
        candidates = list(range(n))
    if egl_device is not None and egl_device in candidates:
        base = candidates.index(egl_device)
        ordered = candidates[base:] + candidates[:base]
    else:
        ordered = candidates
    return [ordered[i % len(ordered)] for i in range(num_workers)]


def env_worker_loop(
    conn,
    env_name: str,
    task_id: int,
    max_steps: int,
    goal_condition: bool,
    egl_gpu: int | None,
) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['JAX_PLATFORMS'] = 'cpu'
    if egl_gpu is not None:
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(egl_gpu)

    import gymnasium

    import ogbench.manipspace  # noqa: F401

    env = gymnasium.make(env_name)
    try:
        while True:
            msg = conn.recv()
            if msg is None or msg[0] == 'stop':
                break
            if msg[0] != 'episode':
                raise ValueError(f'unexpected message: {msg[0]}')

            record = bool(msg[1])
            ob, info = env.reset(options=dict(task_id=task_id))
            goal = info['goal'] if goal_condition else None
            frames: list[np.ndarray] = []
            if record:
                frames.append(env.render())

            success, steps, done = False, 0, False
            while steps < max_steps and not done:
                conn.send(('chunk_request', np.asarray(ob), np.asarray(goal) if goal is not None else None))
                chunk = conn.recv()
                for action in chunk:
                    if steps >= max_steps:
                        break
                    ob, _, term, trunc, info = env.step(np.clip(action, -1.0, 1.0))
                    if record:
                        frames.append(env.render())
                    steps += 1
                    success = bool(info.get('success', False))
                    if term or trunc:
                        done = True
                        break

            conn.send(('episode_done', success, steps, frames))
    finally:
        env.close()


def _start_env_worker(
    ctx,
    env_name: str,
    task_id: int,
    max_steps: int,
    goal_condition: bool,
    egl_gpu: int | None,
):
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    proc = ctx.Process(
        target=env_worker_loop,
        args=(child_conn, env_name, task_id, max_steps, goal_condition, egl_gpu),
        daemon=True,
    )
    proc.start()
    child_conn.close()
    return parent_conn, proc
