# Flow BC

Rectified-flow behavioral cloning with action chunking for OGBench manipulation tasks.

## Setup

From the `ogbench/` directory:

```bash
pip install -e ".[train]"
```

Set rendering backend for headless machines:

```bash
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0   # GPU used for MuJoCo rendering
```

## Train goal-conditioned BC (GC-BC)

Uses OGBench play datasets (random goals at eval time). Presets configure env, eval env, task, and checkpoint dir.

**cube-single, task 1:**

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  python flow_bc/train.py --preset cube_single_gcbc
```

Checkpoints are written to `flow_bc/checkpoints/<preset_name>/` (`best.pkl`, periodic `step_*.pkl`, `final.pkl`, eval videos under `eval/`).

## Train expert BC (single fixed task)

### 1. Collect expert demonstrations

Runs `CubePlanOracle` on a fixed task until `--num_episodes` successful episodes are saved:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 JAX_PLATFORMS=cpu \
  python flow_bc/collect_expert_demos.py \
  --env_name cube-single-v0 \
  --task_id 1 \
  --num_episodes 100 \
  --output flow_bc/data/expert_task1.npz
```

### 2. Train without goal conditioning

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  python flow_bc/train.py \
  --dataset_npz flow_bc/data/expert_task1.npz \
  --no-goal_condition \
  --eval_env_name cube-single-v0 \
  --task_id 1 \
  --checkpoint_dir flow_bc/checkpoints/cube_single_expert
```

Expert BC requires `--dataset_npz` (play data via `--env_name` is goal-conditioned only).

## Evaluate a checkpoint

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  python flow_bc/eval.py \
  --ckpt flow_bc/checkpoints/cube_single_expert/best.pkl \
  --task_id 1 \
  --num_episodes 100 \
  --num_workers 10 \
  --jax_platform cpu
```

Eval reads `eval_env_name` and `goal_condition` from the checkpoint when present.

## Common flags

| Flag | Default | Description |
|------|---------|-------------|
| `--train_steps` | 100000 | Training steps |
| `--batch_size` | 256 | Batch size |
| `--chunk_size` | 4 | Actions predicted per flow sample |
| `--n_flow_steps` | 10 | Euler steps at inference |
| `--eval_interval` | 5000 | Eval frequency (steps) |
| `--eval_episodes` | 10 | Episodes per eval |
| `--eval_workers` | 0 | Parallel eval envs (0 = one per episode) |
| `--jax_platform` | auto | `auto`, `cpu`, or `gpu` |
| `--jax_device` | None | Pin training to one GPU index |

## Module layout

| File | Role |
|------|------|
| `model.py` | Flow velocity net, training loss, action sampling |
| `dataset.py` | Chunked BC / GC-BC dataset loaders |
| `train.py` | Training loop with periodic eval |
| `eval.py` | Standalone parallel evaluation |
| `eval_worker.py` | Subprocess env workers for training-time eval |
| `checkpoint.py` | Save/load `.pkl` checkpoints |
| `collect_expert_demos.py` | Oracle expert data collection |
