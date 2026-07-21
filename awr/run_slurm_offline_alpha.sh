#!/bin/bash
#SBATCH --job-name=awr_offline_iql
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Offline joint IQL on the same OGBench play datasets as FQL.
# Array: tasks {1,2} x alphas {0.3,1,3,10,30}

source ~/.bashrc
conda activate ogbench

export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_DIR="${REPO_DIR:-$HOME/ogbench}"
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EXPECTILE=0.9
DATA_PERCENT="${DATA_PERCENT:-100}"
WANDB_PROJECT="${WANDB_PROJECT:-awr-offline}"

TASKS=(1 2)
ALPHAS=(0.3 1.0 3.0 10.0 30.0)
TASK_ID="${TASKS[$((SLURM_ARRAY_TASK_ID / 5))]}"
ALPHA="${ALPHAS[$((SLURM_ARRAY_TASK_ID % 5))]}"
ENV_NAME="cube-single-play-singletask-task${TASK_ID}-v0"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

run_name="offline_iql_${ENV_NAME}_alpha${ALPHA}_data${DATA_PERCENT}pct_steps${TRAIN_STEPS}"
safe_run_name="${run_name//./p}"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "env = ${ENV_NAME}"
echo "alpha = ${ALPHA}"
echo "data_percent = ${DATA_PERCENT}"
echo "Launching ${run_name}"

python awr/offline.py \
  --env_name "${ENV_NAME}" \
  --train_steps "${TRAIN_STEPS}" \
  --log_interval "${LOG_INTERVAL}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --eval_episodes "${EVAL_EPISODES}" \
  --alpha "${ALPHA}" \
  --expectile "${EXPECTILE}" \
  --data_percent "${DATA_PERCENT}" \
  --device auto \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${run_name}" \
  --wandb_mode online \
  > "logs/${safe_run_name}_${SLURM_JOB_ID}.out" \
  2> "logs/${safe_run_name}_${SLURM_JOB_ID}.err"

echo "Finished ${run_name}"
