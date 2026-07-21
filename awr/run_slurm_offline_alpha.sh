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
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Offline joint IQL (FQL default hparams). Array: tasks {1,2} x data {25,50,75,100}%.
# Training runs on the allocated GPU (--device auto + --gres=gpu:A5000:1).

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
ALPHA="${ALPHA:-10.0}"   # FQL IQL default
EXPECTILE=0.9
CHUNK_SIZE="${CHUNK_SIZE:-4}"   # 1 = single-step; 4 = action chunks
WANDB_PROJECT="${WANDB_PROJECT:-awr-offline}"

TASKS=(1)
PERCENTS=(25 50 75 100)
TASK_ID="${TASKS[$((SLURM_ARRAY_TASK_ID / 4))]}"
DATA_PERCENT="${PERCENTS[$((SLURM_ARRAY_TASK_ID % 4))]}"
ENV_NAME="cube-single-play-singletask-task${TASK_ID}-v0"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

run_name="offline_iql_${ENV_NAME}_alpha${ALPHA}_chunk${CHUNK_SIZE}_data${DATA_PERCENT}pct_steps${TRAIN_STEPS}"
safe_run_name="${run_name//./p}"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "env = ${ENV_NAME}"
echo "alpha = ${ALPHA}"
echo "chunk_size = ${CHUNK_SIZE}"
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
  --chunk_size "${CHUNK_SIZE}" \
  --device auto \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${run_name}" \
  --wandb_mode online \
  > "logs/${safe_run_name}_${SLURM_JOB_ID}.out" \
  2> "logs/${safe_run_name}_${SLURM_JOB_ID}.err"

echo "Finished ${run_name}"
