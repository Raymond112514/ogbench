#!/bin/bash
#SBATCH --job-name=awr_offline_classifier
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Offline classifier-advantage AWR (awr/offline.py --advantage classifier): same joint
# update recipe as offline IQL, but the advantage comes from a progress classifier trained
# on oracle-progress labels instead of a learned V/Q pair.
# Requires a *contiguous* oracle-annotated npz (--data_percent 100; see annotate_ogbench).
# Array: alpha {0.5, 1, 3, 10} x seed {0, 10, 100, 1000} = 16 runs.

source ~/.bashrc
conda activate ogbench

export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_DIR="${REPO_DIR:-$HOME/ogbench}"
TASK_ID="${TASK_ID:-1}"
ENV_NAME="${ENV_NAME:-cube-single-play-singletask-task${TASK_ID}-v0}"
ANNOTATED_PATH="${ANNOTATED_PATH:-/global/home/users/r112358/ogbench/awr/data/cube_single_play_task1_oracle.npz}"

TRAIN_STEPS="${TRAIN_STEPS:-200000}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"   # progress label is s_{t+chunk_size} vs. s_t
CLASSIFIER_HIDDEN="${CLASSIFIER_HIDDEN:-256}"
DATA_PERCENT="${DATA_PERCENT:-10}"
VAL_RATIO="${VAL_RATIO:-0.1}"
CLASSIFIER_UPDATE_EVERY="${CLASSIFIER_UPDATE_EVERY:-10}"
WANDB_PROJECT="${WANDB_PROJECT:-awr-offline}"

ALPHAS=(0.5 1 3 10)
SEEDS=(0 10 100 1000)
ALPHA="${ALPHAS[$((SLURM_ARRAY_TASK_ID / 4))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 4))]}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

run_name="offline_classifier_${ENV_NAME}_alpha${ALPHA}_seed${SEED}_chunk${CHUNK_SIZE}_data${DATA_PERCENT}pct_clfevery${CLASSIFIER_UPDATE_EVERY}_steps${TRAIN_STEPS}"
safe_run_name="${run_name//./p}"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "env = ${ENV_NAME}"
echo "annotated_path = ${ANNOTATED_PATH}"
echo "alpha = ${ALPHA}"
echo "seed = ${SEED}"
echo "chunk_size = ${CHUNK_SIZE}"
echo "data_percent = ${DATA_PERCENT}"
echo "val_ratio = ${VAL_RATIO}"
echo "classifier_update_every = ${CLASSIFIER_UPDATE_EVERY}"
echo "Launching ${run_name}"

python awr/offline.py \
  --env_name "${ENV_NAME}" \
  --advantage classifier \
  --annotated_path "${ANNOTATED_PATH}" \
  --train_steps "${TRAIN_STEPS}" \
  --log_interval "${LOG_INTERVAL}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --eval_episodes "${EVAL_EPISODES}" \
  --alpha "${ALPHA}" \
  --seed "${SEED}" \
  --classifier_hidden "${CLASSIFIER_HIDDEN}" \
  --val_ratio "${VAL_RATIO}" \
  --classifier_update_every "${CLASSIFIER_UPDATE_EVERY}" \
  --data_percent "${DATA_PERCENT}" \
  --chunk_size "${CHUNK_SIZE}" \
  --device auto \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${run_name}" \
  --wandb_mode online \
  > "logs/${safe_run_name}_${SLURM_JOB_ID}.out" \
  2> "logs/${safe_run_name}_${SLURM_JOB_ID}.err"

echo "Finished ${run_name}"
