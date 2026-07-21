#!/bin/bash
#SBATCH --job-name=awr_oracle_ogbench
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=72:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Oracle-label full OGBench cube-single-play datasets for singletask 1 and 2.
# (MuJoCo oracle is CPU-bound; GPU is unused but matches co_rail QoS.)
# Outputs can be loaded later without re-running the oracle:
#   awr/data/cube_single_play_task{1,2}_oracle.npz

source ~/.bashrc
conda activate ogbench

export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cpu

REPO_DIR="${REPO_DIR:-$HOME/ogbench}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/awr/data}"
NUM_WORKERS="${NUM_WORKERS:-16}"
DATA_PERCENT="${DATA_PERCENT:-100}"
MAX_ORACLE_STEPS="${MAX_ORACLE_STEPS:-200}"
WANDB_PROJECT="${WANDB_PROJECT:-awr-offline}"  # unused; kept for consistency

TASKS=(1 2)
TASK_ID="${TASKS[$SLURM_ARRAY_TASK_ID]}"
ENV_NAME="cube-single-play-singletask-task${TASK_ID}-v0"
OUTPUT="${OUT_DIR}/cube_single_play_task${TASK_ID}_oracle.npz"

mkdir -p "${REPO_DIR}/logs" "${OUT_DIR}"
cd "${REPO_DIR}"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "env = ${ENV_NAME}"
echo "output = ${OUTPUT}"
echo "data_percent = ${DATA_PERCENT}"
echo "num_workers = ${NUM_WORKERS}"
echo "Launching oracle annotation"

python awr/annotate_ogbench.py \
  --env_name "${ENV_NAME}" \
  --output "${OUTPUT}" \
  --data_percent "${DATA_PERCENT}" \
  --num_workers "${NUM_WORKERS}" \
  --max_oracle_steps "${MAX_ORACLE_STEPS}" \
  --seed 0

echo "Finished ${OUTPUT}"
