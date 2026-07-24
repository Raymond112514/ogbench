#!/bin/bash
#SBATCH --job-name=sft_filtered_bc_offline
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Offline filtered BC: collect once, oracle-filter, train with periodic eval.
# Array: task_id {1,2,3,4,5} x seed {0,10,100,1000} = 20 runs.

source ~/.bashrc
conda activate ogbench

export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_DIR="${REPO_DIR:-$HOME/ogbench}"
POLICY_CKPT="${POLICY_CKPT:-flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl}"

NUM_EPISODES="${NUM_EPISODES:-10000}"
TRAIN_STEPS="${TRAIN_STEPS:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
COLLECT_WORKERS="${COLLECT_WORKERS:-10}"
NUM_WORKERS="${NUM_WORKERS:-10}"
WANDB_PROJECT="${WANDB_PROJECT:-sft-filtered-bc-offline}"

TASKS=(1 2 3 4 5)
SEEDS=(0 10 100 1000)
TASK_ID="${TASKS[$((SLURM_ARRAY_TASK_ID / 4))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 4))]}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

run_name="filtered_bc_offline_task${TASK_ID}_ep${NUM_EPISODES}_steps${TRAIN_STEPS}_seed${SEED}"
safe_run_name="${run_name//./p}"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "task_id = ${TASK_ID}"
echo "seed = ${SEED}"
echo "policy_ckpt = ${POLICY_CKPT}"
echo "num_episodes = ${NUM_EPISODES}"
echo "Launching ${run_name}"

python sft/offline.py \
  --policy_ckpt "${POLICY_CKPT}" \
  --task_id "${TASK_ID}" \
  --num_episodes "${NUM_EPISODES}" \
  --collect_workers "${COLLECT_WORKERS}" \
  --train_steps "${TRAIN_STEPS}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --eval_episodes "${EVAL_EPISODES}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --device auto \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${run_name}" \
  --wandb_mode online \
  > "logs/${safe_run_name}_${SLURM_JOB_ID}.out" \
  2> "logs/${safe_run_name}_${SLURM_JOB_ID}.err"

echo "Finished ${run_name}"
