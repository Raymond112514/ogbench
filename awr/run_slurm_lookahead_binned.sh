#!/bin/bash
#SBATCH --job-name=lookahead_binned
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Signed Δ-bin oracle-lookahead (wandb project=ogbench_bon).
# Array: task_id {1,2,3,4,5} x num_bins {3,5,7,9} = 20 runs.
# For ac10 (H=10): splits [-10,10] into w equal pieces; end bins absorb tails.

source ~/.bashrc
conda activate ogbench

export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_DIR="${REPO_DIR:-$HOME/ogbench}"
POLICY_CKPT="${POLICY_CKPT:-flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl}"

NUM_EPISODES="${NUM_EPISODES:-100}"
NUM_WORKERS="${NUM_WORKERS:-10}"
BON_N="${BON_N:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-ogbench_bon}"

TASKS=(1 2 3 4 5)
NUM_BINS_LIST=(3 5 7 9)
N_BINS=${#NUM_BINS_LIST[@]}

TASK_ID="${TASKS[$((SLURM_ARRAY_TASK_ID / N_BINS))]}"
NUM_BINS="${NUM_BINS_LIST[$((SLURM_ARRAY_TASK_ID % N_BINS))]}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

run_name="oracle_binned_task${TASK_ID}_w${NUM_BINS}_k${BON_N}_ep${NUM_EPISODES}"
safe_run_name="${run_name//./p}"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "task_id = ${TASK_ID}"
echo "num_bins = ${NUM_BINS}"
echo "Launching ${run_name}"

python awr/eval_oracle_lookahead.py \
  --policy_ckpt "${POLICY_CKPT}" \
  --task_id "${TASK_ID}" \
  --select binned \
  --num_bins "${NUM_BINS}" \
  --bon_n "${BON_N}" \
  --num_episodes "${NUM_EPISODES}" \
  --num_workers "${NUM_WORKERS}" \
  --fixed_seeds \
  --seed 0 \
  --reset_seed 0 \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${run_name}" \
  --wandb_mode online \
  > "logs/${safe_run_name}_${SLURM_JOB_ID}.out" \
  2> "logs/${safe_run_name}_${SLURM_JOB_ID}.err"

echo "Finished ${run_name}"
