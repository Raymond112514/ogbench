#!/bin/bash
#SBATCH --job-name=awr_online_sweep
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Online AWR: advantages {iql, classifier} x task_id {1, 2, 3}.

source ~/.bashrc
conda activate ogbench

export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_DIR="${REPO_DIR:-$HOME/ogbench}"
CHECKPOINT="${CHECKPOINT:-flow_bc/checkpoints/cube_single_gcbc/best.pkl}"
TRAIN_STEPS=2000
AWR_EPOCHS=10
ROUNDS=30
EPISODES=100
NUM_WORKERS=10
WANDB_PROJECT="${WANDB_PROJECT:-awr-online}"

COMBOS=(
  "iql 1"
  "iql 2"
  "iql 3"
  "classifier 1"
  "classifier 2"
  "classifier 3"
)
read -r ADVANTAGE TASK_ID <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

run_name="awr_${ADVANTAGE}_task${TASK_ID}_steps${TRAIN_STEPS}_ep${AWR_EPOCHS}_r${ROUNDS}"
safe_run_name="${run_name//\//_}"

echo "Launching ${run_name}"

python awr/online.py \
  --checkpoint "${CHECKPOINT}" \
  --advantage "${ADVANTAGE}" \
  --task_id "${TASK_ID}" \
  --rounds "${ROUNDS}" \
  --episodes_per_round "${EPISODES}" \
  --num_workers "${NUM_WORKERS}" \
  --train_steps "${TRAIN_STEPS}" \
  --awr_epochs "${AWR_EPOCHS}" \
  --device auto \
  --output_dir "awr/data/online/${safe_run_name}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${run_name}" \
  --wandb_mode online \
  > "logs/${safe_run_name}_${SLURM_JOB_ID}.out" \
  2> "logs/${safe_run_name}_${SLURM_JOB_ID}.err"

echo "Finished ${run_name}"
