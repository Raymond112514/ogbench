#!/bin/bash
#SBATCH --job-name=online_bon_sweep
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Online BoN: methods {classifier, iql} x task_id {1, 2, 3}.
# Fit steps=2000, rounds=30.

# --- env ---
source ~/.bashrc
conda activate ogbench

export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- config ---
REPO_DIR="${REPO_DIR:-$HOME/ogbench}"
CHECKPOINT="${CHECKPOINT:-flow_bc/checkpoints/cube_single_gcbc/best.pkl}"
ENV_NAME="${ENV_NAME:-cube-single-v0}"
TRAIN_STEPS=2000
ROUNDS=30
EPISODES=100
NUM_WORKERS=10
BON_N=8
WANDB_PROJECT="${WANDB_PROJECT:-bon-online}"

# Flat sweep: (method, task_id)
COMBOS=(
  "classifier 1"
  "classifier 2"
  "classifier 3"
  "iql 1"
  "iql 2"
  "iql 3"
)
read -r METHOD TASK_ID <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "method = ${METHOD}, task_id = ${TASK_ID}"
echo "train_steps = ${TRAIN_STEPS}, rounds = ${ROUNDS}"

run_name="${METHOD}_task${TASK_ID}"
safe_run_name="${run_name//\//_}"

echo "Launching ${run_name}"

python bon_sampling/online/online_bon.py \
  --checkpoint "${CHECKPOINT}" \
  --env_name "${ENV_NAME}" \
  --method "${METHOD}" \
  --task_id "${TASK_ID}" \
  --rounds "${ROUNDS}" \
  --episodes_per_round "${EPISODES}" \
  --num_workers "${NUM_WORKERS}" \
  --bon_n "${BON_N}" \
  --train_steps "${TRAIN_STEPS}" \
  --expectile 0.9 \
  --device auto \
  --output_dir "bon_sampling/data/online/${safe_run_name}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${run_name}" \
  --wandb_mode online \
  > "logs/${safe_run_name}_${SLURM_JOB_ID}.out" \
  2> "logs/${safe_run_name}_${SLURM_JOB_ID}.err"

echo "Finished array task ${SLURM_ARRAY_TASK_ID} (method=${METHOD}, task_id=${TASK_ID})"
