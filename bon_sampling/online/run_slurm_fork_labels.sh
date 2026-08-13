#!/bin/bash
#SBATCH --job-name=bon_fork_labels
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-29
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Shared success/failure warm-start, then fork finetune labels (success vs oracle).
# Array: task_id {1..5} x seed {0,10,100} x tau {5,7} = 30 runs.
# Each job runs --finetune_label both (warmstart once, then both forks).

source ~/.bashrc
conda activate ogbench

export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_DIR="${REPO_DIR:-$HOME/ogbench}"
CHECKPOINT="${CHECKPOINT:-flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl}"

ROUNDS="${ROUNDS:-10}"
EPISODES="${EPISODES:-100}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
FINETUNE_STEPS="${FINETUNE_STEPS:-2000}"
NUM_WORKERS="${NUM_WORKERS:-10}"
BON_N="${BON_N:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-bon-fork-labels}"

TASKS=(1 2 3 4 5)
SEEDS=(0 10 100)
TAUS=(5 7)
N_SEEDS=${#SEEDS[@]}
N_TAUS=${#TAUS[@]}
N_PER_TASK=$((N_SEEDS * N_TAUS))

TASK_ID="${TASKS[$((SLURM_ARRAY_TASK_ID / N_PER_TASK))]}"
REM=$((SLURM_ARRAY_TASK_ID % N_PER_TASK))
SEED="${SEEDS[$((REM / N_TAUS))]}"
TAU="${TAUS[$((REM % N_TAUS))]}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

run_name="fork_task${TASK_ID}_tau${TAU}_seed${SEED}_r${ROUNDS}"
safe_run_name="${run_name//./p}"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "task_id = ${TASK_ID}  seed = ${SEED}  tau = ${TAU}"
echo "Launching ${run_name}"

python bon_sampling/online/online_fork_labels.py \
  --checkpoint "${CHECKPOINT}" \
  --phase all \
  --finetune_label both \
  --task_id "${TASK_ID}" \
  --seed "${SEED}" \
  --tau "${TAU}" \
  --rounds "${ROUNDS}" \
  --episodes_per_round "${EPISODES}" \
  --train_steps "${TRAIN_STEPS}" \
  --finetune_steps "${FINETUNE_STEPS}" \
  --num_workers "${NUM_WORKERS}" \
  --bon_n "${BON_N}" \
  --device auto \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${run_name}" \
  --wandb_mode online \
  > "logs/${safe_run_name}_${SLURM_JOB_ID}.out" \
  2> "logs/${safe_run_name}_${SLURM_JOB_ID}.err"

echo "Finished ${run_name}"
