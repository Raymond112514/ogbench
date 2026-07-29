#!/bin/bash
#SBATCH --job-name=sft_success_clf
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# SFT filtered BC using thresholded success/failure classifier feedback.
# Array: task_id {1,2,3,4,5} x seed {0,10,100} = 15 runs.

source ~/.bashrc
conda activate ogbench

export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_DIR="${REPO_DIR:-$HOME/ogbench}"
POLICY_CKPT="${POLICY_CKPT:-flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl}"
ROUNDS="${ROUNDS:-50}"
EPISODES="${EPISODES:-100}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
CLASSIFIER_STEPS="${CLASSIFIER_STEPS:-2000}"
CLASSIFIER_TAU="${CLASSIFIER_TAU:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-10}"
WANDB_PROJECT="${WANDB_PROJECT:-sft-success-classifier}"

TASKS=(1 2 3 4 5)
SEEDS=(0 10 100)
TASK_ID="${TASKS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

run_name="sft_success_clf_task${TASK_ID}_thr${CLASSIFIER_TAU}_seed${SEED}"
safe_run_name="${run_name//./p}"

python sft/online.py \
  --policy_ckpt "${POLICY_CKPT}" \
  --task_id "${TASK_ID}" \
  --rounds "${ROUNDS}" \
  --episodes_per_round "${EPISODES}" \
  --eval_episodes "${EVAL_EPISODES}" \
  --train_steps "${TRAIN_STEPS}" \
  --num_workers "${NUM_WORKERS}" \
  --feedback success_classifier \
  --classifier_tau "${CLASSIFIER_TAU}" \
  --classifier_steps "${CLASSIFIER_STEPS}" \
  --seed "${SEED}" \
  --device auto \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${run_name}" \
  --wandb_mode online \
  > "logs/${safe_run_name}_${SLURM_JOB_ID}.out" \
  2> "logs/${safe_run_name}_${SLURM_JOB_ID}.err"
