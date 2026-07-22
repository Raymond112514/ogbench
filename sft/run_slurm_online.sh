#!/bin/bash
#SBATCH --job-name=sft_online
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Online SFT / binary-advantage AWR starting from GCBC (sft/online.py).
# Array: task_id {1,2,3,4,5}.

source ~/.bashrc
conda activate ogbench

export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_DIR="${REPO_DIR:-$HOME/ogbench}"
POLICY_CKPT="${POLICY_CKPT:-flow_bc/checkpoints/cube_single_gcbc/best.pkl}"

ROUNDS="${ROUNDS:-50}"
EPISODES_PER_ROUND="${EPISODES_PER_ROUND:-100}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
ALPHA="${ALPHA:-10.0}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SEED="${SEED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-sft-online}"

TASKS=(1 2 3 4 5)
TASK_ID="${TASKS[${SLURM_ARRAY_TASK_ID}]}"

mkdir -p "${REPO_DIR}/logs" "${REPO_DIR}/sft/checkpoints"
cd "${REPO_DIR}"

run_name="sft_task${TASK_ID}_r${ROUNDS}_ep${EPISODES_PER_ROUND}_alpha${ALPHA}_seed${SEED}"
safe_run_name="${run_name//./p}"
output_dir="sft/checkpoints/task${TASK_ID}_seed${SEED}"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "task_id = ${TASK_ID}"
echo "policy_ckpt = ${POLICY_CKPT}"
echo "rounds = ${ROUNDS}"
echo "Launching ${run_name}"

python sft/online.py \
  --policy_ckpt "${POLICY_CKPT}" \
  --task_id "${TASK_ID}" \
  --rounds "${ROUNDS}" \
  --episodes_per_round "${EPISODES_PER_ROUND}" \
  --eval_episodes "${EVAL_EPISODES}" \
  --train_steps "${TRAIN_STEPS}" \
  --alpha "${ALPHA}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --output_dir "${output_dir}" \
  --device auto \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${run_name}" \
  --wandb_mode online \
  > "logs/${safe_run_name}_${SLURM_JOB_ID}.out" \
  2> "logs/${safe_run_name}_${SLURM_JOB_ID}.err"

echo "Finished ${run_name}"
