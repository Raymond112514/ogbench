#!/bin/bash
#SBATCH --job-name=sft_tau
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-44
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Online filtered BC tau sweep.
# Array: task_id {1,2,3,4,5} x seed {0,10,100} x tau {3,5,7} = 45 runs.
# With H=10 (ac10): thresholds are H-tau = {7,5,3}.

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
EPISODES_PER_ROUND="${EPISODES_PER_ROUND:-100}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
NUM_WORKERS="${NUM_WORKERS:-10}"
WANDB_PROJECT="${WANDB_PROJECT:-sft-filtered-bc-tau}"

TASKS=(1 2 3 4 5)
SEEDS=(0 10 100)
TAUS=(3 5 7)
N_SEEDS=${#SEEDS[@]}
N_TAUS=${#TAUS[@]}

TASK_ID="${TASKS[$((SLURM_ARRAY_TASK_ID / (N_SEEDS * N_TAUS)))]}"
SEED="${SEEDS[$(((SLURM_ARRAY_TASK_ID / N_TAUS) % N_SEEDS))]}"
TAU="${TAUS[$((SLURM_ARRAY_TASK_ID % N_TAUS))]}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

run_name="filtered_bc_task${TASK_ID}_tau${TAU}_r${ROUNDS}_ep${EPISODES_PER_ROUND}_seed${SEED}"
safe_run_name="${run_name//./p}"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "task_id = ${TASK_ID}"
echo "seed = ${SEED}"
echo "tau = ${TAU}"
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
  --num_workers "${NUM_WORKERS}" \
  --tau "${TAU}" \
  --seed "${SEED}" \
  --device auto \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${run_name}" \
  --wandb_mode online \
  > "logs/${safe_run_name}_${SLURM_JOB_ID}.out" \
  2> "logs/${safe_run_name}_${SLURM_JOB_ID}.err"

echo "Finished ${run_name}"
