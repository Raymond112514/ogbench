#!/bin/bash
#SBATCH --job-name=online_bon_tau
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

# Online BoN (classifier) tau sweep.
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
CHECKPOINT="${CHECKPOINT:-flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl}"
ENV_NAME="${ENV_NAME:-cube-single-v0}"
METHOD="${METHOD:-classifier}"

ROUNDS="${ROUNDS:-30}"
EPISODES="${EPISODES:-100}"
EVAL_CLF_EPISODES="${EVAL_CLF_EPISODES:-20}"
NUM_WORKERS="${NUM_WORKERS:-10}"
BON_N="${BON_N:-8}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
WANDB_PROJECT="${WANDB_PROJECT:-bon-online-ac10-tau}"

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

run_name="${METHOD}_ac10_task${TASK_ID}_tau${TAU}_r${ROUNDS}_bon${BON_N}_seed${SEED}"
safe_run_name="${run_name//./p}"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "method = ${METHOD}"
echo "task_id = ${TASK_ID}"
echo "seed = ${SEED}"
echo "tau = ${TAU}"
echo "checkpoint = ${CHECKPOINT}"
echo "Launching ${run_name}"

python bon_sampling/online/online_bon.py \
  --checkpoint "${CHECKPOINT}" \
  --env_name "${ENV_NAME}" \
  --method "${METHOD}" \
  --task_id "${TASK_ID}" \
  --rounds "${ROUNDS}" \
  --episodes_per_round "${EPISODES}" \
  --eval_clf_episodes "${EVAL_CLF_EPISODES}" \
  --num_workers "${NUM_WORKERS}" \
  --bon_n "${BON_N}" \
  --train_steps "${TRAIN_STEPS}" \
  --tau "${TAU}" \
  --seed "${SEED}" \
  --device auto \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${run_name}" \
  --wandb_mode online \
  > "logs/${safe_run_name}_${SLURM_JOB_ID}.out" \
  2> "logs/${safe_run_name}_${SLURM_JOB_ID}.err"

echo "Finished ${run_name}"
