#!/bin/bash
#SBATCH --job-name=awr_oracle_delta
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-44
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Online AWR with unbinned oracle Δ = d(s)-d(s') as advantage.
# Array: task_id {1..5} x alpha {0.1,0.3,1.0} x seed {0,10,100} = 45 runs.

source ~/.bashrc
conda activate ogbench

export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_DIR="${REPO_DIR:-$HOME/ogbench}"
CHECKPOINT="${CHECKPOINT:-flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl}"

TRAIN_STEPS="${TRAIN_STEPS:-2000}"
AWR_EPOCHS="${AWR_EPOCHS:-10}"
ROUNDS="${ROUNDS:-10}"
INITIAL_EPISODES="${INITIAL_EPISODES:-1000}"
EPISODES_PER_ROUND="${EPISODES_PER_ROUND:-100}"
NUM_WORKERS="${NUM_WORKERS:-10}"
WANDB_PROJECT="${WANDB_PROJECT:-awr-oracle-delta}"

TASKS=(1 2 3 4 5)
ALPHAS=(0.1 0.3 1.0)
SEEDS=(0 10 100)
N_ALPHAS=${#ALPHAS[@]}
N_SEEDS=${#SEEDS[@]}
N_PER_TASK=$((N_ALPHAS * N_SEEDS))

TASK_ID="${TASKS[$((SLURM_ARRAY_TASK_ID / N_PER_TASK))]}"
REM=$((SLURM_ARRAY_TASK_ID % N_PER_TASK))
ALPHA="${ALPHAS[$((REM / N_SEEDS))]}"
SEED="${SEEDS[$((REM % N_SEEDS))]}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

run_name="awr_oracle_delta_task${TASK_ID}_seed${SEED}_a${ALPHA}_r${ROUNDS}"
safe_run_name="${run_name//./p}"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "task_id = ${TASK_ID}  alpha = ${ALPHA}  seed = ${SEED}"
echo "Launching ${run_name}"

python awr/online.py \
  --checkpoint "${CHECKPOINT}" \
  --advantage oracle_delta \
  --task_id "${TASK_ID}" \
  --seed "${SEED}" \
  --rounds "${ROUNDS}" \
  --initial_episodes "${INITIAL_EPISODES}" \
  --episodes_per_round "${EPISODES_PER_ROUND}" \
  --num_workers "${NUM_WORKERS}" \
  --train_steps "${TRAIN_STEPS}" \
  --awr_epochs "${AWR_EPOCHS}" \
  --alpha "${ALPHA}" \
  --device auto \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${run_name}" \
  --wandb_mode online \
  > "logs/${safe_run_name}_${SLURM_JOB_ID}.out" \
  2> "logs/${safe_run_name}_${SLURM_JOB_ID}.err"

echo "Finished ${run_name}"
