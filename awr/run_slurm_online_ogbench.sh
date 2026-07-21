#!/bin/bash
#SBATCH --job-name=awr_online_ogbench_iql
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Online IQL AWR on OGBench (awr/online_ogbench.py --advantage iql): warm-start on the
# play dataset, then collect / retrain rounds.
# Array: task_id {1, 2} x seed {0, 10, 100, 1000} = 8 runs.

source ~/.bashrc
conda activate ogbench

export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_DIR="${REPO_DIR:-$HOME/ogbench}"

DATA_PERCENT="${DATA_PERCENT:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"
WARM_START_STEPS="${WARM_START_STEPS:-20000}"
MAX_ROUNDS="${MAX_ROUNDS:-50}"
EPISODES_PER_ROUND="${EPISODES_PER_ROUND:-100}"
IQL_STEPS_PER_ROUND="${IQL_STEPS_PER_ROUND:-2000}"
POLICY_STEPS_PER_ROUND="${POLICY_STEPS_PER_ROUND:-2000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
ALPHA="${ALPHA:-1.0}"   # FQL AWR default
LOG_INTERVAL="${LOG_INTERVAL:-1000}"
WANDB_PROJECT="${WANDB_PROJECT:-awr-online-ogbench}"

TASKS=(1 2)
SEEDS=(0 10 100 1000)
TASK_ID="${TASKS[$((SLURM_ARRAY_TASK_ID / 4))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 4))]}"
ENV_NAME="cube-single-play-singletask-task${TASK_ID}-v0"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

run_name="online_iql_${ENV_NAME}_seed${SEED}_chunk${CHUNK_SIZE}_data${DATA_PERCENT}pct_warm${WARM_START_STEPS}_r${MAX_ROUNDS}"
safe_run_name="${run_name//./p}"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "task_id = ${TASK_ID}"
echo "env = ${ENV_NAME}"
echo "seed = ${SEED}"
echo "data_percent = ${DATA_PERCENT}"
echo "Launching ${run_name}"

python awr/online_ogbench.py \
  --env_name "${ENV_NAME}" \
  --advantage iql \
  --chunk_size "${CHUNK_SIZE}" \
  --data_percent "${DATA_PERCENT}" \
  --warm_start_steps "${WARM_START_STEPS}" \
  --max_rounds "${MAX_ROUNDS}" \
  --episodes_per_round "${EPISODES_PER_ROUND}" \
  --iql_steps_per_round "${IQL_STEPS_PER_ROUND}" \
  --policy_steps_per_round "${POLICY_STEPS_PER_ROUND}" \
  --eval_episodes "${EVAL_EPISODES}" \
  --alpha "${ALPHA}" \
  --seed "${SEED}" \
  --log_interval "${LOG_INTERVAL}" \
  --device auto \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${run_name}" \
  --wandb_mode online \
  > "logs/${safe_run_name}_${SLURM_JOB_ID}.out" \
  2> "logs/${safe_run_name}_${SLURM_JOB_ID}.err"

echo "Finished ${run_name}"
