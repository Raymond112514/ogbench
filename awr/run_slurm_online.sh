#!/bin/bash
#SBATCH --job-name=awr_online_sweep
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Online AWR (GCBC bootstrap): advantages {iql, classifier} x seed {0, 1, 2} on task 1.
# Initial: 1000 GCBC episodes -> fit advantage -> extract AWR.
# Then 30 rounds: 100 episodes from extracted policy -> fit -> extract.

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
INITIAL_EPISODES=1000
EPISODES_PER_ROUND=100
NUM_WORKERS=10
TASK_ID=1
ALPHA="${ALPHA:-10.0}"
WANDB_PROJECT="${WANDB_PROJECT:-awr-online}"

COMBOS=(
  "iql 0"
  "iql 1"
  "iql 2"
  "classifier 0"
  "classifier 1"
  "classifier 2"
)
read -r ADVANTAGE SEED <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

run_name="awr_${ADVANTAGE}_task${TASK_ID}_seed${SEED}_a${ALPHA}_init${INITIAL_EPISODES}_ep${EPISODES_PER_ROUND}_r${ROUNDS}_fit${TRAIN_STEPS}_awr${AWR_EPOCHS}"
safe_run_name="${run_name//./p}"

echo "Launching ${run_name}"

python awr/online.py \
  --checkpoint "${CHECKPOINT}" \
  --advantage "${ADVANTAGE}" \
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
