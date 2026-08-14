#!/bin/bash
#SBATCH --job-name=bon_bin_clf
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-37
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Softmax Δ-bin classifier BoN, ranked by E[ℓ].
# Configs: task_id {1..5} x seed {0,10,100} x num_bins {3,5,7,9,11} = 75.
# Each node runs two configs concurrently on one GPU (38 array tasks).

source ~/.bashrc
conda activate ogbench

export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

REPO_DIR="${REPO_DIR:-$HOME/ogbench}"
CHECKPOINT="${CHECKPOINT:-flow_bc/checkpoints/cube_single_gcbc_ac10/best.pkl}"

ROUNDS="${ROUNDS:-10}"
EPISODES="${EPISODES:-100}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
NUM_WORKERS="${NUM_WORKERS:-10}"
BON_N="${BON_N:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-ogbench_bon_sweep_bins}"

TASKS=(1 2 3 4 5)
SEEDS=(0 10 100)
BINS=(3 5 7 9 11)
N_SEEDS=${#SEEDS[@]}
N_BINS=${#BINS[@]}
N_PER_TASK=$((N_SEEDS * N_BINS))
N_CONFIGS=$((${#TASKS[@]} * N_PER_TASK))

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

run_one() {
  local cfg_id="$1"
  local TASK_ID SEED NUM_BINS REM run_name safe_run_name

  TASK_ID="${TASKS[$((cfg_id / N_PER_TASK))]}"
  REM=$((cfg_id % N_PER_TASK))
  SEED="${SEEDS[$((REM / N_BINS))]}"
  NUM_BINS="${BINS[$((REM % N_BINS))]}"

  run_name="binclf_task${TASK_ID}_w${NUM_BINS}"
  safe_run_name="binclf_task${TASK_ID}_w${NUM_BINS}_seed${SEED}"

  echo "cfg ${cfg_id}: task_id=${TASK_ID} seed=${SEED} num_bins=${NUM_BINS}"
  echo "Launching ${run_name}"

  python bon_sampling/online/online_bon.py \
    --checkpoint "${CHECKPOINT}" \
    --method bin_classifier \
    --num_bins "${NUM_BINS}" \
    --task_id "${TASK_ID}" \
    --seed "${SEED}" \
    --rounds "${ROUNDS}" \
    --episodes_per_round "${EPISODES}" \
    --train_steps "${TRAIN_STEPS}" \
    --num_workers "${NUM_WORKERS}" \
    --bon_n "${BON_N}" \
    --device auto \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_name "${run_name}" \
    --wandb_mode online \
    > "logs/${safe_run_name}_${SLURM_JOB_ID}.out" \
    2> "logs/${safe_run_name}_${SLURM_JOB_ID}.err"
  echo "Finished ${run_name}"
}

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"

pids=()
for offset in 0 1; do
  cfg_id=$((SLURM_ARRAY_TASK_ID * 2 + offset))
  if (( cfg_id < N_CONFIGS )); then
    run_one "${cfg_id}" &
    pids+=($!)
  fi
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
