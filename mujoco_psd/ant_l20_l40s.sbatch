#!/usr/bin/env bash
#SBATCH --job-name=ant-psd-l20
#SBATCH --partition=gpu
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=ant-psd-l20-%j.log

set -euo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_ROOT}}"
cd "$PROJECT_ROOT"
PROJECT_ROOT="$PWD"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/mujoco_psd/runs}"

# One job uses one direct seed. The array task ID, if supplied, is the seed.
TASK_ID="${SLURM_ARRAY_TASK_ID:-none}"
SEED="${SLURM_ARRAY_TASK_ID:-${SEED:-0}}"
CONDITION="ant-psd-intrinsic-l20"

EPOCHS="${EPOCHS:-2500}"
DEVICE="cuda"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.oscar-venv/bin/python}"
RESUME="${RESUME:-0}"

LOG_GROUP="${LOG_GROUP:-psd-ant-figure5}"
RUN_NAME="${RUN_NAME:-${CONDITION}__seed${SEED}__epochs${EPOCHS}}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"

COMMAND=(
    "$PYTHON_BIN" -B -u "$PROJECT_ROOT/mujoco_psd/train_ant.py"
    --epochs "$EPOCHS" --seed "$SEED" --device "$DEVICE" --out "$RUN_DIR"
)
if [[ "$RESUME" == "1" ]]; then
    COMMAND+=(--resume)
fi
EVALUATION_COMMAND=(
    "$PYTHON_BIN" -B -u "$PROJECT_ROOT/mujoco_psd/evaluate_ant.py"
    "$RUN_DIR/checkpoint.pt" --device "$DEVICE" --out "$RUN_DIR/evaluation"
)

printf -v FINAL_COMMAND '%q ' "${COMMAND[@]}"
printf -v FINAL_EVALUATION '%q ' "${EVALUATION_COMMAND[@]}"
printf 'Project root: %s\nCondition: %s\nSeed: %s\nTask id: %s\n' "$PROJECT_ROOT" "$CONDITION" "$SEED" "$TASK_ID"
printf 'Log group: %s\nRun name: %s\nRun directory: %s\n' "$LOG_GROUP" "$RUN_NAME" "$RUN_DIR"
printf 'Final command: %s\nEvaluation command: %s\n' "$FINAL_COMMAND" "$FINAL_EVALUATION"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo 'DRY_RUN=1'
    exit 0
fi

mkdir -p -- "$RUN_DIR"
unset LD_LIBRARY_PATH VIRTUAL_ENV
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg
"$PYTHON_BIN" -B -c 'import torch; actual = torch.cuda.get_device_name(0); print("Allocated GPU:", actual); assert "L40S" in actual.upper(), actual'
"${COMMAND[@]}"
"${EVALUATION_COMMAND[@]}"
