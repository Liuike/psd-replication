#!/usr/bin/env bash
#SBATCH --job-name=psd-forward-fig5
#SBATCH --partition=gpu
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=psd-forward-fig5-%j.log

set -euo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_ROOT}}"
cd "$PROJECT_ROOT"
PROJECT_ROOT="$PWD"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/mujoco_psd/runs}"

# Each job selects one robot; an optional array task maps directly to its seed.
TASK_ID="${SLURM_ARRAY_TASK_ID:-none}"
SEED="${SLURM_ARRAY_TASK_ID:-${SEED:-0}}"
ROBOT="${ROBOT:-walker2d}"
EPOCHS="${EPOCHS:-10000}"
REWARD_MODE="intrinsic_velocity"
DEVICE="cuda"
RESUME="${RESUME:-0}"
PRESERVE_PILOT="${PRESERVE_PILOT:-0}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.oscar-venv/bin/python}"

case "$ROBOT" in
    ant)
        TRAINER="train_ant.py"
        EVALUATOR="evaluate_ant_figure5.py"
        PERIODS="10,13,16,20"
        FORWARD_TARGET="0.5"
        ;;
    walker2d)
        TRAINER="train_walker2d.py"
        EVALUATOR="evaluate_walker2d.py"
        PERIODS="20,26,33,40"
        FORWARD_TARGET="1.0"
        ;;
    *)
        printf 'Unknown ROBOT: %s\n' "$ROBOT" >&2
        exit 1
        ;;
esac
CONDITION="${ROBOT}-figure5-intrinsic-velocity"
LOG_GROUP="${LOG_GROUP:-psd-figure5-combined-forward}"
RUN_NAME="${RUN_NAME:-${CONDITION}__seed${SEED}__epochs${EPOCHS}}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"

COMMAND=(
    "$PYTHON_BIN" -B -u "$PROJECT_ROOT/mujoco_psd/$TRAINER"
    --epochs "$EPOCHS" --seed "$SEED" --periods "$PERIODS"
    --device "$DEVICE" --reward-mode "$REWARD_MODE" --out "$RUN_DIR"
)
if [[ "$RESUME" == "1" ]]; then
    COMMAND+=(--resume)
fi
EVALUATION_COMMAND=(
    "$PYTHON_BIN" -B -u "$PROJECT_ROOT/mujoco_psd/$EVALUATOR"
    "$RUN_DIR/checkpoint.pt" --device "$DEVICE" --out "$RUN_DIR/evaluation"
)

printf -v FINAL_COMMAND '%q ' "${COMMAND[@]}"
printf -v FINAL_EVALUATION '%q ' "${EVALUATION_COMMAND[@]}"
printf 'Project root: %s\nCondition: %s\nSeed: %s\nTask id: %s\n' "$PROJECT_ROOT" "$CONDITION" "$SEED" "$TASK_ID"
printf 'Log group: %s\nRun name: %s\nRun directory: %s\n' "$LOG_GROUP" "$RUN_NAME" "$RUN_DIR"
printf 'Reward: r_PSD + min(v_x / %s, 1), unit weights, signed velocity\n' "$FORWARD_TARGET"
printf 'Final command: %s\nEvaluation command: %s\n' "$FINAL_COMMAND" "$FINAL_EVALUATION"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo 'DRY_RUN=1'
    exit 0
fi

mkdir -p -- "$RUN_DIR"
unset LD_LIBRARY_PATH VIRTUAL_ENV
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$RUN_DIR/.matplotlib"
"$PYTHON_BIN" -B -c 'import torch; actual = torch.cuda.get_device_name(0); print("Allocated GPU:", actual); assert "L40S" in actual.upper(), actual'
if [[ "$RESUME" == "1" && "$PRESERVE_PILOT" == "1" ]]; then
    PILOT_ARCHIVE="$RUN_DIR/pilot_epoch0050"
    PILOT_REPORT="${ROBOT}_figure5_evaluation.json"
    if [[ ! -d "$PILOT_ARCHIVE" ]]; then
        PILOT_TEMP="$(mktemp -d "$RUN_DIR/.pilot_epoch0050.XXXXXX")"
        cp -- "$RUN_DIR/summary.json" "$PILOT_TEMP/summary.json"
        cp -r -- "$RUN_DIR/evaluation" "$PILOT_TEMP/evaluation"
        "$PYTHON_BIN" -B -c 'import json,sys; s=json.load(open(sys.argv[1])); e=json.load(open(sys.argv[2])); assert s["epoch"] == e["checkpoint_epoch"] == 50 and s["config"]["reward"] == "intrinsic_velocity", "Expected completed combined-reward epoch-50 pilot"' "$PILOT_TEMP/summary.json" "$PILOT_TEMP/evaluation/$PILOT_REPORT"
        mv -T -- "$PILOT_TEMP" "$PILOT_ARCHIVE"
    fi
    "$PYTHON_BIN" -B -c 'import json,sys; s=json.load(open(sys.argv[1])); e=json.load(open(sys.argv[2])); assert s["epoch"] == e["checkpoint_epoch"] == 50 and s["config"]["reward"] == "intrinsic_velocity", "Pilot archive is incomplete or inconsistent"' "$PILOT_ARCHIVE/summary.json" "$PILOT_ARCHIVE/evaluation/$PILOT_REPORT"
fi
"${COMMAND[@]}"
"${EVALUATION_COMMAND[@]}"
