#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/george-vengrovski/Documents/projects/TinyBird"
RUNS_DIR="$ROOT/runs"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage:
  shell/continue_pretrain.sh BASE_RUN DATASET_KEY NEW_RUN_NAME [EXTRA_STEPS] [INPUT_NORMALIZATION] [BATCH_SIZE]

Example:
  shell/continue_pretrain.sh \
    xcm_voronoi_mask_no_normalize_32h_10w \
    zf \
    xcm_voronoi_mask_no_normalize_32h_10w_zf_continue10k \
    10000 \
    per_file_zscore \
    48

Dataset keys:
  zf
  bf
  canary

Notes:
  - Creates a lean clone in runs/NEW_RUN_NAME
  - Rewrites train_dir, val_dir, run_name, steps, input_normalization, and batch_size in the cloned config.json
  - Copies loss_log.txt and only the latest checkpoint from the base run
  - Continues training from the cloned run directory
  - INPUT_NORMALIZATION can be: audio_params, per_file_zscore
  - BATCH_SIZE defaults to 48
EOF
}

if [[ $# -lt 3 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
  usage
  exit 0
fi

BASE_RUN_ARG="$1"
DATASET_KEY="$2"
NEW_RUN_NAME="$3"
EXTRA_STEPS="${4:-10000}"
INPUT_NORMALIZATION="${5:-audio_params}"
BATCH_SIZE="${6:-48}"

if [[ "$INPUT_NORMALIZATION" != "audio_params" ]] && [[ "$INPUT_NORMALIZATION" != "per_file_zscore" ]]; then
  echo "Unknown input normalization: $INPUT_NORMALIZATION" 1>&2
  usage
  exit 1
fi

resolve_run_dir() {
  local run_arg="$1"
  if [[ -d "$run_arg" ]]; then
    python - <<'PY' "$run_arg"
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
    return
  fi
  if [[ -d "$ROOT/$run_arg" ]]; then
    python - <<'PY' "$ROOT/$run_arg"
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
    return
  fi
  if [[ -d "$RUNS_DIR/$run_arg" ]]; then
    python - <<'PY' "$RUNS_DIR/$run_arg"
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
    return
  fi
  echo "Base run not found: $run_arg" 1>&2
  exit 1
}

case "$DATASET_KEY" in
  zf)
    TRAIN_DIR="/media/george-vengrovski/disk2/specs/zf_64hop_32khz_train"
    VAL_DIR="/media/george-vengrovski/disk2/specs/zf_64hop_32khz_val"
    ;;
  bf)
    TRAIN_DIR="/media/george-vengrovski/disk2/specs/bf_64hop_32khz_train"
    VAL_DIR="/media/george-vengrovski/disk2/specs/bf_64hop_32khz_val"
    ;;
  canary)
    TRAIN_DIR="/media/george-vengrovski/disk2/specs/canary_individual_identification_64hop_32khz_train"
    VAL_DIR="/media/george-vengrovski/disk2/specs/canary_individual_identification_64hop_32khz_val"
    ;;
  *)
    echo "Unknown dataset key: $DATASET_KEY" 1>&2
    usage
    exit 1
    ;;
esac

BASE_RUN_DIR="$(resolve_run_dir "$BASE_RUN_ARG")"
TARGET_RUN_DIR="$RUNS_DIR/$NEW_RUN_NAME"

if [[ -e "$TARGET_RUN_DIR" ]]; then
  echo "Target run already exists: $TARGET_RUN_DIR" 1>&2
  exit 1
fi

if [[ ! -f "$BASE_RUN_DIR/config.json" ]]; then
  echo "Missing config.json in base run: $BASE_RUN_DIR" 1>&2
  exit 1
fi

if [[ ! -d "$BASE_RUN_DIR/weights" ]]; then
  echo "Missing weights dir in base run: $BASE_RUN_DIR/weights" 1>&2
  exit 1
fi

if [[ ! -d "$TRAIN_DIR" ]]; then
  echo "Missing train dir: $TRAIN_DIR" 1>&2
  exit 1
fi

if [[ ! -d "$VAL_DIR" ]]; then
  echo "Missing val dir: $VAL_DIR" 1>&2
  exit 1
fi

LATEST_CHECKPOINT="$(
python - <<'PY' "$BASE_RUN_DIR/weights"
import sys
from pathlib import Path

weights_dir = Path(sys.argv[1])
checkpoints = sorted(
    weights_dir.glob("model_step_*.pth"),
    key=lambda path: int(path.stem.split("_step_")[1]),
)
if not checkpoints:
    raise SystemExit("No checkpoints found")
print(checkpoints[-1])
PY
)"

mkdir -p "$TARGET_RUN_DIR/weights"
cp "$BASE_RUN_DIR/config.json" "$TARGET_RUN_DIR/config.json"
cp "$LATEST_CHECKPOINT" "$TARGET_RUN_DIR/weights/"

if [[ -f "$BASE_RUN_DIR/loss_log.txt" ]]; then
  cp "$BASE_RUN_DIR/loss_log.txt" "$TARGET_RUN_DIR/loss_log.txt"
fi

if [[ -f "$BASE_RUN_DIR/audio_params.json" ]]; then
  cp "$BASE_RUN_DIR/audio_params.json" "$TARGET_RUN_DIR/audio_params.json"
fi

python - <<'PY' "$TARGET_RUN_DIR/config.json" "$TRAIN_DIR" "$VAL_DIR" "$NEW_RUN_NAME" "$EXTRA_STEPS" "$INPUT_NORMALIZATION" "$BATCH_SIZE"
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
train_dir = sys.argv[2]
val_dir = sys.argv[3]
run_name = sys.argv[4]
steps = int(sys.argv[5])
input_normalization = sys.argv[6]
batch_size = int(sys.argv[7])

config = json.loads(config_path.read_text(encoding="utf-8"))
config["train_dir"] = train_dir
config["val_dir"] = val_dir
config["run_name"] = run_name
config["steps"] = steps
config["input_normalization"] = input_normalization
config["batch_size"] = batch_size
config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
PY

echo "Cloned base run:"
echo "  from: $BASE_RUN_DIR"
echo "  to:   $TARGET_RUN_DIR"
echo "Latest checkpoint: $(basename "$LATEST_CHECKPOINT")"
echo "Using dataset:"
echo "  train_dir: $TRAIN_DIR"
echo "  val_dir:   $VAL_DIR"
echo "Extra steps: $EXTRA_STEPS"
echo "Normalization: $INPUT_NORMALIZATION"
echo "Batch size: $BATCH_SIZE"

cd "$ROOT"
"$PYTHON_BIN" src/pretrain.py --continue_from "$TARGET_RUN_DIR"
