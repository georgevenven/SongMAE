#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/george-vengrovski/Documents/projects/TinyBird"
RUNS_DIR="$ROOT/runs"
PYTHON_BIN="${PYTHON_BIN:-python}"

TRAIN_DIR="${TRAIN_DIR:-/media/george-vengrovski/disk2/data2vec_train_data/songmae_mid_train/train}"
VAL_DIR="${VAL_DIR:-/media/george-vengrovski/disk2/data2vec_train_data/songmae_mid_train/eval}"

STEPS="${STEPS:-20000}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
BATCH_SIZE="${BATCH_SIZE:-48}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RUN_SUFFIX="${RUN_SUFFIX:-e2000}"

DATA2VEC_INPUT_NORMALIZATION="${DATA2VEC_INPUT_NORMALIZATION:-audio_params}"
SONGMAE_INPUT_NORMALIZATION="${SONGMAE_INPUT_NORMALIZATION:-audio_params}"

SUITE_PREFIX="${SUITE_PREFIX:-songmae_mid_train}"
SONGMAE_BASE_RUN="${SONGMAE_BASE_RUN:-/media/george-vengrovski/Desk SSD/LambdaLabsIndividual_ID_RUNS/xcl_train_audio_params_no_patchnorm_bs256_small12m_ckpt50k_plus_last}"

if [[ "$STEPS" =~ ^[0-9]+$ ]] && (( STEPS % 1000 == 0 )); then
  STEP_LABEL="$((STEPS / 1000))k"
else
  STEP_LABEL="$STEPS"
fi

usage() {
  cat <<EOF
Usage:
  shell/train_xcl_transfer_suite.sh ACTION

Actions:
  all
  data2vec
  songmae
  status

Default dataset:
  TRAIN_DIR=$TRAIN_DIR
  VAL_DIR=$VAL_DIR

Default run names:
  ${SUITE_PREFIX}_songmae_scratch_${STEP_LABEL}_${RUN_SUFFIX}
  ${SUITE_PREFIX}_songmae_from_songmae_${STEP_LABEL}_${RUN_SUFFIX}
  ${SUITE_PREFIX}_data2vec_from_songmae_${STEP_LABEL}_${RUN_SUFFIX}

Key env overrides:
  STEPS=$STEPS
  EVAL_EVERY=$EVAL_EVERY
  BATCH_SIZE=$BATCH_SIZE
  NUM_WORKERS=$NUM_WORKERS
  DATA2VEC_INPUT_NORMALIZATION=$DATA2VEC_INPUT_NORMALIZATION
  SONGMAE_INPUT_NORMALIZATION=$SONGMAE_INPUT_NORMALIZATION
  RUN_SUFFIX=$RUN_SUFFIX
  SUITE_PREFIX=$SUITE_PREFIX
  TRAIN_DIR=/path/to/train
  VAL_DIR=/path/to/val
EOF
}

if [[ $# -lt 1 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
  usage
  exit 0
fi

ACTION="$1"

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Missing directory: $path" 1>&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing file: $path" 1>&2
    exit 1
  fi
}

latest_checkpoint() {
  local weights_dir="$1"
  local pattern="$2"
  python - <<'PY' "$weights_dir" "$pattern"
import sys
from pathlib import Path

weights_dir = Path(sys.argv[1])
pattern = sys.argv[2]
paths = sorted(
    weights_dir.glob(pattern),
    key=lambda path: int(path.stem.split("_step_")[1]),
)
if not paths:
    raise SystemExit(1)
print(paths[-1])
PY
}

ensure_new_run_dir() {
  local run_dir="$1"
  if [[ -e "$run_dir" ]]; then
    echo "Target run already exists: $run_dir" 1>&2
    exit 1
  fi
}

clone_songmae_seed() {
  local base_run="$1"
  local target_run="$2"
  local latest_model

  ensure_new_run_dir "$target_run"
  require_file "$base_run/config.json"
  require_dir "$base_run/weights"

  latest_model="$(latest_checkpoint "$base_run/weights" "model_step_*.pth")"

  mkdir -p "$target_run/weights"
  cp "$base_run/config.json" "$target_run/config.json"
  cp "$latest_model" "$target_run/weights/"

  if [[ -f "$base_run/loss_log.txt" ]]; then
    cp "$base_run/loss_log.txt" "$target_run/loss_log.txt"
  fi

  if [[ -f "$base_run/audio_params.json" ]]; then
    cp "$base_run/audio_params.json" "$target_run/audio_params.json"
  fi

  python - <<'PY' "$target_run/config.json" "$TRAIN_DIR" "$VAL_DIR" "$(basename "$target_run")" "$STEPS" "$EVAL_EVERY" "$BATCH_SIZE" "$NUM_WORKERS" "$SONGMAE_INPUT_NORMALIZATION"
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
train_dir = sys.argv[2]
val_dir = sys.argv[3]
run_name = sys.argv[4]
steps = int(sys.argv[5])
eval_every = int(sys.argv[6])
batch_size = int(sys.argv[7])
num_workers = int(sys.argv[8])
input_normalization = sys.argv[9]

config = json.loads(config_path.read_text(encoding="utf-8"))
config["train_dir"] = train_dir
config["val_dir"] = val_dir
config["run_name"] = run_name
config["steps"] = steps
config["eval_every"] = eval_every
config["batch_size"] = batch_size
config["num_workers"] = num_workers
config["input_normalization"] = input_normalization
config["wandb"] = False
config["normalize_patches"] = False
config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
PY

  echo "Prepared SongMAE seed:"
  echo "  base:   $base_run"
  echo "  target: $target_run"
  echo "  model:  $(basename "$latest_model")"
}

clone_data2vec_seed() {
  local base_run="$1"
  local target_run="$2"
  local latest_model
  local latest_trainer

  ensure_new_run_dir "$target_run"
  require_file "$base_run/config.json"
  require_dir "$base_run/weights"

  latest_model="$(latest_checkpoint "$base_run/weights" "model_step_*.pth")"
  latest_trainer="$(latest_checkpoint "$base_run/weights" "trainer_step_*.pth")"

  mkdir -p "$target_run/weights"
  cp "$base_run/config.json" "$target_run/config.json"
  cp "$latest_model" "$target_run/weights/"
  cp "$latest_trainer" "$target_run/weights/"

  if [[ -f "$base_run/loss_log.txt" ]]; then
    cp "$base_run/loss_log.txt" "$target_run/loss_log.txt"
  fi

  if [[ -f "$base_run/audio_params.json" ]]; then
    cp "$base_run/audio_params.json" "$target_run/audio_params.json"
  fi

  python - <<'PY' "$target_run/config.json" "$TRAIN_DIR" "$VAL_DIR" "$(basename "$target_run")" "$STEPS" "$EVAL_EVERY" "$BATCH_SIZE" "$NUM_WORKERS" "$DATA2VEC_INPUT_NORMALIZATION"
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
train_dir = sys.argv[2]
val_dir = sys.argv[3]
run_name = sys.argv[4]
steps = int(sys.argv[5])
eval_every = int(sys.argv[6])
batch_size = int(sys.argv[7])
num_workers = int(sys.argv[8])
input_normalization = sys.argv[9]

config = json.loads(config_path.read_text(encoding="utf-8"))
config["train_dir"] = train_dir
config["val_dir"] = val_dir
config["run_name"] = run_name
config["steps"] = steps
config["eval_every"] = eval_every
config["batch_size"] = batch_size
config["num_workers"] = num_workers
config["input_normalization"] = input_normalization
config["wandb"] = False
config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
PY

  echo "Prepared data2vec seed:"
  echo "  base:    $base_run"
  echo "  target:  $target_run"
  echo "  student: $(basename "$latest_model")"
  echo "  trainer: $(basename "$latest_trainer")"
}

run_data2vec_from_songmae() {
  local run_name="${SUITE_PREFIX}_data2vec_from_songmae_${STEP_LABEL}_${RUN_SUFFIX}"
  echo "=== START $run_name ==="
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$ROOT/src/data2vec_train.py" \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --run_name "$run_name" \
    --steps "$STEPS" \
    --lr 1e-4 \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --patch_height 32 \
    --patch_width 10 \
    --num_timebins 1000 \
    --dropout 0.1 \
    --mask_p 0.75 \
    --mask_c 0.1 \
    --mask_type voronoi \
    --eval_every "$EVAL_EVERY" \
    --warmup_steps 1000 \
    --min_lr 1e-5 \
    --amp \
    --weight_decay 0.1 \
    --input_normalization "$DATA2VEC_INPUT_NORMALIZATION" \
    --ema_decay 0.99 \
    --teacher_target_feature ffn \
    --loss_type mse \
    --init_from_pretrained_run "$SONGMAE_BASE_RUN"
}

run_songmae_scratch() {
  local run_name="${SUITE_PREFIX}_songmae_scratch_${STEP_LABEL}_${RUN_SUFFIX}"
  echo "=== START $run_name ==="
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$ROOT/src/pretrain.py" \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --run_name "$run_name" \
    --steps "$STEPS" \
    --lr 3e-4 \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --patch_height 32 \
    --patch_width 10 \
    --num_timebins 1000 \
    --dropout 0.1 \
    --mask_p 0.75 \
    --mask_c 0.1 \
    --mask_type voronoi \
    --eval_every "$EVAL_EVERY" \
    --warmup_steps 1000 \
    --min_lr 1e-5 \
    --amp \
    --weight_decay 0.1 \
    --no_normalize_patches \
    --input_normalization "$SONGMAE_INPUT_NORMALIZATION"
}

run_songmae_from_songmae() {
  local run_name="${SUITE_PREFIX}_songmae_from_songmae_${STEP_LABEL}_${RUN_SUFFIX}"
  local run_dir="$RUNS_DIR/$run_name"
  clone_songmae_seed "$SONGMAE_BASE_RUN" "$run_dir"
  echo "=== START $run_name ==="
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$ROOT/src/pretrain.py" --continue_from "$run_dir"
}

print_run_status() {
  local run_name="$1"
  local run_dir="$RUNS_DIR/$run_name"

  echo "$run_name"
  if [[ ! -d "$run_dir" ]]; then
    echo "  status: missing"
    return
  fi

  echo "  dir: $run_dir"

  python - <<'PY' "$run_dir"
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
weights = run_dir / "weights"
model_steps = sorted(
    [
        int(path.stem.split("_step_")[1])
        for path in weights.glob("model_step_*.pth")
    ]
)
trainer_steps = sorted(
    [
        int(path.stem.split("_step_")[1])
        for path in weights.glob("trainer_step_*.pth")
    ]
)
print(f"  latest_model_step: {model_steps[-1] if model_steps else 'none'}")
print(f"  latest_trainer_step: {trainer_steps[-1] if trainer_steps else 'none'}")
PY

  if [[ -f "$run_dir/loss_log.txt" ]]; then
    echo "  last_log:"
    tail -n 1 "$run_dir/loss_log.txt" | sed 's/^/    /'
  else
    echo "  last_log: none"
  fi
}

status() {
  print_run_status "${SUITE_PREFIX}_songmae_scratch_${STEP_LABEL}_${RUN_SUFFIX}"
  print_run_status "${SUITE_PREFIX}_songmae_from_songmae_${STEP_LABEL}_${RUN_SUFFIX}"
  print_run_status "${SUITE_PREFIX}_data2vec_from_songmae_${STEP_LABEL}_${RUN_SUFFIX}"
}

require_dir "$TRAIN_DIR"
require_dir "$VAL_DIR"
require_dir "$SONGMAE_BASE_RUN"

case "$ACTION" in
  all)
    run_songmae_scratch
    run_songmae_from_songmae
    run_data2vec_from_songmae
    ;;
  data2vec)
    run_data2vec_from_songmae
    ;;
  songmae)
    run_songmae_scratch
    run_songmae_from_songmae
    ;;
  status)
    status
    ;;
  *)
    echo "Unknown action: $ACTION" 1>&2
    usage
    exit 1
    ;;
esac
