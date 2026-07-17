#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/xcl_micro_500k_p32x4_default}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/mse_v_steps}"
STEP_INTERVAL="${STEP_INTERVAL:-50000}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-42}"
OVERWRITE="${OVERWRITE:-0}"

DATASETS=(
  "zf|files/annotation jsons/zf_annotations.json|/media/george-vengrovski/disk2/specs/zebra_finch_5ms"
  "canary|files/annotation jsons/canary_annotations.json|/media/george-vengrovski/disk2/specs/canary_5ms"
  "bf|files/annotation jsons/bf_annotations.json|/media/george-vengrovski/disk2/specs/bengalese_finch_5ms"
)

mapfile -t CHECKPOINTS < <(find "$RUN_DIR/weights" -maxdepth 1 -name "model_step_*.pth" | sort -V)
test "${#CHECKPOINTS[@]}" -gt 0

for path in "${CHECKPOINTS[@]}"; do
  checkpoint="$(basename "$path")"
  step="${checkpoint#model_step_}"
  step="${step%.pth}"
  step_number=$((10#$step))
  if (( step_number % STEP_INTERVAL != 0 && (step_number + 1) % STEP_INTERVAL != 0 )); then
    continue
  fi
  for row in "${DATASETS[@]}"; do
    IFS="|" read -r species annotations spec_dir <<< "$row"
    summary="$OUT_ROOT/$species/step_$step/MSE analysis/summary.json"
    if [[ -f "$summary" && "$OVERWRITE" != "1" ]]; then
      echo "exists: $summary"
      continue
    fi
    echo "running: species=$species checkpoint=$checkpoint"
    "$PYTHON_BIN" -m src.evals.eval_reconstructions \
      --run_dir "$RUN_DIR" \
      --checkpoint "$checkpoint" \
      --spec_dir "$spec_dir" \
      --annotation_file "$annotations" \
      --recording_mode events \
      --out_dir "$OUT_ROOT/$species/step_$step" \
      --num_samples "$NUM_SAMPLES" \
      --batch_size "$BATCH_SIZE" \
      --seed "$SEED" \
      --numbers_only
  done
done
