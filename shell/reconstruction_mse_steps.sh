#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/xcl_micro_500k_p32x4_default}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/mse_v_steps}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-42}"

DATASETS=(
  "zf|files/annotation jsons/zf_annotations.json|/media/george-vengrovski/disk2/specs/zebra_finch_5ms"
  "canary|files/annotation jsons/canary_annotations.json|/media/george-vengrovski/disk2/specs/canary_5ms"
  "bf|files/annotation jsons/bf_annotations.json|/media/george-vengrovski/disk2/specs/bengalese_finch_5ms"
)

mapfile -t CHECKPOINTS < <(find "$RUN_DIR/weights" -maxdepth 1 -name "model_step_*.pth" | sort)
test "${#CHECKPOINTS[@]}" -gt 0
last=$((${#CHECKPOINTS[@]} - 1))

for i in "${!CHECKPOINTS[@]}"; do
  if (( i % CHECKPOINT_INTERVAL != 0 && i != last )); then
    continue
  fi
  checkpoint="$(basename "${CHECKPOINTS[$i]}")"
  step="${checkpoint#model_step_}"
  step="${step%.pth}"
  for row in "${DATASETS[@]}"; do
    IFS="|" read -r species annotations spec_dir <<< "$row"
    echo "running: species=$species checkpoint=$checkpoint"
    "$PYTHON_BIN" src/evals/eval_reconstructions.py \
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
