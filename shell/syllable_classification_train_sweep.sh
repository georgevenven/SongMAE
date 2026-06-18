#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
EMBED_ROOT="${EMBED_ROOT:-$ROOT/results/syllable_classification_zf_models_20260616/zf}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/syllable_classification_train_sweep/zf}"
ANNOTATIONS="${ANNOTATIONS:-$ROOT/files/annotation jsons/zf_annotations.json}"
MODEL="${MODEL:-mlp}"
FEATURE_KEY="${FEATURE_KEY:-encoded_embeddings}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
LR="${LR:-0.001}"
TRAIN_SECONDS="${TRAIN_SECONDS:-32 64 128 256 512 MAX}"
MODELS="${MODELS:-songmae_old songmae_new hubert bird_mae songmae_random aves}"
BIRDS="${BIRDS:-}"
OVERWRITE="${OVERWRITE:-0}"

read -r -a BUDGETS <<< "$TRAIN_SECONDS"
read -r -a MODEL_NAMES <<< "$MODELS"
read -r -a BIRD_NAMES <<< "$BIRDS"
mkdir -p "$OUT_ROOT"

selected_bird() {
  local bird="$1"
  [[ "${#BIRD_NAMES[@]}" -eq 0 ]] && return 0
  for wanted in "${BIRD_NAMES[@]}"; do
    [[ "$bird" == "$wanted" ]] && return 0
  done
  return 1
}

for bird_dir in "$EMBED_ROOT"/*; do
  [[ -d "$bird_dir" ]] || continue
  bird="$(basename "$bird_dir")"
  selected_bird "$bird" || continue
  for model_name in "${MODEL_NAMES[@]}"; do
    embeddings="$bird_dir/$model_name/embeddings/embeddings.npz"
    [[ -f "$embeddings" ]] || continue
    for budget in "${BUDGETS[@]}"; do
      run_dir="$OUT_ROOT/$bird/$model_name/train_${budget}s"
      metrics="$run_dir/metrics.json"
      tmp="$run_dir/metrics.tmp"
      if [[ -f "$metrics" && "$OVERWRITE" != "1" ]]; then
        echo "exists: $metrics"
        continue
      fi
      mkdir -p "$run_dir"
      echo "running: bird=$bird model=$model_name train=$budget"
      if "$PYTHON_BIN" src/evals/syllable_classification.py \
        --embeddings "$embeddings" \
        --annotations "$ANNOTATIONS" \
        --model "$MODEL" \
        --feature_key "$FEATURE_KEY" \
        --val_fraction "$VAL_FRACTION" \
        --seed "$SEED" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --max_train_seconds "$budget" > "$tmp"; then
        mv "$tmp" "$metrics"
      else
        rm -f "$tmp"
        echo "failed: bird=$bird model=$model_name train=$budget" 1>&2
      fi
    done
  done
done
