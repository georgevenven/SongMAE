#!/usr/bin/env bash
set -euo pipefail

K_VALUES="${K_VALUES:-1,5,10,50,100}"
source "$(dirname "$0")/syllable_knn_lib.sh"

OUT_ROOT="${OUT_ROOT:-$ROOT/results/knn/manuscript_reference_fit/best_layers}"
RAW_ROOT="$OUT_ROOT/raw"
PCA_ROOT="$OUT_ROOT/pca128"
SONGMAE_CHECKPOINT="model_step_499999.pth"
WORKER_INDEX="${WORKER_INDEX:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MODELS=(
  xcl_large_500k_p32x1_c005
  xcl_large_500k_p32x4_c010
  birdaves_biox_base
  hubert_base_ls960
)
LAYERS=({0..11})

mkdir -p "$OUT_ROOT"
job=0
for row in "${DATASETS[@]}"; do
  IFS="|" read -r species json spec_dir wav_dir <<< "$row"
  while IFS= read -r bird; do
    for model in "${MODELS[@]}"; do
      for layer in "${LAYERS[@]}"; do
        if ((job % NUM_WORKERS == WORKER_INDEX)); then
          raw="$(knn_out_dir "$RAW_ROOT" "$species" "$bird" "$model" "$layer" end_of_block)"
          pca="$(knn_out_dir "$PCA_ROOT" "$species" "$bird" "$model" "$layer" end_of_block)"
          if run_knn_pair "$raw" "$pca" "$model" "$json" "$spec_dir" "$wav_dir" "$bird" "$layer" end_of_block; then
            rm -f "$raw/FAILED" "$pca/FAILED"
          else
            touch "$raw/FAILED" "$pca/FAILED"
          fi
        fi
        ((job += 1))
      done
    done
  done < <(birds_in_json "$json")
done
