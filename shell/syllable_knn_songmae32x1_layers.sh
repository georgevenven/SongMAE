#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/syllable_knn_lib.sh"

MODEL="${MODEL:-xcl_base_100k_p32x1_c010}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/syllable_knn_songmae32x1_layers_rawdims}"
LAYERS="${LAYERS:-0 1 2 3 4 5}"
TARGET_FEATURE_TYPE="end_of_block"
read -r -a LAYER_LIST <<< "$LAYERS"

mkdir -p "$OUT_ROOT"
date | tee -a "$OUT_ROOT/run.log"
for row in "${DATASETS[@]}"; do
  IFS="|" read -r species json spec_dir wav_dir <<< "$row"
  while IFS= read -r bird; do
    for layer in "${LAYER_LIST[@]}"; do
      out_dir="$(knn_out_dir "$OUT_ROOT" "$species" "$bird" "$MODEL" "$layer" "$TARGET_FEATURE_TYPE")"
      run_knn "$MODEL" "$out_dir" "$json" "$spec_dir" "$wav_dir" "$bird" "$layer" "0" "$TARGET_FEATURE_TYPE" | tee -a "$OUT_ROOT/run.log"
      cleanup_embeddings "$out_dir"
    done
  done < <(birds_in_json "$json")
done
find "$OUT_ROOT" -name summary.json | sort | wc -l | tee -a "$OUT_ROOT/run.log"
