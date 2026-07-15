#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/syllable_knn_lib.sh"

OUT_ROOT="${OUT_ROOT:-$ROOT/results/syllable_knn_allbirds_models_rawdims}"
MODELS="${MODELS:-xcl_large_500k_p32x4_c010 birdaves_biox_base hubert_base_ls960}"
TARGET_FEATURE_TYPE="${TARGET_FEATURE_TYPE:-end_of_block}"
LAYER="${LAYER:--1}"
read -r -a MODEL_LIST <<< "$MODELS"

mkdir -p "$OUT_ROOT"
date | tee -a "$OUT_ROOT/run.log"
for row in "${DATASETS[@]}"; do
  IFS="|" read -r species json spec_dir wav_dir <<< "$row"
  while IFS= read -r bird; do
    for model in "${MODEL_LIST[@]}"; do
      out_dir="$(knn_out_dir "$OUT_ROOT" "$species" "$bird" "$model" "$LAYER" "$TARGET_FEATURE_TYPE")"
      run_knn "$model" "$out_dir" "$json" "$spec_dir" "$wav_dir" "$bird" "$LAYER" "$TARGET_FEATURE_TYPE" | tee -a "$OUT_ROOT/run.log"
      cleanup_embeddings "$out_dir"
    done
  done < <(birds_in_json "$json")
done
find "$OUT_ROOT" -name summary.json | sort | wc -l | tee -a "$OUT_ROOT/run.log"
