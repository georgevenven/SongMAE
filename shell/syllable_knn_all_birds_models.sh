#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/syllable_knn_lib.sh"

OUT_ROOT="${OUT_ROOT:-$ROOT/results/knn/all_birds_models_all_layers_zscore}"
SONGMAE_CHECKPOINT="model_step_499999.pth"
MODELS=(xcl_large_500k_p32x4_c010 birdaves_biox_base hubert_base_ls960)
LAYERS=({0..11})

mkdir -p "$OUT_ROOT"
date | tee -a "$OUT_ROOT/run.log"
for row in "${DATASETS[@]}"; do
  IFS="|" read -r species json spec_dir wav_dir <<< "$row"
  while IFS= read -r bird; do
    for model in "${MODELS[@]}"; do
      for layer in "${LAYERS[@]}"; do
        out_dir="$(knn_out_dir "$OUT_ROOT" "$species" "$bird" "$model" "$layer" end_of_block)"
        run_knn "$model" "$out_dir" "$json" "$spec_dir" "$wav_dir" "$bird" "$layer" end_of_block | tee -a "$OUT_ROOT/run.log"
        cleanup_embeddings "$out_dir"
      done
    done
  done < <(birds_in_json "$json")
done
find "$OUT_ROOT" -name summary.json | sort | wc -l | tee -a "$OUT_ROOT/run.log"
