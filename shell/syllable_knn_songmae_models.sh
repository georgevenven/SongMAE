#!/usr/bin/env bash
set -euo pipefail

PCA_COMPONENTS="${PCA_COMPONENTS:-0}"
source "$(dirname "$0")/syllable_knn_lib.sh"

OUT_ROOT="${OUT_ROOT:-$ROOT/results/knn/songmae_500k_all_models_all_layers_raw}"
SONGMAE_CHECKPOINT="model_step_499999.pth"
WORKER_INDEX="${WORKER_INDEX:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MODELS=(
  xcl_large_500k_p32x1_c005 xcl_large_500k_p32x4_c010
  xcl_base_500k_p32x1_c005 xcl_base_500k_p32x4_c010
  xcl_micro_500k_p32x1_c005 xcl_micro_500k_p32x4_c010
)

mkdir -p "$OUT_ROOT"
job=0
for row in "${DATASETS[@]}"; do
  IFS="|" read -r species json spec_dir wav_dir <<< "$row"
  while IFS= read -r bird; do
    for model in "${MODELS[@]}"; do
      [[ "$model" == xcl_large_* ]] && layers=({0..11}) || layers=({0..5})
      for layer in "${layers[@]}"; do
        if ((job % NUM_WORKERS == WORKER_INDEX)); then
          out_dir="$(knn_out_dir "$OUT_ROOT" "$species" "$bird" "$model" "$layer" end_of_block)"
          run_knn "$model" "$out_dir" "$json" "$spec_dir" "$wav_dir" "$bird" "$layer" end_of_block
          cleanup_embeddings "$out_dir"
        fi
        ((job += 1))
      done
    done
  done < <(birds_in_json "$json")
done
