#!/usr/bin/env bash
set -euo pipefail

K_VALUES="${K_VALUES:-1,5,10,50,100}"
SONGMAE_CHECKPOINT=model_step_099999.pth
source "$(dirname "$0")/syllable_knn_lib.sh"

OUT_ROOT="${OUT_ROOT:-$ROOT/results/knn/manuscript_reference_fit/micro_ablations}"
RAW_ROOT="$OUT_ROOT/raw"
PCA_ROOT="$OUT_ROOT/pca128"
WORKER_INDEX="${WORKER_INDEX:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MODELS=(
  xcl_micro_100k_p32x1_random
  Xcl_micro_100k_p128x1_default
  Xcl_micro_100k_p16x1_default
  Xcl_micro_100k_p4x4_default
  Xcl_micro_100k_p32x1_c0025
  Xcl_micro_100k_p32x1_c005
  Xcl_micro_100k_p32x1_c010
  xcl_micro_100k_p32x4_c0025
  xcl_micro_100k_p32x4_c005
  xcl_micro_100k_p32x4_c010
)

job=0
for row in "${DATASETS[@]}"; do
  IFS="|" read -r species annotations specs wavs <<< "$row"
  [[ -n "${DATASET_FILTER:-}" && "$species" != "$DATASET_FILTER" ]] && continue
  while IFS= read -r bird; do
    [[ -n "${BIRD_FILTER:-}" && "$bird" != "$BIRD_FILTER" ]] && continue
    for model in "${MODELS[@]}"; do
      [[ -n "${MODEL_FILTER:-}" && "$model" != "$MODEL_FILTER" ]] && continue
      raw="$(knn_out_dir "$RAW_ROOT" "$species" "$bird" "$model" 5 end_of_block)"
      pca="$(knn_out_dir "$PCA_ROOT" "$species" "$bird" "$model" 5 end_of_block)"
      if ((job % NUM_WORKERS == WORKER_INDEX)); then
        run_knn_pair "$raw" "$pca" "$model" "$annotations" "$specs" "$wavs" "$bird" 5 end_of_block
      fi
      ((job += 1))
    done
  done < <(birds_in_json "$annotations")
done
