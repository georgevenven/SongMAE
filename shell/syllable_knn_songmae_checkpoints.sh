#!/usr/bin/env bash
set -euo pipefail

K_VALUES="${K_VALUES:-1,5,10,50,100}"
source "$(dirname "$0")/syllable_knn_lib.sh"

OUT_ROOT="${OUT_ROOT:-$ROOT/results/knn/manuscript_reference_fit/checkpoints}"
RAW_ROOT="$OUT_ROOT/raw"
PCA_ROOT="$OUT_ROOT/pca128"
WORKER_INDEX="${WORKER_INDEX:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"
CHECKPOINTS=(000000 020000 050000 100000 499999)
SHAPES=("32x1|p32x1_c005")

job=0
for row in "${DATASETS[@]}"; do
  IFS="|" read -r species annotations specs _ <<< "$row"
  [[ -n "${DATASET_FILTER:-}" && "$species" != "$DATASET_FILTER" ]] && continue
  while IFS= read -r bird; do
    [[ -n "${BIRD_FILTER:-}" && "$bird" != "$BIRD_FILTER" ]] && continue
    for size in large base micro; do
      layer=5
      [[ "$size" == large ]] && layer=11
      for shape_row in "${SHAPES[@]}"; do
        IFS="|" read -r shape suffix <<< "$shape_row"
        run="xcl_${size}_500k_${suffix}"
        for step in "${CHECKPOINTS[@]}"; do
          name="${size}_${shape}_step_${step}"
          raw="$RAW_ROOT/$species/$bird/$name/layer_$layer/end_of_block"
          pca="$PCA_ROOT/$species/$bird/$name/layer_$layer/end_of_block"
          if ((job % NUM_WORKERS == WORKER_INDEX)); then
            if ! KNN_NAME="$name" SONGMAE_CHECKPOINT="model_step_${step}.pth" \
              run_knn_pair "$raw" "$pca" "$run" "$annotations" "$specs" "" "$bird" "$layer" end_of_block; then
              touch "$raw/FAILED" "$pca/FAILED"
            fi
          fi
          ((job += 1))
        done
      done
    done
  done < <(birds_in_json "$annotations")
done
