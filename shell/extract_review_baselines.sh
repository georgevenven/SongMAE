#!/usr/bin/env bash
# Preparation only until explicitly invoked: extract compatible embedding folders, no probes.
set -euo pipefail
cd "$(dirname "$0")/.."
source shell/linear_probe_lib.sh

OUT_ROOT=${OUT_ROOT:-$LINEAR_PROBE_ROOT/results/review_baselines}
LAYER=${LAYER:-11}
MODELS=(
  "beats_iter3_plus_as2m|beats|1"
  "beats_iter3_plus_as2m_speed0p5|beats|0.5"
  "beats_iter3_plus_as2m_speed0p25|beats|0.25"
  "birdmae_base_speed1|birdmae|1"
  "birdmae_base_speed0p5|birdmae|0.5"
  "birdmae_base_speed0p25|birdmae|0.25"
  "birdmae_base_speed0p125|birdmae|0.125"
  "birdmae_base_speed0p0625|birdmae|0.0625"
  "birdmae_base_speed0p03125|birdmae|0.03125"
)

for dataset_row in "${LINEAR_PROBE_DATASETS[@]}"; do
  IFS='|' read -r dataset annotations specs <<< "$dataset_row"
  linear_probe_selected "$dataset" "$DATASET_FILTER" || continue
  while read -r bird; do
    linear_probe_selected "$bird" "$BIRD_FILTER" || continue
    for model_row in "${MODELS[@]}"; do
      IFS='|' read -r name model speed <<< "$model_row"
      linear_probe_selected "$name" "$MODEL_FILTER" || continue
      output="$OUT_ROOT/$dataset/$bird/$name/embeddings"
      [[ -f "$output/metadata.json" ]] && { echo "exists: $output"; continue; }
      "$PYTHON_BIN" src/external_models/review_baselines.py \
        --model "$model" --speed "$speed" --encoder_layer_idx "$LAYER" \
        --spec_dir "$specs" --wav_dir "$WAV_ROOT" --annotation_file "$annotations" \
        --bird "$bird" --recording_mode events --out_dir "$output" \
        --chunk_timebins 1000 --num_timebins "$NUM_TIMEBINS" \
        --balanced_events "$FOLDS" --event_seed "$SEED"
    done
  done < <(linear_probe_birds "$annotations")
done
