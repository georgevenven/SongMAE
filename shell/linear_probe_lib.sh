#!/usr/bin/env bash

set -euo pipefail

LINEAR_PROBE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/george-vengrovski/anaconda3/envs/mae/bin/python}
NUM_TIMEBINS=${NUM_TIMEBINS:-720000}
FOLDS=${FOLDS:-3}
PCA_COMPONENTS=${PCA_COMPONENTS:-128}
MAX_ITER=${MAX_ITER:-5000}
LOGREG_C=${LOGREG_C:-0.001}
SEED=${SEED:-42}
CLEAN_EMBEDDINGS=${CLEAN_EMBEDDINGS:-1}
DATASET_FILTER=${DATASET_FILTER:-}
BIRD_FILTER=${BIRD_FILTER:-}
MODEL_FILTER=${MODEL_FILTER:-}
LABEL_CAPS=${LABEL_CAPS:-"1 5 10 20 50 100"}

ANNOTATION_ROOT=${ANNOTATION_ROOT:-$LINEAR_PROBE_ROOT/files/annotation jsons}
SPEC_ROOT=${SPEC_ROOT:-/media/george-vengrovski/disk2/specs}
WAV_ROOT=${WAV_ROOT:-/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae}
BIRDAVES_MODEL=$LINEAR_PROBE_ROOT/files/birdaves-biox-base.torchaudio.pt
BIRDAVES_CONFIG=$LINEAR_PROBE_ROOT/files/birdaves-biox-base.torchaudio.model_config.json

LINEAR_PROBE_DATASETS=(
  "zf|$ANNOTATION_ROOT/zf_annotations.json|$SPEC_ROOT/zebra_finch_5ms"
  "bf|$ANNOTATION_ROOT/bf_annotations.json|$SPEC_ROOT/bengalese_finch_5ms"
  "canary|$ANNOTATION_ROOT/canary_annotations.json|$SPEC_ROOT/canary_5ms"
)

linear_probe_selected() {
  [[ -z "$2" || " $2 " == *" $1 "* ]]
}

linear_probe_birds() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
birds = {row["recording"]["bird_id"] for row in data["recordings"]}
print("\n".join(sorted(birds)))
PY
}

linear_probe_extract() {
  local extractor=$1 run_dir=$2 checkpoint=$3 annotations=$4 specs=$5 bird=$6 output=$7 layer=$8
  local layer_args=()
  [[ -n "$layer" ]] && layer_args=(--encoder_layer_idx "$layer")
  [[ "$output" == "$OUT_ROOT/"*"/embeddings" ]]
  rm -rf "$output" "$output.tmp"
  if [[ "$extractor" == aves ]]; then
    "$PYTHON_BIN" src/external_models/aves.py \
      --spec_dir "$specs" --wav_dir "$WAV_ROOT" --annotation_file "$annotations" \
      --out_dir "$output" --bird "$bird" --recording_mode events \
      --aves_model_path "$BIRDAVES_MODEL" --aves_config_path "$BIRDAVES_CONFIG" \
      --model_name birdaves_biox_base --audio_sr 16000 --chunk_timebins 1000 \
      --num_timebins "$NUM_TIMEBINS" --balanced_events "$FOLDS" --event_seed "$SEED" \
      "${layer_args[@]}"
    return
  fi
  if [[ "$extractor" == hubert ]]; then
    "$PYTHON_BIN" src/external_models/hubert.py \
      --spec_dir "$specs" --wav_dir "$WAV_ROOT" --annotation_file "$annotations" \
      --out_dir "$output" --bird "$bird" --recording_mode events \
      --model_name facebook/hubert-base-ls960 --audio_sr 16000 --chunk_timebins 1000 \
      --num_timebins "$NUM_TIMEBINS" --balanced_events "$FOLDS" --event_seed "$SEED" \
      "${layer_args[@]}"
    return
  fi
  "$PYTHON_BIN" -m src.core.extract_embedding \
    --spec_dir "$specs" --run_dir "$run_dir" --checkpoint "$checkpoint" \
    --out_dir "$output" --json_path "$annotations" --bird "$bird" \
    --recording_mode events --minimal --target_feature_type end_of_block \
    --num_timebins "$NUM_TIMEBINS" --balanced_events "$FOLDS" --event_seed "$SEED" \
    "${layer_args[@]}"
}

run_linear_probe_suite() {
  cd "$LINEAR_PROBE_ROOT"
  OUT_ROOT=$(realpath -m "$OUT_ROOT")
  mkdir -p "$OUT_ROOT"
  for dataset_row in "${LINEAR_PROBE_DATASETS[@]}"; do
    IFS="|" read -r dataset annotations specs <<< "$dataset_row"
    linear_probe_selected "$dataset" "$DATASET_FILTER" || continue
    while read -r bird; do
      linear_probe_selected "$bird" "$BIRD_FILTER" || continue
      manifest="$OUT_ROOT/manifests/$dataset/$bird.json"
      for model_row in "${LINEAR_PROBE_MODELS[@]}"; do
        IFS="|" read -r model extractor run_dir checkpoint layer <<< "$model_row"
        linear_probe_selected "$model" "$MODEL_FILTER" || continue
        if [[ "$extractor" == songmae && ! -f "$run_dir/weights/$checkpoint" ]]; then
          echo "missing checkpoint, skipping: $run_dir/weights/$checkpoint" >&2
          continue
        fi

        model_dir="$OUT_ROOT/$dataset/$bird/$model"
        embeddings="$model_dir/embeddings"
        metrics="$model_dir/metrics.json"
        [[ -f "$metrics" ]] && { echo "exists: $metrics"; continue; }
        mkdir -p "$model_dir"
        if [[ ! -f "$embeddings/metadata.json" ]]; then
          echo "extracting: dataset=$dataset bird=$bird model=$model"
          if ! linear_probe_extract \
            "$extractor" "$run_dir" "$checkpoint" "$annotations" "$specs" "$bird" "$embeddings" "$layer"; then
            echo "extraction failed: dataset=$dataset bird=$bird model=$model" >&2
            continue
          fi
        fi

        manifest_args=(--manifest_in "$manifest")
        [[ -f "$manifest" ]] || manifest_args=(--manifest_out "$manifest")
        echo "probing: dataset=$dataset bird=$bird model=$model"
        if "$PYTHON_BIN" src/evals/syllable_classification.py \
          --embeddings "$embeddings" --annotations "$annotations" --folds "$FOLDS" \
          "${manifest_args[@]}" --pca_components "$PCA_COMPONENTS" \
          --pca_cache "$embeddings/pca_${PCA_COMPONENTS}_seed${SEED}.npy" \
          --max_iter "$MAX_ITER" --logreg_c "$LOGREG_C" --seed "$SEED" \
          > "$model_dir/metrics.tmp"; then
          mv "$model_dir/metrics.tmp" "$metrics"
          [[ "$CLEAN_EMBEDDINGS" == 1 ]] && rm -rf "$embeddings"
        else
          rm -f "$model_dir/metrics.tmp"
          echo "probe failed: dataset=$dataset bird=$bird model=$model" >&2
        fi
      done
    done < <(linear_probe_birds "$annotations")
  done
}

run_capped_linear_probe_suite() {
  cd "$LINEAR_PROBE_ROOT"
  OUT_ROOT=$(realpath -m "$OUT_ROOT")
  mkdir -p "$OUT_ROOT"
  for dataset_row in "${LINEAR_PROBE_DATASETS[@]}"; do
    IFS="|" read -r dataset annotations specs <<< "$dataset_row"
    linear_probe_selected "$dataset" "$DATASET_FILTER" || continue
    while read -r bird; do
      linear_probe_selected "$bird" "$BIRD_FILTER" || continue
      for model_row in "${LINEAR_PROBE_MODELS[@]}"; do
        IFS="|" read -r model extractor run_dir checkpoint layer <<< "$model_row"
        linear_probe_selected "$model" "$MODEL_FILTER" || continue
        if [[ "$extractor" == songmae && ! -f "$run_dir/weights/$checkpoint" ]]; then
          echo "missing checkpoint, skipping: $run_dir/weights/$checkpoint" >&2
          continue
        fi

        model_dir="$OUT_ROOT/$dataset/$bird/$model"
        embeddings="$model_dir/embeddings"
        missing=()
        for cap in $LABEL_CAPS; do
          tag=$(printf '%03d' "$cap")
          [[ -f "$model_dir/cap_$tag/metrics.json" ]] || missing+=("$cap")
        done
        ((${#missing[@]})) || { echo "exists: $model_dir"; continue; }
        if [[ ! -f "$embeddings/metadata.json" ]]; then
          echo "extracting: dataset=$dataset bird=$bird model=$model"
          if ! linear_probe_extract \
            "$extractor" "$run_dir" "$checkpoint" "$annotations" "$specs" "$bird" "$embeddings" "$layer"; then
            echo "extraction failed: dataset=$dataset bird=$bird model=$model" >&2
            continue
          fi
        fi

        for cap in "${missing[@]}"; do
          tag=$(printf '%03d' "$cap")
          cap_dir="$model_dir/cap_$tag"
          metrics="$cap_dir/metrics.json"
          manifest="$OUT_ROOT/manifests/$dataset/$bird/cap_$tag.json"
          mkdir -p "$cap_dir"
          manifest_args=(--manifest_in "$manifest")
          [[ -f "$manifest" ]] || manifest_args=(--manifest_out "$manifest")
          echo "probing: dataset=$dataset bird=$bird model=$model K=$cap"
          if "$PYTHON_BIN" src/evals/syllable_classification_capped.py \
            --embeddings "$embeddings" --annotations "$annotations" \
            --label_cap "$cap" --folds "$FOLDS" "${manifest_args[@]}" \
            --pca_components "$PCA_COMPONENTS" \
            --pca_cache "$embeddings/pca_${PCA_COMPONENTS}_seed${SEED}.npy" \
            --max_iter "$MAX_ITER" --logreg_c "$LOGREG_C" --seed "$SEED" \
            > "$cap_dir/metrics.tmp"; then
            mv "$cap_dir/metrics.tmp" "$metrics"
          else
            rm -f "$cap_dir/metrics.tmp"
            echo "probe failed: dataset=$dataset bird=$bird model=$model K=$cap" >&2
          fi
        done
        [[ "$CLEAN_EMBEDDINGS" == 1 ]] && rm -rf "$embeddings"
      done
    done < <(linear_probe_birds "$annotations")
  done
}
