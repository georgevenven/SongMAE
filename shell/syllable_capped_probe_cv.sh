#!/usr/bin/env bash
# Capped-label linear probing across birds and encoders.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)

PYTHON_BIN=${PYTHON_BIN:-/home/george-vengrovski/anaconda3/envs/mae/bin/python}
OUT_ROOT=${OUT_ROOT:-$ROOT/results/syllable_capped_probe_cv}
MANIFEST_ROOT=${MANIFEST_ROOT:-$OUT_ROOT/manifests}
LABEL_CAPS=(${LABEL_CAPS:-1 2 5 10 20 50})
FOLDS=${FOLDS:-3}
PCA_COMPONENTS=${PCA_COMPONENTS:-128}
NUM_TIMEBINS=${NUM_TIMEBINS:-500000}
STEPS=${STEPS:-1000}
BATCH_SIZE=${BATCH_SIZE:-256}
SEED=${SEED:-42}
BACKBONE_SEED=${BACKBONE_SEED:-42}
CLEAN_EMBEDDINGS=${CLEAN_EMBEDDINGS:-1}
ALL_BIRDS=${ALL_BIRDS:-0}
DATASET_FILTER=${DATASET_FILTER:-}
BIRD_FILTER=${BIRD_FILTER:-}
MODEL_FILTER=${MODEL_FILTER:-}

WAV_ROOT=/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae
BIRDAVES_MODEL=$ROOT/files/birdaves-biox-base.torchaudio.pt
BIRDAVES_CONFIG=$ROOT/files/birdaves-biox-base.torchaudio.model_config.json
HUBERT_MODEL=facebook/hubert-base-ls960

DATASETS=(
  "zf|files/annotation jsons/zf_annotations.json|/media/george-vengrovski/disk2/specs/zebra_finch_5ms|Y389 S389 Y453"
  "bf|files/annotation jsons/bf_annotations.json|/media/george-vengrovski/disk2/specs/bengalese_finch_5ms|bird1 bird3 bird4"
  "canary|files/annotation jsons/canary_annotations.json|/media/george-vengrovski/disk2/specs/canary_5ms|llb11_annot llb3_annot llb16_annot"
)

# slug | extractor | run directory | checkpoint
MODELS=(
  "micro_p32x1_voronoi_c010_trained_100k|songmae|runs/Xcl_micro_100k_p32x1_c010|model_step_099999.pth"
  "large_p32x1_trained_375k|songmae|runs/xcl_large_500k_p32x1_c005|model_step_375000.pth"
  "large_p32x4_trained_500k|songmae|runs/xcl_large_500k_p32x4_c010|model_step_499999.pth"
  "birdaves_base|aves||"
  "hubert_base_ls960|hubert||"
  "large_p32x1_random_init|random|runs/xcl_large_500k_p32x1_c005|"
  "large_p32x4_random_init|random|runs/xcl_large_500k_p32x4_c010|"
  "micro_p32x1_random_masking_trained_100k|songmae|runs/xcl_micro_100k_p32x1_random|model_step_099999.pth"
  "micro_p32x1_voronoi_c010_random_init|random|runs/Xcl_micro_100k_p32x1_c010|"
)
if [[ -n "${CUSTOM_RUN_DIR:-}" ]]; then
  : "${CUSTOM_MODEL:?}" "${CUSTOM_CHECKPOINT:?}"
  MODELS=("$CUSTOM_MODEL|songmae|$CUSTOM_RUN_DIR|$CUSTOM_CHECKPOINT")
fi

selected() {
  [[ -z "$2" || " $2 " == *" $1 "* ]]
}

birds() {
  if [[ "$ALL_BIRDS" == 0 ]]; then
    tr ' ' '\n' <<< "$2"
    return
  fi
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
print("\n".join(sorted({row["recording"]["bird_id"] for row in data["recordings"]})))
PY
}

extract_embeddings() {
  local extractor=$1 run_dir=$2 checkpoint=$3 annotations=$4 specs=$5 bird=$6 out=$7
  rm -rf "$out" "$out.tmp"
  if [[ "$extractor" == aves ]]; then
    "$PYTHON_BIN" src/external_models/aves.py \
      --spec_dir "$specs" --wav_dir "$WAV_ROOT" --annotation_file "$annotations" \
      --out_dir "$out" --bird "$bird" --recording_mode events \
      --aves_model_path "$BIRDAVES_MODEL" --aves_config_path "$BIRDAVES_CONFIG" \
      --model_name birdaves_biox_base --audio_sr 16000 \
      --wav_exts .wav,.flac,.ogg,.mp3 --chunk_timebins 1000 --num_timebins "$NUM_TIMEBINS" \
      --balanced_events "$FOLDS" --event_seed "$SEED"
    return
  fi
  if [[ "$extractor" == hubert ]]; then
    "$PYTHON_BIN" src/external_models/hubert.py \
      --spec_dir "$specs" --wav_dir "$WAV_ROOT" --annotation_file "$annotations" \
      --out_dir "$out" --bird "$bird" --recording_mode events \
      --model_name "$HUBERT_MODEL" --audio_sr 16000 \
      --wav_exts .wav,.flac,.ogg,.mp3 --chunk_timebins 1000 --num_timebins "$NUM_TIMEBINS" \
      --balanced_events "$FOLDS" --event_seed "$SEED"
    return
  fi

  local args=()
  [[ -n "$checkpoint" ]] && args+=(--checkpoint "$checkpoint")
  [[ "$extractor" == random ]] && args+=(--random_init --random_seed "$BACKBONE_SEED")
  "$PYTHON_BIN" -m src.core.extract_embedding \
    --spec_dir "$specs" --run_dir "$run_dir" "${args[@]}" --out_dir "$out" \
    --json_path "$annotations" --num_timebins "$NUM_TIMEBINS" \
    --recording_mode events --bird "$bird" --minimal --target_feature_type end_of_block \
    --balanced_events "$FOLDS" --event_seed "$SEED"
}

mkdir -p "$OUT_ROOT"
for dataset_row in "${DATASETS[@]}"; do
  IFS="|" read -r dataset annotations specs dataset_birds <<< "$dataset_row"
  selected "$dataset" "$DATASET_FILTER" || continue
  while read -r bird; do
    selected "$bird" "$BIRD_FILTER" || continue
    manifest_dir="$MANIFEST_ROOT/$dataset/$bird"
    for model_row in "${MODELS[@]}"; do
      IFS="|" read -r model extractor run_dir checkpoint <<< "$model_row"
      selected "$model" "$MODEL_FILTER" || continue
      model_dir="$OUT_ROOT/$dataset/$bird/$model"
      embeddings="$model_dir/embeddings"
      complete=1
      for cap in "${LABEL_CAPS[@]}"; do
        result="$model_dir/cap_$(printf '%03d' "$cap")/metrics.json"
        manifest="$manifest_dir/cap_$(printf '%03d' "$cap").json"
        [[ -f "$result" && -f "$manifest" ]] || complete=0
      done
      if [[ "$complete" == 1 ]]; then
        echo "complete: dataset=$dataset bird=$bird model=$model"
        continue
      fi
      if [[ ! -f "$embeddings/metadata.json" ]]; then
        mkdir -p "$model_dir"
        started=$SECONDS
        echo "extracting: dataset=$dataset bird=$bird model=$model"
        extract_embeddings "$extractor" "$run_dir" "$checkpoint" "$annotations" "$specs" "$bird" "$embeddings"
        printf '%s\n' "$((SECONDS - started))" > "$model_dir/extraction_seconds.txt"
      fi

      pca_cache="$embeddings/pca_${PCA_COMPONENTS}_seed${SEED}.npy"
      for cap in "${LABEL_CAPS[@]}"; do
        result_dir="$model_dir/cap_$(printf '%03d' "$cap")"
        metrics="$result_dir/metrics.json"
        manifest="$manifest_dir/cap_$(printf '%03d' "$cap").json"
        if [[ -f "$metrics" && -f "$manifest" ]]; then
          echo "exists: $metrics"
          continue
        fi
        mkdir -p "$result_dir"
        manifest_args=(--manifest_in "$manifest")
        [[ -f "$manifest" ]] || manifest_args=(--manifest_out "$manifest")
        echo "probing: dataset=$dataset bird=$bird model=$model cap=$cap"
        "$PYTHON_BIN" src/evals/syllable_classification_capped.py \
          --embeddings "$embeddings" --annotations "$annotations" \
          --label_cap "$cap" \
          --folds "$FOLDS" "${manifest_args[@]}" \
          --pca_components "$PCA_COMPONENTS" --pca_cache "$pca_cache" \
          --steps "$STEPS" --batch_size "$BATCH_SIZE" --seed "$SEED" \
          > "$result_dir/metrics.tmp"
        mv "$result_dir/metrics.tmp" "$metrics"
      done
      if [[ "$CLEAN_EMBEDDINGS" == 1 ]]; then
        rm -rf "$embeddings"
      fi
    done
  done < <(birds "$annotations" "$dataset_birds")
done
