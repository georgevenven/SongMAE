#!/usr/bin/env bash
# Raw-dimension cosine UMAPs for four models on every annotated zf, bf, and canary bird.
set -u

cd "$(dirname "$0")/.."
ROOT=$(pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/george-vengrovski/anaconda3/envs/mae/bin/python}
OUT_ROOT=${OUT_ROOT:-/media/george-vengrovski/disk1/tinybird_umap_50birds_4models_250k_20260720}
NUM_TIMEBINS=${NUM_TIMEBINS:-250000}
MAX_POINTS=${MAX_POINTS:-250000}
DATASET_FILTER=${DATASET_FILTER:-}
BIRD_FILTER=${BIRD_FILTER:-}
MODEL_FILTER=${MODEL_FILTER:-}
MODEL_GROUP=${MODEL_GROUP:-large_comparison}
DRY_RUN=${DRY_RUN:-0}

WAV_ROOT=/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae
BIRDAVES_MODEL=$ROOT/files/birdaves-biox-base.torchaudio.pt
BIRDAVES_CONFIG=$ROOT/files/birdaves-biox-base.torchaudio.model_config.json

DATASETS=(
  "zf|files/annotation jsons/zf_annotations.json|/media/george-vengrovski/disk2/specs/zebra_finch_5ms"
  "bf|files/annotation jsons/bf_annotations.json|/media/george-vengrovski/disk2/specs/bengalese_finch_5ms"
  "canary|files/annotation jsons/canary_annotations.json|/media/george-vengrovski/disk2/specs/canary_5ms"
)

case "$MODEL_GROUP" in
  large_comparison)
    MODELS=(
      "songmae_32x1|songmae|runs/xcl_large_500k_p32x1_c005|model_step_499999.pth|11"
      "songmae_32x4|songmae|runs/xcl_large_500k_p32x4_c010|model_step_499999.pth|9"
      "birdaves|aves|-|-|6"
      "hubert|hubert|-|-|0"
    )
    ;;
  micro_base)
    MODELS=(
      "micro_32x1|songmae|runs/xcl_micro_500k_p32x1_c005|model_step_499999.pth|5"
      "micro_32x4|songmae|runs/xcl_micro_500k_p32x4_c010|model_step_499999.pth|5"
      "base_32x1|songmae|runs/xcl_base_500k_p32x1_c005|model_step_499999.pth|5"
      "base_32x4|songmae|runs/xcl_base_500k_p32x4_c010|model_step_499999.pth|5"
    )
    ;;
  *) echo "unknown MODEL_GROUP: $MODEL_GROUP" >&2; exit 2 ;;
esac

selected() {
  [[ -z "$2" || " $2 " == *" $1 "* ]]
}

birds() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
print("\n".join(sorted({row["recording"]["bird_id"] for row in data["recordings"]})))
PY
}

complete() {
  local out=$1 extractor=$2
  local model_dir=$out/$extractor
  [[ -f "$out/summary.json" && -f "$model_dir/umap_points.npy" && \
     -f "$model_dir/labels.npy" && -f "$model_dir/umap.png" && \
     -f "$model_dir/embeddings/metadata.json" && \
     -f "$model_dir/embeddings/encoded_embeddings.npy" && \
     -f "$model_dir/embeddings/recording_stem.npy" && \
     -f "$model_dir/embeddings/token_start_ms.npy" && \
     -f "$model_dir/embeddings/token_end_ms.npy" ]]
}

run_one() {
  local dataset=$1 annotations=$2 specs=$3 bird=$4 name=$5 extractor=$6 run_dir=$7 checkpoint=$8 layer=$9
  local out=$OUT_ROOT/$dataset/$bird/$name
  if complete "$out" "$extractor"; then
    echo "exists: dataset=$dataset bird=$bird model=$name"
    return
  fi

  local reuse=()
  if [[ -f "$out/$extractor/embeddings/metadata.json" && \
        -f "$out/$extractor/embeddings/encoded_embeddings.npy" && \
        -f "$out/$extractor/embeddings/token_start_ms.npy" && \
        -f "$out/$extractor/embeddings/token_end_ms.npy" ]]; then
    reuse=(--reuse)
  elif [[ "$DRY_RUN" != 1 ]]; then
    [[ "$out" == "$OUT_ROOT/"* ]]
    rm -rf "$out"
  fi

  local command=(
    "$PYTHON_BIN" src/embeddings/syllable_umap.py
    --spec_dir "$specs"
    --annotation_file "$annotations"
    --out_dir "$out"
    --models "$extractor"
    --wav_dir "$WAV_ROOT"
    --bird "$bird"
    --recording_mode events
    --num_timebins "$NUM_TIMEBINS"
    --max_points "$MAX_POINTS"
    --minimal
    --encoder_layer_idx "$layer"
    --target_feature_type end_of_block
    --umap_neighbors 100
    --umap_min_dist 0
    --umap_metric cosine
    --seed 42
    "${reuse[@]}"
  )
  if [[ "$extractor" == songmae ]]; then
    command+=(--songmae_run_dir "$run_dir" --checkpoint "$checkpoint")
  elif [[ "$extractor" == aves ]]; then
    command+=(--aves_model_path "$BIRDAVES_MODEL" --aves_config_path "$BIRDAVES_CONFIG")
  fi

  echo ">> dataset=$dataset bird=$bird model=$name"
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    return
  fi
  "${command[@]}"
}

failures=0
mkdir -p "$OUT_ROOT"
for dataset_row in "${DATASETS[@]}"; do
  IFS="|" read -r dataset annotations specs <<< "$dataset_row"
  selected "$dataset" "$DATASET_FILTER" || continue
  while read -r bird; do
    selected "$bird" "$BIRD_FILTER" || continue
    for model_row in "${MODELS[@]}"; do
      IFS="|" read -r name extractor run_dir checkpoint layer <<< "$model_row"
      selected "$name" "$MODEL_FILTER" || continue
      if ! run_one "$dataset" "$annotations" "$specs" "$bird" \
          "$name" "$extractor" "$run_dir" "$checkpoint" "$layer"; then
        echo "failed: dataset=$dataset bird=$bird model=$name" >&2
        failures=$((failures + 1))
      fi
    done
  done < <(birds "$annotations")
done

echo "finished: failures=$failures"
((failures == 0))
