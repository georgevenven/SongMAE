#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/george-vengrovski/anaconda3/envs/canary-vae/bin/python}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/scalar_mix_3birds_32s}"
NUM_TIMEBINS="${NUM_TIMEBINS:-720000}"
TRAIN_SECONDS="${TRAIN_SECONDS:-32}"

SONGMAE_RUN="$ROOT/runs/xcl_large_500k_p32x4_c010"
SONGMAE_CHECKPOINT="$SONGMAE_RUN/weights/model_step_499999.pth"
BIRDAVES_MODEL="$ROOT/files/birdaves-biox-base.torchaudio.pt"
BIRDAVES_CONFIG="$ROOT/files/birdaves-biox-base.torchaudio.model_config.json"
WAV_ROOT="/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae"

DATASETS=(
  "zf|Y453|files/annotation jsons/zf_annotations.json|/media/george-vengrovski/disk2/specs/zebra_finch_5ms"
  "bf|bird1|files/annotation jsons/bf_annotations.json|/media/george-vengrovski/disk2/specs/bengalese_finch_5ms"
  "canary|llb3_annot|files/annotation jsons/canary_annotations.json|/media/george-vengrovski/disk2/specs/canary_5ms"
)
MODELS=(songmae birdaves hubert)

extract() {
  local model="$1" bird="$2" annotations="$3" specs="$4" out="$5"
  case "$model" in
    songmae)
      "$PYTHON_BIN" -m src.core.extract_embedding \
        --run_dir "$SONGMAE_RUN" --checkpoint "$SONGMAE_CHECKPOINT" \
        --spec_dir "$specs" --json_path "$annotations" --bird "$bird" \
        --recording_mode events --out_dir "$out" --num_timebins "$NUM_TIMEBINS" \
        --target_feature_type end_of_block --all_layers --minimal
      ;;
    birdaves)
      "$PYTHON_BIN" src/external_models/aves.py \
        --spec_dir "$specs" --wav_dir "$WAV_ROOT" --annotation_file "$annotations" \
        --out_dir "$out" --bird "$bird" --recording_mode events \
        --aves_model_path "$BIRDAVES_MODEL" --aves_config_path "$BIRDAVES_CONFIG" \
        --model_name birdaves_biox_base --chunk_timebins 1000 \
        --num_timebins "$NUM_TIMEBINS" --all_layers
      ;;
    hubert)
      "$PYTHON_BIN" src/external_models/hubert.py \
        --spec_dir "$specs" --wav_dir "$WAV_ROOT" --annotation_file "$annotations" \
        --out_dir "$out" --bird "$bird" --recording_mode events \
        --model_name facebook/hubert-base-ls960 --chunk_timebins 1000 \
        --num_timebins "$NUM_TIMEBINS" --all_layers
      ;;
  esac
}

mkdir -p "$OUT_ROOT"
for dataset in "${DATASETS[@]}"; do
  IFS='|' read -r species bird annotations specs <<< "$dataset"
  for model in "${MODELS[@]}"; do
    run="$OUT_ROOT/$species/$bird/$model/train_${TRAIN_SECONDS}s"
    embeddings="$OUT_ROOT/$species/$bird/$model/embeddings"
    [[ -f "$run/metrics.json" ]] && continue

    echo "[$(date -Is)] $species $bird $model"
    mkdir -p "$run"
    extract "$model" "$bird" "$annotations" "$specs" "$embeddings"
    "$PYTHON_BIN" src/evals/syllable_classification.py \
      --embeddings "$embeddings" --annotations "$annotations" \
      --model scalar_mix --max_train_seconds "$TRAIN_SECONDS" \
      --epochs 20 --batch_size 128 --lr 0.001 --seed 42 --val_fraction 0.2 \
      --split_json "$run/split.json" > "$run/metrics.json.tmp"
    mv "$run/metrics.json.tmp" "$run/metrics.json"
    rm -rf "$embeddings"
  done
done
