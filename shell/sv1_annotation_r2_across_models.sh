#!/usr/bin/env bash

# For each model: extract every selected dataset, fit ONE SV1 across them all, score each
# dataset (per-ms, over the whole recording, vs its own unit coverage), then render the heatmap
# + spectrogram panels. Run with one dataset for a per-species SV1, or many for a shared one.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/sv1_annotation_r2_sweep}"
MODELS="${MODELS:-xcl_micro_500k_p32x1_default xcl_micro_500k_p32x4_default hubert}"
FEATURE_KEY="${FEATURE_KEY:-encoded_embeddings}"
OVERWRITE="${OVERWRITE:-0}"
CLEAN_EMBEDDINGS="${CLEAN_EMBEDDINGS:-1}"
MAX_SECONDS="${MAX_SECONDS:-1800}"
TIMEBINS_PER_SECOND="${TIMEBINS_PER_SECOND:-200}"
NUM_TIMEBINS="${NUM_TIMEBINS:-$((MAX_SECONDS * TIMEBINS_PER_SECOND))}"
RUN_PLOTS="${RUN_PLOTS:-1}"

SONGMAE_RUN_DIR="${SONGMAE_RUN_DIR:-$ROOT/runs/xcl_full_500k_bs256_5s_p32x10}"
XCL_MICRO_P128X1_RUN_DIR="${XCL_MICRO_P128X1_RUN_DIR:-$ROOT/runs/xcl_micro_500k_p128x1_default}"
XCL_MICRO_P16X1_RUN_DIR="${XCL_MICRO_P16X1_RUN_DIR:-$ROOT/runs/xcl_micro_500k_p16x1_default}"
XCL_MICRO_P32X1_RUN_DIR="${XCL_MICRO_P32X1_RUN_DIR:-$ROOT/runs/xcl_micro_500k_p32x1_default}"
XCL_MICRO_P32X4_RUN_DIR="${XCL_MICRO_P32X4_RUN_DIR:-$ROOT/runs/xcl_micro_500k_p32x4_default}"
AVES_MODEL_PATH="${AVES_MODEL_PATH:-$ROOT/files/aves-base-bio.torchaudio.pt}"
AVES_CONFIG_PATH="${AVES_CONFIG_PATH:-$ROOT/files/aves-base-bio.torchaudio.model_config.json}"
AVES_AUDIO_SR="${AVES_AUDIO_SR:-16000}"
HUBERT_MODEL_NAME="${HUBERT_MODEL_NAME:-facebook/hubert-large-ll60k}"
HUBERT_AUDIO_SR="${HUBERT_AUDIO_SR:-16000}"
WAV_EXTS="${WAV_EXTS:-.wav,.flac,.ogg,.mp3}"

# dataset|annotation json|spec_dir|wav_dir|recording_mode
# full_recordings so SV1 is fit on the whole recording (song + silence) and R^2 is scored over
# the whole recording vs unit coverage.
# Positional args filter this list, e.g. `bash shell/sv1_annotation_r2_across_models.sh american_robin`.
DATASETS=(
  "zf|files/annotation jsons/zf_annotations.json|/media/george-vengrovski/disk2/specs/zebra_finch_5ms|/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae|full_recordings"
  "bf|files/annotation jsons/bf_annotations.json|/media/george-vengrovski/disk2/specs/bengalese_finch_5ms|/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae|full_recordings"
  "canary|files/annotation jsons/canary_annotations.json|/media/george-vengrovski/disk2/specs/canary_5ms|/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae|full_recordings"
  "american_robin|files/annotation jsons/american_robin_annotations.json|/media/george-vengrovski/disk2/specs/american_robin_5ms|/media/george-vengrovski/disk2/raw_data/american_robin/data/rmbl_robin/RMBL-Robin/data|full_recordings"
  # great_tit disabled: unreliable unit labels (see readme note). Re-enable if relabeled.
  # "great_tit|files/annotation jsons/great_tit_annotations.json|/media/george-vengrovski/disk2/specs/great_tit_5ms|/media/george-vengrovski/disk2/raw_data/great_tit|full_recordings"
  "swamp_sparrow|files/annotation jsons/swamp_sparrow_annotations.json|/media/george-vengrovski/disk2/specs/swamp_sparrow_5ms|/media/george-vengrovski/disk2/raw_data/swamp_sparrow/data/figshare_5625310/extracted_wavs|full_recordings"
)

usage() {
  echo "Usage: $0 [zf|bf|canary|american_robin|swamp_sparrow ...]" 1>&2
  : # swamp_sparrow JSON stems were de-prefixed to match spec files (see readme note).
}

selected_dataset() {
  local dataset="$1"
  if [[ "${#TARGETS[@]}" -eq 0 ]]; then
    return 0
  fi
  for target in "${TARGETS[@]}"; do
    [[ "$dataset" == "$target" ]] && return 0
  done
  return 1
}

songmae_run_dir() {
  case "$1" in
    songmae|songmae_random) echo "$SONGMAE_RUN_DIR" ;;
    xcl_micro_500k_p128x1_default) echo "$XCL_MICRO_P128X1_RUN_DIR" ;;
    xcl_micro_500k_p16x1_default) echo "$XCL_MICRO_P16X1_RUN_DIR" ;;
    xcl_micro_500k_p32x1_default) echo "$XCL_MICRO_P32X1_RUN_DIR" ;;
    xcl_micro_500k_p32x4_default) echo "$XCL_MICRO_P32X4_RUN_DIR" ;;
    *) return 1 ;;
  esac
}

# Extract the whole dataset (no --bird filter) so SV1 is computed over every recording.
extract_embeddings() {
  local model="$1" json="$2" spec_dir="$3" wav_dir="$4" mode="$5" out_dir="$6"
  case "$model" in
    songmae|songmae_random|xcl_micro_500k_p128x1_default|xcl_micro_500k_p16x1_default|xcl_micro_500k_p32x1_default|xcl_micro_500k_p32x4_default)
      cmd=(
        "$PYTHON_BIN" -m src.core.extract_embedding
        --run_dir "$(songmae_run_dir "$model")"
        --spec_dir "$spec_dir"
        --json_path "$json"
        --recording_mode "$mode"
        --out_dir "$out_dir"
        --num_timebins "$NUM_TIMEBINS"
      )
      if [[ -n "${SONGMAE_CHECKPOINT:-}" ]]; then cmd+=(--checkpoint "$SONGMAE_CHECKPOINT"); fi
      if [[ "$model" == "songmae_random" ]]; then cmd+=(--random_init); fi
      "${cmd[@]}"
      ;;
    aves)
      "$PYTHON_BIN" src/external_models/aves.py \
        --spec_dir "$spec_dir" \
        --wav_dir "$wav_dir" \
        --annotation_file "$json" \
        --out_dir "$out_dir" \
        --recording_mode "$mode" \
        --aves_model_path "$AVES_MODEL_PATH" \
        --aves_config_path "$AVES_CONFIG_PATH" \
        --audio_sr "$AVES_AUDIO_SR" \
        --wav_exts "$WAV_EXTS" \
        --num_timebins "$NUM_TIMEBINS"
      ;;
    hubert)
      "$PYTHON_BIN" src/external_models/hubert.py \
        --spec_dir "$spec_dir" \
        --wav_dir "$wav_dir" \
        --annotation_file "$json" \
        --out_dir "$out_dir" \
        --recording_mode "$mode" \
        --model_name "$HUBERT_MODEL_NAME" \
        --audio_sr "$HUBERT_AUDIO_SR" \
        --wav_exts "$WAV_EXTS" \
        --num_timebins "$NUM_TIMEBINS"
      ;;
    *)
      echo "Unknown model: $model" 1>&2
      return 1
      ;;
  esac
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TARGETS=("$@")
read -r -a MODEL_LIST <<< "$MODELS"
mkdir -p "$OUT_ROOT"

# One SV1 per model, fit across every selected dataset at once; each scored against its own units.
for model in "${MODEL_LIST[@]}"; do
  model_dir="$OUT_ROOT/$model"
  metrics_path="$model_dir/metrics.json"
  if [[ -f "$metrics_path" && "$OVERWRITE" != "1" ]]; then
    echo "exists: $metrics_path"
    continue
  fi
  rm -rf "$model_dir"
  mkdir -p "$model_dir/overlays"
  dataset_args=()
  for row in "${DATASETS[@]}"; do
    IFS="|" read -r dataset json spec_dir wav_dir recording_mode <<< "$row"
    selected_dataset "$dataset" || continue
    embed_dir="$model_dir/embeddings/$dataset"
    mkdir -p "$embed_dir"
    echo "extracting: dataset=$dataset model=$model"
    if ! extract_embeddings "$model" "$json" "$spec_dir" "$wav_dir" "$recording_mode" "$embed_dir"; then
      echo "extract failed: dataset=$dataset model=$model" 1>&2
      rm -rf "$embed_dir"
      continue
    fi
    dataset_args+=(--dataset "$dataset=$embed_dir=$json")
  done
  if [[ "${#dataset_args[@]}" -eq 0 ]]; then
    echo "no datasets extracted for model=$model" 1>&2
    continue
  fi
  echo "fitting SV1: model=$model datasets=$(( ${#dataset_args[@]} / 2 ))"
  if ! "$PYTHON_BIN" src/evals/sv1_annotation_r2.py \
    --model "$model" \
    "${dataset_args[@]}" \
    --feature_key "$FEATURE_KEY" \
    --out_json "$model_dir/metrics.tmp" \
    --overlay_dir "$model_dir/overlays" > /dev/null; then
    rm -f "$model_dir/metrics.tmp"
    echo "eval failed: model=$model" 1>&2
  else
    mv "$model_dir/metrics.tmp" "$metrics_path"
  fi
  if [[ "$CLEAN_EMBEDDINGS" == "1" ]]; then
    rm -rf "$model_dir/embeddings"
    echo "cleaned: $model_dir/embeddings"
  fi
done

if [[ "$RUN_PLOTS" == "1" ]]; then
  echo "plotting: $OUT_ROOT"
  "$PYTHON_BIN" src/plotting_utils/sv1_annotation_r2_heatmap.py --results_root "$OUT_ROOT"
fi
