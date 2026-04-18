#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/george-vengrovski/Documents/projects/TinyBird"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="$ROOT/individual_id/individual_identification_linear_probe.py"
RESULTS_DIR="$ROOT/results/individual_id_linear_probe"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
PERCH_ENV_BIN="/home/george-vengrovski/anaconda3/envs/perch/bin/python"
PERCH_CUDNN_DIR="/home/george-vengrovski/anaconda3/envs/perch/lib/python3.11/site-packages/nvidia/cudnn/lib"

: "${IID_SPECIES:?Set IID_SPECIES before running this script.}"
: "${IID_SPEC_DIR:?Set IID_SPEC_DIR before running this script.}"
: "${IID_RUN_DIR:?Set IID_RUN_DIR before running this script.}"
: "${IID_ANNOTATION_JSON:?Set IID_ANNOTATION_JSON before running this script.}"
: "${IID_LINEAR_OUT_DIR:?Set IID_LINEAR_OUT_DIR before running this script.}"

IID_ENCODER="${IID_ENCODER:-SongMAE}"
IID_RECORDING_MODE="${IID_RECORDING_MODE:-events}"
IID_SONGS_PER_BIRD="${IID_SONGS_PER_BIRD:-30}"
IID_MAX_BIRDS="${IID_MAX_BIRDS:-0}"
IID_SEED="${IID_SEED:-42}"
IID_POOL_WINDOW="${IID_POOL_WINDOW:-30}"
# Linear probes default to non-overlapping pooled windows, "before" SongMAE
# embeddings, and train-only raw-audio speed perturbation so SongMAE and AVES
# see the same underlying augmentation.
IID_LINEAR_POOL_HOP="${IID_LINEAR_POOL_HOP:-$IID_POOL_WINDOW}"
IID_POOL_MODE="${IID_POOL_MODE:-mean}"
IID_LINEAR_VAL_FRACTION="${IID_LINEAR_VAL_FRACTION:-0.2}"
IID_LINEAR_C="${IID_LINEAR_C:-1.0}"
IID_LINEAR_MAX_ITER="${IID_LINEAR_MAX_ITER:-2000}"
IID_LINEAR_NORMALIZATION_PRESET="${IID_LINEAR_NORMALIZATION_PRESET:-zscore_rescaled}"
IID_LINEAR_AUDIO_PARAMS_STATS_DIR="${IID_LINEAR_AUDIO_PARAMS_STATS_DIR:-$IID_SPEC_DIR}"
IID_LINEAR_SONGMAE_EMBEDDING_VARIANT="${IID_LINEAR_SONGMAE_EMBEDDING_VARIANT:-before}"
IID_LINEAR_WINDOW_MEAN_POOL="${IID_LINEAR_WINDOW_MEAN_POOL:-0}"
IID_LINEAR_WINDOW_CONCAT="${IID_LINEAR_WINDOW_CONCAT:-0}"
IID_LINEAR_WINDOW_TOKEN_PROBE="${IID_LINEAR_WINDOW_TOKEN_PROBE:-0}"
IID_LINEAR_TRAIN_AUDIO_SPEED_MIN_PCT="${IID_LINEAR_TRAIN_AUDIO_SPEED_MIN_PCT:-0.0}"
IID_LINEAR_TRAIN_AUDIO_SPEED_MAX_PCT="${IID_LINEAR_TRAIN_AUDIO_SPEED_MAX_PCT:-0.5}"
IID_AVES_MODEL_PATH="${IID_AVES_MODEL_PATH:-$ROOT/files/aves-base-bio.torchaudio.pt}"
IID_AVES_CONFIG_PATH="${IID_AVES_CONFIG_PATH:-$ROOT/files/aves-base-bio.torchaudio.model_config.json}"
IID_PERCH_MODEL_NAME="${IID_PERCH_MODEL_NAME:-perch_v2}"
IID_PERCH_AUDIO_SR="${IID_PERCH_AUDIO_SR:-32000}"
IID_PERCH_WINDOW_SECONDS="${IID_PERCH_WINDOW_SECONDS:-5.0}"
IID_WAV_ROOT="${IID_WAV_ROOT:-${IID_AVES_WAV_ROOT:-}}"
IID_WAV_MANIFEST="${IID_WAV_MANIFEST:-${IID_AVES_WAV_MANIFEST:-}}"
IID_WAV_EXTS="${IID_WAV_EXTS:-${IID_AVES_WAV_EXTS:-.wav,.flac,.ogg,.mp3}}"
IID_AVES_AUDIO_SR="${IID_AVES_AUDIO_SR:-16000}"
IID_AUDIO_CONTEXT_SECONDS="${IID_AUDIO_CONTEXT_SECONDS:-2.0}"

if [[ "$IID_ENCODER" == "Perch" ]]; then
  if [[ "$PYTHON_BIN" == "python" ]]; then
    PYTHON_BIN="$PERCH_ENV_BIN"
  fi
  export LD_LIBRARY_PATH="$PERCH_CUDNN_DIR:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "$RESULTS_DIR"

cmd=(
  "$PYTHON_BIN" "$SCRIPT_PATH"
  --encoder "$IID_ENCODER"
  --species "$IID_SPECIES"
  --spec_dir "$IID_SPEC_DIR"
  --run_dir "$IID_RUN_DIR"
  --annotation_json "$IID_ANNOTATION_JSON"
  --out_dir "$IID_LINEAR_OUT_DIR"
  --recording_mode "$IID_RECORDING_MODE"
  --songs_per_bird "$IID_SONGS_PER_BIRD"
  --max_birds "$IID_MAX_BIRDS"
  --seed "$IID_SEED"
  --pool_window "$IID_POOL_WINDOW"
  --pool_hop "$IID_LINEAR_POOL_HOP"
  --pool_mode "$IID_POOL_MODE"
  --val_fraction "$IID_LINEAR_VAL_FRACTION"
  --c "$IID_LINEAR_C"
  --max_iter "$IID_LINEAR_MAX_ITER"
)

if [[ -n "${IID_CHECKPOINT:-}" ]]; then
  cmd+=(--checkpoint "$IID_CHECKPOINT")
fi
if [[ "$IID_ENCODER" == "Spec" ]] && [[ -n "$IID_LINEAR_NORMALIZATION_PRESET" ]]; then
  cmd+=(--normalization_preset "$IID_LINEAR_NORMALIZATION_PRESET")
fi
if [[ "$IID_ENCODER" == "Spec" ]] && [[ -n "$IID_LINEAR_AUDIO_PARAMS_STATS_DIR" ]]; then
  cmd+=(--audio_params_stats_dir "$IID_LINEAR_AUDIO_PARAMS_STATS_DIR")
fi
if [[ -n "${IID_LINEAR_SPEC_NORMALIZATION:-}" ]]; then
  cmd+=(--spec_normalization "$IID_LINEAR_SPEC_NORMALIZATION")
fi
if [[ -n "${IID_LINEAR_SPEC_NORMALIZATION_STATS_DIR:-}" ]]; then
  cmd+=(--spec_normalization_stats_dir "$IID_LINEAR_SPEC_NORMALIZATION_STATS_DIR")
fi
if [[ -n "$IID_WAV_ROOT" ]]; then
  cmd+=(--wav_root "$IID_WAV_ROOT")
fi
if [[ -n "$IID_WAV_MANIFEST" ]]; then
  cmd+=(--wav_manifest "$IID_WAV_MANIFEST")
fi
if [[ -n "$IID_WAV_EXTS" ]]; then
  cmd+=(--wav_exts "$IID_WAV_EXTS")
fi
if [[ -n "$IID_AUDIO_CONTEXT_SECONDS" ]]; then
  cmd+=(--audio_context_seconds "$IID_AUDIO_CONTEXT_SECONDS")
fi
if [[ "$IID_LINEAR_TRAIN_AUDIO_SPEED_MAX_PCT" != "0.0" ]]; then
  cmd+=(--train_audio_speed_min_pct "$IID_LINEAR_TRAIN_AUDIO_SPEED_MIN_PCT")
  cmd+=(--train_audio_speed_max_pct "$IID_LINEAR_TRAIN_AUDIO_SPEED_MAX_PCT")
fi
if [[ "$IID_LINEAR_WINDOW_MEAN_POOL" == "1" ]]; then
  cmd+=(--window_mean_pool)
fi
if [[ "$IID_LINEAR_WINDOW_CONCAT" == "1" ]]; then
  cmd+=(--window_concat_pool)
fi
if [[ "$IID_LINEAR_WINDOW_TOKEN_PROBE" == "1" ]]; then
  cmd+=(--window_token_probe)
fi
if [[ "$IID_ENCODER" == "SongMAE" ]]; then
  cmd+=(--songmae_embedding_variant "$IID_LINEAR_SONGMAE_EMBEDDING_VARIANT")
fi
if [[ "$IID_ENCODER" == "AVES" ]]; then
  cmd+=(--aves_model_path "$IID_AVES_MODEL_PATH")
  cmd+=(--aves_config_path "$IID_AVES_CONFIG_PATH")
  cmd+=(--aves_audio_sr "$IID_AVES_AUDIO_SR")
fi
if [[ "$IID_ENCODER" == "Perch" ]]; then
  cmd+=(--perch_model_name "$IID_PERCH_MODEL_NAME")
  cmd+=(--perch_audio_sr "$IID_PERCH_AUDIO_SR")
  cmd+=(--perch_window_seconds "$IID_PERCH_WINDOW_SECONDS")
fi

"${cmd[@]}"
