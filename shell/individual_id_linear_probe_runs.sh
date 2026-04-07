#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/george-vengrovski/Documents/projects/TinyBird"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="$ROOT/individual_id/individual_identification_linear_probe.py"
RESULTS_DIR="$ROOT/results/individual_id_linear_probe"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

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
IID_POOL_HOP="${IID_POOL_HOP:-5}"
IID_POOL_MODE="${IID_POOL_MODE:-mean}"
IID_LINEAR_VAL_FRACTION="${IID_LINEAR_VAL_FRACTION:-0.2}"
IID_LINEAR_C="${IID_LINEAR_C:-1.0}"
IID_LINEAR_MAX_ITER="${IID_LINEAR_MAX_ITER:-2000}"
IID_LINEAR_NORMALIZATION_PRESET="${IID_LINEAR_NORMALIZATION_PRESET:-vanilla}"
IID_LINEAR_SONGMAE_EMBEDDING_VARIANT="${IID_LINEAR_SONGMAE_EMBEDDING_VARIANT:-before}"

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
  --pool_hop "$IID_POOL_HOP"
  --pool_mode "$IID_POOL_MODE"
  --val_fraction "$IID_LINEAR_VAL_FRACTION"
  --c "$IID_LINEAR_C"
  --max_iter "$IID_LINEAR_MAX_ITER"
  --normalization_preset "$IID_LINEAR_NORMALIZATION_PRESET"
)

if [[ -n "${IID_CHECKPOINT:-}" ]]; then
  cmd+=(--checkpoint "$IID_CHECKPOINT")
fi
if [[ -n "${IID_LINEAR_AUDIO_PARAMS_STATS_DIR:-}" ]]; then
  cmd+=(--audio_params_stats_dir "$IID_LINEAR_AUDIO_PARAMS_STATS_DIR")
fi
if [[ -n "${IID_LINEAR_SPEC_NORMALIZATION:-}" ]]; then
  cmd+=(--spec_normalization "$IID_LINEAR_SPEC_NORMALIZATION")
fi
if [[ -n "${IID_LINEAR_SPEC_NORMALIZATION_STATS_DIR:-}" ]]; then
  cmd+=(--spec_normalization_stats_dir "$IID_LINEAR_SPEC_NORMALIZATION_STATS_DIR")
fi
if [[ "$IID_ENCODER" == "SongMAE" ]]; then
  cmd+=(--songmae_embedding_variant "$IID_LINEAR_SONGMAE_EMBEDDING_VARIANT")
  if [[ -n "${IID_LINEAR_SONGMAE_FEATURE_NORMALIZATION:-}" ]]; then
    cmd+=(--songmae_feature_normalization "$IID_LINEAR_SONGMAE_FEATURE_NORMALIZATION")
  fi
  if [[ -n "${IID_LINEAR_SONGMAE_INPUT_NORMALIZATION:-}" ]]; then
    cmd+=(--songmae_input_normalization "$IID_LINEAR_SONGMAE_INPUT_NORMALIZATION")
  fi
  if [[ -n "${IID_LINEAR_SONGMAE_INPUT_NORMALIZATION_STATS_DIR:-}" ]]; then
    cmd+=(--songmae_input_normalization_stats_dir "$IID_LINEAR_SONGMAE_INPUT_NORMALIZATION_STATS_DIR")
  fi
fi

"${cmd[@]}"
