#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/george-vengrovski/Documents/projects/TinyBird"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="$ROOT/individual_id/run_individual_id_cluster.py"
RESULTS_DIR="$ROOT/results/individual_id_cluster"

: "${IID_SPECIES:?Set IID_SPECIES before running this script.}"
: "${IID_SPEC_DIR:?Set IID_SPEC_DIR before running this script.}"
: "${IID_RUN_DIR:?Set IID_RUN_DIR before running this script.}"
: "${IID_ANNOTATION_JSON:?Set IID_ANNOTATION_JSON before running this script.}"
: "${IID_CLUSTER_OUT_DIR:?Set IID_CLUSTER_OUT_DIR before running this script.}"

IID_RECORDING_MODE="${IID_RECORDING_MODE:-events}"
IID_SONGS_PER_BIRD="${IID_SONGS_PER_BIRD:-30}"
IID_MAX_BIRDS="${IID_MAX_BIRDS:-0}"
IID_SEED="${IID_SEED:-42}"
IID_POOL_WINDOW="${IID_POOL_WINDOW:-30}"
IID_POOL_HOP="${IID_POOL_HOP:-5}"
IID_POOL_MODE="${IID_POOL_MODE:-mean}"
IID_CLUSTER_EMBEDDING_VARIANT="${IID_CLUSTER_EMBEDDING_VARIANT:-before}"
IID_CLUSTER_NORMALIZATION_PRESET="${IID_CLUSTER_NORMALIZATION_PRESET:-zscore_rescaled}"
IID_CLUSTER_AUDIO_PARAMS_STATS_DIR="${IID_CLUSTER_AUDIO_PARAMS_STATS_DIR:-$IID_SPEC_DIR}"
IID_CLUSTER_MIN_CLUSTER_SIZE="${IID_CLUSTER_MIN_CLUSTER_SIZE:-100}"
IID_CLUSTER_MIN_CLUSTER_HITS="${IID_CLUSTER_MIN_CLUSTER_HITS:-1}"
IID_CLUSTER_OVERLAP_THRESHOLD="${IID_CLUSTER_OVERLAP_THRESHOLD:-0.3}"

mkdir -p "$RESULTS_DIR"

cmd=(
  "$PYTHON_BIN" "$SCRIPT_PATH"
  --species "$IID_SPECIES"
  --spec_dir "$IID_SPEC_DIR"
  --run_dir "$IID_RUN_DIR"
  --annotation_json "$IID_ANNOTATION_JSON"
  --out_dir "$IID_CLUSTER_OUT_DIR"
  --recording_mode "$IID_RECORDING_MODE"
  --songs_per_bird "$IID_SONGS_PER_BIRD"
  --max_birds "$IID_MAX_BIRDS"
  --seed "$IID_SEED"
  --pool_window "$IID_POOL_WINDOW"
  --pool_hop "$IID_POOL_HOP"
  --pool_mode "$IID_POOL_MODE"
  --embedding_variant "$IID_CLUSTER_EMBEDDING_VARIANT"
  --min_cluster_size "$IID_CLUSTER_MIN_CLUSTER_SIZE"
  --min_cluster_hits "$IID_CLUSTER_MIN_CLUSTER_HITS"
  --overlap_threshold "$IID_CLUSTER_OVERLAP_THRESHOLD"
)

if [[ -n "${IID_CHECKPOINT:-}" ]]; then
  cmd+=(--checkpoint "$IID_CHECKPOINT")
fi
if [[ -n "$IID_CLUSTER_NORMALIZATION_PRESET" ]]; then
  cmd+=(--normalization_preset "$IID_CLUSTER_NORMALIZATION_PRESET")
fi
if [[ -n "$IID_CLUSTER_AUDIO_PARAMS_STATS_DIR" ]]; then
  cmd+=(--audio_params_stats_dir "$IID_CLUSTER_AUDIO_PARAMS_STATS_DIR")
fi
if [[ -n "${IID_CLUSTER_SONGMAE_INPUT_NORMALIZATION:-}" ]]; then
  cmd+=(--songmae_input_normalization "$IID_CLUSTER_SONGMAE_INPUT_NORMALIZATION")
fi
if [[ -n "${IID_CLUSTER_SONGMAE_INPUT_NORMALIZATION_STATS_DIR:-}" ]]; then
  cmd+=(--songmae_input_normalization_stats_dir "$IID_CLUSTER_SONGMAE_INPUT_NORMALIZATION_STATS_DIR")
fi
if [[ -n "${IID_CLUSTER_MIN_SAMPLES:-}" ]]; then
  cmd+=(--min_samples "$IID_CLUSTER_MIN_SAMPLES")
fi
if [[ "${IID_CLUSTER_DROP_SILENCE:-0}" == "1" ]]; then
  cmd+=(--drop_silence)
fi

"${cmd[@]}"
