#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/george-vengrovski/Documents/projects/TinyBird"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="$ROOT/individual_id/run_individual_id_umap.py"
RESULTS_DIR="$ROOT/results/individual_id_umap"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"

: "${IID_SPECIES:?Set IID_SPECIES before running this script.}"
: "${IID_SPEC_DIR:?Set IID_SPEC_DIR before running this script.}"
: "${IID_RUN_DIR:?Set IID_RUN_DIR before running this script.}"
: "${IID_ANNOTATION_JSON:?Set IID_ANNOTATION_JSON before running this script.}"
: "${IID_UMAP_OUT_DIR:?Set IID_UMAP_OUT_DIR before running this script.}"

IID_ENCODER="${IID_ENCODER:-SongMAE}"
IID_RECORDING_MODE="${IID_RECORDING_MODE:-events}"
IID_SONGS_PER_BIRD="${IID_SONGS_PER_BIRD:-30}"
IID_MAX_BIRDS="${IID_MAX_BIRDS:-0}"
IID_SEED="${IID_SEED:-42}"
IID_POOL_WINDOW="${IID_POOL_WINDOW:-30}"
IID_POOL_HOP="${IID_POOL_HOP:-5}"
IID_POOL_MODE="${IID_POOL_MODE:-mean}"
IID_UMAP_NORMALIZATION_PRESET="${IID_UMAP_NORMALIZATION_PRESET:-zscore_rescaled}"
IID_UMAP_AUDIO_PARAMS_STATS_DIR="${IID_UMAP_AUDIO_PARAMS_STATS_DIR:-$IID_SPEC_DIR}"
IID_UMAP_SONGMAE_INPUT_NORMALIZATION="${IID_UMAP_SONGMAE_INPUT_NORMALIZATION:-}"
IID_UMAP_SONGMAE_INPUT_NORMALIZATION_STATS_DIR="${IID_UMAP_SONGMAE_INPUT_NORMALIZATION_STATS_DIR:-$IID_SPEC_DIR}"

mkdir -p "$RESULTS_DIR"

cmd=(
  "$PYTHON_BIN" "$SCRIPT_PATH"
  --encoder "$IID_ENCODER"
  --species "$IID_SPECIES"
  --spec_dir "$IID_SPEC_DIR"
  --run_dir "$IID_RUN_DIR"
  --annotation_json "$IID_ANNOTATION_JSON"
  --out_dir "$IID_UMAP_OUT_DIR"
  --recording_mode "$IID_RECORDING_MODE"
  --songs_per_bird "$IID_SONGS_PER_BIRD"
  --max_birds "$IID_MAX_BIRDS"
  --seed "$IID_SEED"
  --pool_window "$IID_POOL_WINDOW"
  --pool_hop "$IID_POOL_HOP"
  --pool_mode "$IID_POOL_MODE"
)

if [[ -n "${IID_CHECKPOINT:-}" ]]; then
  cmd+=(--checkpoint "$IID_CHECKPOINT")
fi
if [[ -n "$IID_UMAP_NORMALIZATION_PRESET" ]]; then
  cmd+=(--normalization_preset "$IID_UMAP_NORMALIZATION_PRESET")
fi
if [[ -n "$IID_UMAP_AUDIO_PARAMS_STATS_DIR" ]]; then
  cmd+=(--audio_params_stats_dir "$IID_UMAP_AUDIO_PARAMS_STATS_DIR")
fi
if [[ -n "${IID_UMAP_SPEC_NORMALIZATION:-}" ]]; then
  cmd+=(--spec_normalization "$IID_UMAP_SPEC_NORMALIZATION")
fi
if [[ -n "${IID_UMAP_SPEC_NORMALIZATION_STATS_DIR:-}" ]]; then
  cmd+=(--spec_normalization_stats_dir "$IID_UMAP_SPEC_NORMALIZATION_STATS_DIR")
fi
if [[ "$IID_ENCODER" == "SongMAE" ]]; then
  if [[ -n "$IID_UMAP_SONGMAE_INPUT_NORMALIZATION" ]]; then
    cmd+=(--songmae_input_normalization "$IID_UMAP_SONGMAE_INPUT_NORMALIZATION")
  fi
  if [[ -n "$IID_UMAP_SONGMAE_INPUT_NORMALIZATION_STATS_DIR" ]]; then
    cmd+=(--songmae_input_normalization_stats_dir "$IID_UMAP_SONGMAE_INPUT_NORMALIZATION_STATS_DIR")
  fi
fi
if [[ "${IID_UMAP_PER_BIRD_UMAPS:-0}" == "1" ]]; then
  cmd+=(--per_bird_umaps)
fi
if [[ -n "${IID_UMAP_NEIGHBORS:-}" ]]; then
  cmd+=(--umap_neighbors "$IID_UMAP_NEIGHBORS")
fi
if [[ -n "${IID_UMAP_MIN_DIST:-}" ]]; then
  cmd+=(--umap_min_dist "$IID_UMAP_MIN_DIST")
fi
if [[ -n "${IID_UMAP_METRIC:-}" ]]; then
  cmd+=(--umap_metric "$IID_UMAP_METRIC")
fi

"${cmd[@]}"
