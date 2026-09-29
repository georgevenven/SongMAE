#!/usr/bin/env bash
# Mac: after WAIT_LOG ends in "sweep complete", run the kNN layer sweep for the given SongMAE runs.
# Usage (repo root): SongMAE_TMLR_Revision/scripts/mac_seed_knn.sh WAIT_LOG RUN [RUN ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
wait_log=$1; shift
until tail -1 "$wait_log" 2>/dev/null | grep -q "sweep complete"; do sleep 60; done
D=$HOME/Documents/songmae_data
EXTRA_SONGMAE_RUNS="$*" MODEL_FILTER="$*" SPEC_ROOT=$D/specs WAV_ROOT=$D/wavs PYTHON_BIN=$HOME/miniforge3/envs/mae/bin/python \
  caffeinate -i SongMAE_TMLR_Revision/scripts/knn_layer_sweep_review_baselines.sh
