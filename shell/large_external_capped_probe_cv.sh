#!/usr/bin/env bash
# All label caps for the two large SongMAEs and external baselines.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)

OUT_ROOT=${OUT_ROOT:-$ROOT/results/large_external_capped_probe_cv} \
LABEL_CAPS="2 5 10 20 50" \
FOLDS=3 \
NUM_TIMEBINS=${NUM_TIMEBINS:-200000} \
STEPS=${STEPS:-1000} \
BATCH_SIZE=${BATCH_SIZE:-256} \
ALL_BIRDS=1 \
MODEL_FILTER="large_p32x4_trained_500k large_p32x1_trained_375k hubert_base_ls960 birdaves_base" \
  bash shell/syllable_capped_probe_cv.sh
