#!/usr/bin/env bash
# Capped-label probes for the large SongMAEs, BirdAVES, and HuBERT.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
OUT_ROOT=${OUT_ROOT:-$ROOT/results/linear_probe_models_by_k}
LABEL_CAPS=${LABEL_CAPS:-"1 5 10 20 50 100"}

LINEAR_PROBE_MODELS=(
  "xcl_large_500k_p32x1_c005|songmae|runs/xcl_large_500k_p32x1_c005|model_step_499999.pth"
  "xcl_large_500k_p32x4_c010|songmae|runs/xcl_large_500k_p32x4_c010|model_step_499999.pth"
  "birdaves_biox_base|aves||"
  "hubert_base_ls960|hubert||"
)

source shell/linear_probe_lib.sh
run_capped_linear_probe_suite
