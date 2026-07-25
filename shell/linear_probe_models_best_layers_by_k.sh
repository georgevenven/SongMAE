#!/usr/bin/env bash
# PCA-128 capped-label probes at each model's pooled best kNN layer.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
OUT_ROOT=${OUT_ROOT:-$ROOT/results/linear_probe_models_best_layers_by_k_pca128_logreg_c0001}
LABEL_CAPS=${LABEL_CAPS:-"1 5 10 20 50 100"}

LINEAR_PROBE_MODELS=(
  "xcl_large_500k_p32x1_c005|songmae|runs/xcl_large_500k_p32x1_c005|model_step_499999.pth|11"
  "xcl_large_500k_p32x4_c010|songmae|runs/xcl_large_500k_p32x4_c010|model_step_499999.pth|10"
  "birdaves_biox_base|aves|||7"
  "hubert_base_ls960|hubert|||0"
)

source shell/linear_probe_lib.sh
run_capped_linear_probe_suite
