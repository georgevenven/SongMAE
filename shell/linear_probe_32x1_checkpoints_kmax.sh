#!/usr/bin/env bash
# K=max probes over shared Large, Base, and Micro 32x1 checkpoints.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
OUT_ROOT=${OUT_ROOT:-$ROOT/results/linear_probe_32x1_checkpoints_kmax_pca128_logreg_c0001}
CHECKPOINTS=${CHECKPOINTS:-"000000 010000 050000 499999"}

LINEAR_PROBE_MODELS=()
for size in large base micro; do
  run="xcl_${size}_500k_p32x1_c005"
  for step in $CHECKPOINTS; do
    LINEAR_PROBE_MODELS+=("${size}_step_${step}|songmae|runs/$run|model_step_${step}.pth")
  done
done

source shell/linear_probe_lib.sh
run_linear_probe_suite
