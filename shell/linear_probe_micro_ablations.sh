#!/usr/bin/env bash
# Every model in the micro-model linear-probe ablation table.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
OUT_ROOT=${OUT_ROOT:-$ROOT/results/linear_probe_micro_ablations_kmax_pca128_logreg_c0001}

LINEAR_PROBE_MODELS=(
  "xcl_micro_100k_p32x1_random|songmae|runs/xcl_micro_100k_p32x1_random|model_step_099999.pth"
  "Xcl_micro_100k_p32x1_default|songmae|runs/Xcl_micro_100k_p32x1_default|model_step_099999.pth"
  "Xcl_micro_100k_p128x1_default|songmae|runs/Xcl_micro_100k_p128x1_default|model_step_099999.pth"
  "Xcl_micro_100k_p16x1_default|songmae|runs/Xcl_micro_100k_p16x1_default|model_step_099999.pth"
  "xcl_micro_100k_p32x4_qknorm_gelu_lr1e-4_bs128|songmae|runs/xcl_micro_100k_p32x4_qknorm_gelu_lr1e-4_bs128|model_step_099999.pth"
  "Xcl_micro_100k_p4x4_default|songmae|runs/Xcl_micro_100k_p4x4_default|model_step_099999.pth"
  "Xcl_micro_100k_p32x1_c0025|songmae|runs/Xcl_micro_100k_p32x1_c0025|model_step_099999.pth"
  "Xcl_micro_100k_p32x1_c005|songmae|runs/Xcl_micro_100k_p32x1_c005|model_step_099999.pth"
  "Xcl_micro_100k_p32x1_c010|songmae|runs/Xcl_micro_100k_p32x1_c010|model_step_099999.pth"
  "Xcl_micro_100k_p32x1_c020|songmae|runs/Xcl_micro_100k_p32x1_c020|model_step_099999.pth"
  "xcl_micro_100k_p32x4_c0025|songmae|runs/xcl_micro_100k_p32x4_c0025|model_step_099999.pth"
  "xcl_micro_100k_p32x4_c005|songmae|runs/xcl_micro_100k_p32x4_c005|model_step_099999.pth"
  "xcl_micro_100k_p32x4_c010|songmae|runs/xcl_micro_100k_p32x4_c010|model_step_099999.pth"
  "xcl_micro_100k_p32x4_c020|songmae|runs/xcl_micro_100k_p32x4_c020|model_step_099999.pth"
)

source shell/linear_probe_lib.sh
run_linear_probe_suite
