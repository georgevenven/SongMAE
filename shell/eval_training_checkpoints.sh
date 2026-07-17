#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
STEP_INTERVAL="${STEP_INTERVAL:-50000}"
LINEAR_OUT_ROOT="${LINEAR_OUT_ROOT:-$ROOT/results/syllable_linear_probe_steps}"
RECON_OUT_ROOT="${RECON_OUT_ROOT:-$ROOT/results/recon_v_train_steps}"

RUNS=(
  xcl_large_500k_p32x4_c010
  xcl_large_500k_p32x1_c005
  xcl_base_500k_p32x4_c010
  xcl_base_500k_p32x1_c005
  xcl_micro_500k_p32x4_c010
  xcl_micro_500k_p32x1_c005
)
[[ "$#" -eq 0 ]] || RUNS=("$@")

for run in "${RUNS[@]}"; do
  run_dir="$ROOT/runs/$run"
  if [[ ! -f "$run_dir/config.json" ]] || ! compgen -G "$run_dir/weights/model_step_*.pth" > /dev/null; then
    echo "skipping: $run has no checkpoints"
    continue
  fi
  mapfile -t checkpoints < <(find "$run_dir/weights" -maxdepth 1 -name "model_step_*.pth" | sort -V)

  for path in "${checkpoints[@]}"; do
    checkpoint="$(basename "$path")"
    step="${checkpoint#model_step_}"
    step="${step%.pth}"
    step_number=$((10#$step))
    if (( step_number % STEP_INTERVAL != 0 && (step_number + 1) % STEP_INTERVAL != 0 )); then
      continue
    fi
    PYTHON_BIN="$PYTHON_BIN" \
      MODELS="$run" \
      SONGMAE_CHECKPOINT="$checkpoint" \
      OUT_ROOT="$LINEAR_OUT_ROOT/step_$step" \
      bash shell/linear_probe_across_models.sh
  done

  PYTHON_BIN="$PYTHON_BIN" \
    RUN_DIR="$run_dir" \
    OUT_ROOT="$RECON_OUT_ROOT/$run" \
    STEP_INTERVAL="$STEP_INTERVAL" \
    bash shell/reconstruction_mse_steps.sh
done
