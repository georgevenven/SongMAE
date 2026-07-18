#!/usr/bin/env bash
# Three-fold capped-K=5 probes across SongMAE training checkpoints.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)

PYTHON_BIN=${PYTHON_BIN:-/home/george-vengrovski/anaconda3/envs/mae/bin/python}
STEP_INTERVAL=${STEP_INTERVAL:-25000}
OUT_ROOT=${OUT_ROOT:-$ROOT/results/syllable_capped_probe_steps}
NUM_TIMEBINS=${NUM_TIMEBINS:-200000}
STEPS=${STEPS:-1000}
BATCH_SIZE=${BATCH_SIZE:-256}

RUNS=(
  xcl_large_500k_p32x4_c010
  xcl_large_500k_p32x1_c005
  xcl_base_500k_p32x4_c010
  xcl_base_500k_p32x1_c005
  xcl_micro_500k_p32x4_c010
  xcl_micro_500k_p32x1_c005
)
[[ "$#" == 0 ]] || RUNS=("$@")

for run in "${RUNS[@]}"; do
  run_dir="$ROOT/runs/$run"
  [[ -f "$run_dir/config.json" ]] || { echo "missing run, skipping: $run" >&2; continue; }
  while read -r path; do
    checkpoint=$(basename "$path")
    step=${checkpoint#model_step_}
    step=${step%.pth}
    step_number=$((10#$step))
    (( step_number % STEP_INTERVAL == 0 || (step_number + 1) % STEP_INTERVAL == 0 )) || continue
    echo "checkpoint probe: model=$run step=$step"
    PYTHON_BIN="$PYTHON_BIN" \
    OUT_ROOT="$OUT_ROOT/step_$step" \
    MANIFEST_ROOT="$OUT_ROOT/manifests" \
    LABEL_CAPS=5 FOLDS=3 ALL_BIRDS=1 \
    NUM_TIMEBINS="$NUM_TIMEBINS" STEPS="$STEPS" BATCH_SIZE="$BATCH_SIZE" \
    CUSTOM_MODEL="$run" CUSTOM_RUN_DIR="$run_dir" CUSTOM_CHECKPOINT="$checkpoint" \
      bash shell/syllable_capped_probe_cv.sh
  done < <(find "$run_dir/weights" -maxdepth 1 -type f -name 'model_step_*.pth' | sort -V)
done
