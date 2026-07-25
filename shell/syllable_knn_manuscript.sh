#!/usr/bin/env bash
set -euo pipefail

worker="${1:?worker index required}"
workers="${NUM_WORKERS:-3}"
root="${RESULTS_ROOT:-results/knn/manuscript_reference_fit}"

export WORKER_INDEX="$worker"
export NUM_WORKERS="$workers"

OUT_ROOT="$root/micro_ablations" bash shell/syllable_knn_micro_ablations.sh
OUT_ROOT="$root/checkpoints" bash shell/syllable_knn_songmae_checkpoints.sh
OUT_ROOT="$root/best_layers" bash shell/syllable_knn_four_models_layers.sh
