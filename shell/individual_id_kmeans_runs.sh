#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/george-vengrovski/Documents/projects/TinyBird"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="$ROOT/individual_id/run_individual_id_kmeans.py"
RESULTS_DIR="$ROOT/results/individual_id_kmeans"

run_zf() {
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --species Zebra_Finch \
    --spec_dir /media/george-vengrovski/disk2/specs/zf_64hop_32khz \
    --run_dir "$ROOT/runs/xcm_voronoi_mask_no_normalize_32h_10w_zf_continue10k_bs24_20260302_133416" \
    --annotation_json "$ROOT/files/zf_annotations.json" \
    --out_dir "$RESULTS_DIR/zf" \
    --recording_mode events \
    --songs_per_bird 30 \
    --pool_window 30 \
    --pool_hop 5 \
    --embedding_variant before
}

run_bf() {
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --species bf \
    --spec_dir /media/george-vengrovski/disk2/specs/bf_64hop_32khz \
    --run_dir "$ROOT/runs/xcm_voronoi_mask_no_normalize_32h_10w_bf_continue10k_bs24_20260311_131742" \
    --annotation_json "$ROOT/files/bf_annotations.json" \
    --out_dir "$RESULTS_DIR/bf" \
    --recording_mode events \
    --songs_per_bird 30 \
    --pool_window 30 \
    --pool_hop 5 \
    --embedding_variant before
}

run_canary() {
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --species canary \
    --spec_dir /media/george-vengrovski/disk2/specs/canary_individual_identification_64hop_32khz \
    --run_dir "$ROOT/runs/xcm_voronoi_mask_no_normalize_32h_10w_canary_continue10k_bs24_20260312_141845" \
    --annotation_json "$ROOT/files/canary_annotations_for_individual_id.json" \
    --out_dir "$RESULTS_DIR/canary_w30_h5_k100" \
    --recording_mode events \
    --songs_per_bird 30 \
    --pool_window 30 \
    --pool_hop 5 \
    --embedding_variant before \
    --n_clusters 100
}

usage() {
  echo "Usage: $0 [all|zf|bf|canary]"
}

mkdir -p "$RESULTS_DIR"

target="${1:-all}"
case "$target" in
  all)
    # run_zf
    # run_bf
    run_canary
    ;;
  zf)
    run_zf
    ;;
  bf)
    run_bf
    ;;
  canary)
    run_canary
    ;;
  *)
    usage
    exit 1
    ;;
esac
