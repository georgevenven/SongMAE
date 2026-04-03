#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/george-vengrovski/Documents/projects/TinyBird"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="$ROOT/individual_id/run_individual_id_umap.py"
RESULTS_DIR="$ROOT/results/individual_id_umap"
POOL_WINDOW="${POOL_WINDOW:-30}"
POOL_HOP="${POOL_HOP:-5}"
RESULTS_SUFFIX="${RESULTS_SUFFIX:-w${POOL_WINDOW}_h${POOL_HOP}}"

: <<'OLD_BIRDS'
run_zf() {
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --encoder SongMAE \
    --species Zebra_Finch \
    --spec_dir /media/george-vengrovski/disk2/specs/zf_64hop_32khz \
    --run_dir "$ROOT/runs/xcm_voronoi_mask_no_normalize_32h_10w_zf_continue10k_bs24_20260302_133416" \
    --annotation_json "$ROOT/files/zf_annotations.json" \
    --out_dir "$RESULTS_DIR/zf_w30_h5" \
    --recording_mode events \
    --songs_per_bird 30 \
    --pool_window 30 \
    --pool_hop 5 \
    --pool_mode mean \
    --normalization_preset zscore_rescaled \
    --audio_params_stats_dir /media/george-vengrovski/disk2/specs/zf_64hop_32khz
}

run_bf() {
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --encoder SongMAE \
    --species bf \
    --spec_dir /media/george-vengrovski/disk2/specs/bf_64hop_32khz \
    --run_dir "$ROOT/runs/xcm_voronoi_mask_no_normalize_32h_10w_bf_continue10k_bs24_20260311_131742" \
    --annotation_json "$ROOT/files/bf_annotations.json" \
    --out_dir "$RESULTS_DIR/bf_w30_h5" \
    --recording_mode events \
    --songs_per_bird 30 \
    --pool_window 30 \
    --pool_hop 5 \
    --pool_mode mean \
    --normalization_preset zscore_rescaled \
    --audio_params_stats_dir /media/george-vengrovski/disk2/specs/bf_64hop_32khz
}

run_canary() {
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --encoder SongMAE \
    --species canary \
    --spec_dir /media/george-vengrovski/disk2/specs/canary_individual_identification_64hop_32khz \
    --run_dir "$ROOT/runs/xcm_voronoi_mask_no_normalize_32h_10w_canary_continue10k_bs24_20260312_141845" \
    --annotation_json "$ROOT/files/canary_annotations_for_individual_id.json" \
    --out_dir "$RESULTS_DIR/canary_w30_h5" \
    --recording_mode events \
    --songs_per_bird 30 \
    --pool_window 30 \
    --pool_hop 5 \
    --pool_mode mean \
    --normalization_preset zscore_rescaled \
    --audio_params_stats_dir /media/george-vengrovski/disk2/specs/canary_individual_identification_64hop_32khz
}
OLD_BIRDS

run_chiffchaff() {
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --encoder SongMAE \
    --species chiffchaff \
    --spec_dir /media/george-vengrovski/disk2/specs/chiffchaff_64hop_32khz \
    --run_dir "$ROOT/runs/xcm_voronoi_mask_no_normalize_32h_10w" \
    --annotation_json "$ROOT/files/chiffchaff_annotations.json" \
    --out_dir "$RESULTS_DIR/chiffchaff_${RESULTS_SUFFIX}" \
    --recording_mode events \
    --songs_per_bird 30 \
    --pool_window "$POOL_WINDOW" \
    --pool_hop "$POOL_HOP" \
    --pool_mode mean \
    --songmae_input_normalization audio_params \
    --songmae_input_normalization_stats_dir /media/george-vengrovski/disk2/specs/chiffchaff_64hop_32khz
}

run_little_owl() {
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --encoder SongMAE \
    --species little_owl \
    --spec_dir /media/george-vengrovski/disk2/specs/little_owl_64hop_32khz \
    --run_dir "$ROOT/runs/xcm_voronoi_mask_no_normalize_32h_10w" \
    --annotation_json "$ROOT/files/little_owl_annotations.json" \
    --out_dir "$RESULTS_DIR/little_owl_${RESULTS_SUFFIX}" \
    --recording_mode events \
    --songs_per_bird 30 \
    --pool_window "$POOL_WINDOW" \
    --pool_hop "$POOL_HOP" \
    --pool_mode mean \
    --songmae_input_normalization audio_params \
    --songmae_input_normalization_stats_dir /media/george-vengrovski/disk2/specs/little_owl_64hop_32khz
}

run_tree_pipit() {
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --encoder SongMAE \
    --species tree_pipit \
    --spec_dir /media/george-vengrovski/disk2/specs/tree_pipit_64hop_32khz \
    --run_dir "$ROOT/runs/xcm_voronoi_mask_no_normalize_32h_10w" \
    --annotation_json "$ROOT/files/tree_pipit_annotations.json" \
    --out_dir "$RESULTS_DIR/tree_pipit_${RESULTS_SUFFIX}" \
    --recording_mode events \
    --songs_per_bird 30 \
    --pool_window "$POOL_WINDOW" \
    --pool_hop "$POOL_HOP" \
    --pool_mode mean \
    --songmae_input_normalization audio_params \
    --songmae_input_normalization_stats_dir /media/george-vengrovski/disk2/specs/tree_pipit_64hop_32khz
}

run_european_starling() {
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --encoder SongMAE \
    --species european_starling \
    --spec_dir /media/george-vengrovski/disk2/specs/european_starling_64hop_32khz \
    --run_dir "$ROOT/runs/xcm_voronoi_mask_no_normalize_32h_10w" \
    --annotation_json "$ROOT/files/european_starling_annotations_unprefixed.json" \
    --out_dir "$RESULTS_DIR/european_starling_${RESULTS_SUFFIX}" \
    --recording_mode events \
    --songs_per_bird 30 \
    --pool_window "$POOL_WINDOW" \
    --pool_hop "$POOL_HOP" \
    --pool_mode mean \
    --songmae_input_normalization audio_params \
    --songmae_input_normalization_stats_dir /media/george-vengrovski/disk2/specs/european_starling_64hop_32khz
}

usage() {
  echo "Usage: $0 [all|chiffchaff|little_owl|tree_pipit|european_starling]"
}

mkdir -p "$RESULTS_DIR"

target="${1:-all}"
case "$target" in
  all)
    run_chiffchaff
    run_little_owl
    run_tree_pipit
    run_european_starling
    ;;
  chiffchaff)
    run_chiffchaff
    ;;
  little_owl)
    run_little_owl
    ;;
  tree_pipit)
    run_tree_pipit
    ;;
  european_starling)
    run_european_starling
    ;;
  *)
    usage
    exit 1
    ;;
esac
