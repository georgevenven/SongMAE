#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/george-vengrovski/Documents/projects/TinyBird"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="$ROOT/individual_id/run_individual_id_umap.py"
RESULTS_DIR="$ROOT/results/individual_id_umap"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
POOL_WINDOW="${POOL_WINDOW:-30}"
POOL_HOP="${POOL_HOP:-5}"
RESULTS_SUFFIX="${RESULTS_SUFFIX:-w${POOL_WINDOW}_h${POOL_HOP}}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/merged_data2vec_from_xcm_40k}"

if [[ -z "${CHECKPOINT:-}" ]]; then
  CHECKPOINT="$(find "$RUN_DIR/weights" -maxdepth 1 -type f -name 'model_step_*.pth' | sort | tail -n 1 | xargs -r basename)"
fi

if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "No model_step_*.pth checkpoint found in $RUN_DIR/weights" >&2
  exit 1
fi

CHECKPOINT_STEP="$(printf '%s\n' "$CHECKPOINT" | sed -E 's/^model_step_0*([0-9]+)\.pth$/\1/')"
CHECKPOINT_SUFFIX="${CHECKPOINT_SUFFIX:-step${CHECKPOINT_STEP}}"

run_zf() {
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --encoder SongMAE \
    --species Zebra_Finch \
    --spec_dir /media/george-vengrovski/disk2/specs/zf_64hop_32khz \
    --run_dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --annotation_json "$ROOT/files/zf_annotations.json" \
    --out_dir "$RESULTS_DIR/zf_data2vec_${CHECKPOINT_SUFFIX}_${RESULTS_SUFFIX}" \
    --recording_mode events \
    --songs_per_bird 30 \
    --pool_window "$POOL_WINDOW" \
    --pool_hop "$POOL_HOP" \
    --pool_mode mean \
    --songmae_input_normalization audio_params \
    --songmae_input_normalization_stats_dir /media/george-vengrovski/disk2/specs/zf_64hop_32khz
}

run_bf() {
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --encoder SongMAE \
    --species bf \
    --spec_dir /media/george-vengrovski/disk2/specs/bf_64hop_32khz \
    --run_dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --annotation_json "$ROOT/files/bf_annotations.json" \
    --out_dir "$RESULTS_DIR/bf_data2vec_${CHECKPOINT_SUFFIX}_${RESULTS_SUFFIX}" \
    --recording_mode events \
    --songs_per_bird 30 \
    --pool_window "$POOL_WINDOW" \
    --pool_hop "$POOL_HOP" \
    --pool_mode mean \
    --songmae_input_normalization audio_params \
    --songmae_input_normalization_stats_dir /media/george-vengrovski/disk2/specs/bf_64hop_32khz
}

run_canary() {
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --encoder SongMAE \
    --species canary \
    --spec_dir /media/george-vengrovski/disk2/specs/canary_individual_identification_64hop_32khz \
    --run_dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --annotation_json "$ROOT/files/canary_annotations_for_individual_id.json" \
    --out_dir "$RESULTS_DIR/canary_data2vec_${CHECKPOINT_SUFFIX}_${RESULTS_SUFFIX}" \
    --recording_mode events \
    --songs_per_bird 30 \
    --pool_window "$POOL_WINDOW" \
    --pool_hop "$POOL_HOP" \
    --pool_mode mean \
    --songmae_input_normalization audio_params \
    --songmae_input_normalization_stats_dir /media/george-vengrovski/disk2/specs/canary_individual_identification_64hop_32khz
}

run_chiffchaff() {
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --encoder SongMAE \
    --species chiffchaff \
    --spec_dir /media/george-vengrovski/disk2/specs/chiffchaff_64hop_32khz \
    --run_dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --annotation_json "$ROOT/files/chiffchaff_annotations.json" \
    --out_dir "$RESULTS_DIR/chiffchaff_data2vec_${CHECKPOINT_SUFFIX}_${RESULTS_SUFFIX}" \
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
    --run_dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --annotation_json "$ROOT/files/little_owl_annotations.json" \
    --out_dir "$RESULTS_DIR/little_owl_data2vec_${CHECKPOINT_SUFFIX}_${RESULTS_SUFFIX}" \
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
    --run_dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --annotation_json "$ROOT/files/tree_pipit_annotations.json" \
    --out_dir "$RESULTS_DIR/tree_pipit_data2vec_${CHECKPOINT_SUFFIX}_${RESULTS_SUFFIX}" \
    --recording_mode events \
    --songs_per_bird 30 \
    --pool_window "$POOL_WINDOW" \
    --pool_hop "$POOL_HOP" \
    --pool_mode mean \
    --songmae_input_normalization audio_params \
    --songmae_input_normalization_stats_dir /media/george-vengrovski/disk2/specs/tree_pipit_64hop_32khz
}

run_orangutan() {
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --encoder SongMAE \
    --species orangutan \
    --spec_dir /media/george-vengrovski/disk2/specs/orangutan_64hop_32khz \
    --run_dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --annotation_json "$ROOT/files/orangutan_annotations.json" \
    --out_dir "$RESULTS_DIR/orangutan_data2vec_${CHECKPOINT_SUFFIX}_${RESULTS_SUFFIX}" \
    --recording_mode events \
    --songs_per_bird 20 \
    --pool_window "$POOL_WINDOW" \
    --pool_hop "$POOL_HOP" \
    --pool_mode mean \
    --songmae_input_normalization audio_params \
    --songmae_input_normalization_stats_dir /media/george-vengrovski/disk2/specs/orangutan_64hop_32khz
}

run_lapp_ovenbird() {
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --encoder SongMAE \
    --species lapp_ovenbird \
    --spec_dir /media/george-vengrovski/disk2/specs/ovenbird_lapp_sample_64hop_32khz \
    --run_dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --annotation_json "$ROOT/files/lapp_ovenbird.json" \
    --out_dir "$RESULTS_DIR/lapp_ovenbird_data2vec_${CHECKPOINT_SUFFIX}_${RESULTS_SUFFIX}" \
    --recording_mode events \
    --songs_per_bird 10 \
    --pool_window "$POOL_WINDOW" \
    --pool_hop "$POOL_HOP" \
    --pool_mode mean \
    --songmae_input_normalization audio_params \
    --songmae_input_normalization_stats_dir /media/george-vengrovski/disk2/specs/ovenbird_lapp_sample_64hop_32khz
}

run_european_starling() {
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --encoder SongMAE \
    --species european_starling \
    --spec_dir /media/george-vengrovski/disk2/specs/european_starling_64hop_32khz \
    --run_dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --annotation_json "$ROOT/files/european_starling_annotations_unprefixed.json" \
    --out_dir "$RESULTS_DIR/european_starling_data2vec_${CHECKPOINT_SUFFIX}_${RESULTS_SUFFIX}" \
    --recording_mode events \
    --songs_per_bird 30 \
    --pool_window "$POOL_WINDOW" \
    --pool_hop "$POOL_HOP" \
    --pool_mode mean \
    --songmae_input_normalization audio_params \
    --songmae_input_normalization_stats_dir /media/george-vengrovski/disk2/specs/european_starling_64hop_32khz
}

usage() {
  echo "Usage: $0 [all|european_starling|zf|bf|canary|chiffchaff|little_owl|tree_pipit|orangutan|lapp_ovenbird]"
}

mkdir -p "$RESULTS_DIR"

target="${1:-all}"
case "$target" in
  all)
    run_zf
    run_bf
    run_canary
    run_chiffchaff
    run_little_owl
    run_tree_pipit
    run_orangutan
    run_lapp_ovenbird
    run_european_starling
    ;;
  european_starling)
    run_european_starling
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
  chiffchaff)
    run_chiffchaff
    ;;
  little_owl)
    run_little_owl
    ;;
  tree_pipit)
    run_tree_pipit
    ;;
  orangutan)
    run_orangutan
    ;;
  lapp_ovenbird)
    run_lapp_ovenbird
    ;;
  *)
    usage
    exit 1
    ;;
esac
