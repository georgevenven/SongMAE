#!/usr/bin/env bash

set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${RUN_DIR:-$ROOT/runs/xcl_base_100k_p32x1_c010}"
CHECKPOINT="${CHECKPOINT:-model_step_499999.pth}"
OUT_ROOT="${OUT_ROOT:-results/individual_id_umap/stats_affinity_grid_pca512_maxpts25000}"
MAX_POINTS="${MAX_POINTS:-25000}"
PCA_DIM="${PCA_DIM:-512}"
UMAP_NEIGHBORS="${UMAP_NEIGHBORS:-50}"
UMAP_MIN_DIST="${UMAP_MIN_DIST:-0.1}"
UMAP_METRIC="${UMAP_METRIC:-cosine}"
FORCE="${FORCE:-0}"
NO_COLLAGE="${NO_COLLAGE:-0}"

mkdir -p "$OUT_ROOT/logs"
STATUS_FILE="$OUT_ROOT/status.tsv"
printf "species\twindow\trecording_feature\tstatus\texit_code\tout_dir\tlog\n" > "$STATUS_FILE"

species_list=(
  zf
  bf
  canary
  chiffchaff
  european_starling
  little_owl
  ovenbird
  tree_pipit
)

if [[ -n "${SPECIES_LIST:-}" ]]; then
  read -r -a species_list <<< "$SPECIES_LIST"
fi

declare -A annotation_json=(
  [zf]="files/annotation jsons/zf_annotations.json"
  [bf]="files/annotation jsons/bf_annotations.json"
  [canary]="files/annotation jsons/canary_annotations_for_individual_id.json"
  [chiffchaff]="files/annotation jsons/chiffchaff_annotations.json"
  [european_starling]="files/annotation jsons/european_starling_annotations.json"
  [little_owl]="files/annotation jsons/little_owl_annotations.json"
  [ovenbird]="files/annotation jsons/lapp_ovenbird.json"
  [tree_pipit]="files/annotation jsons/tree_pipit_annotations.json"
)

declare -A spec_dir=(
  [zf]="/media/george-vengrovski/disk2/specs/zf_64hop_32khz"
  [bf]="/media/george-vengrovski/disk2/specs/bf_64hop_32khz"
  [canary]="/media/george-vengrovski/disk2/specs/canary_individual_identification_64hop_32khz"
  [chiffchaff]="/media/george-vengrovski/disk2/specs/chiffchaff_64hop_32khz"
  [european_starling]="/media/george-vengrovski/disk2/specs/european_starling_64hop_32khz_prefixed"
  [little_owl]="/media/george-vengrovski/disk2/specs/little_owl_64hop_32khz"
  [ovenbird]="/media/george-vengrovski/disk2/specs/ovenbird_lapp_sample_64hop_32khz"
  [tree_pipit]="/media/george-vengrovski/disk2/specs/tree_pipit_64hop_32khz"
)

declare -A knn_npz=(
  [zf]="results/individual_id_knn_graph_metrics/bird_knn_matrix_laplacian_cap30/zf/knn_attribution_matrices.npz"
  [bf]="results/individual_id_knn_graph_metrics/bird_knn_matrix_laplacian_cap30/bf/knn_attribution_matrices.npz"
  [canary]="results/individual_id_knn_graph_metrics/bird_knn_matrix_laplacian_cap30/canary/knn_attribution_matrices.npz"
  [chiffchaff]="results/individual_id_knn_graph_metrics/bird_knn_matrix_laplacian_cap30/chiffchaff/knn_attribution_matrices.npz"
  [european_starling]="results/individual_id_knn_graph_metrics/bird_knn_matrix_laplacian_cap30/european_starling/knn_attribution_matrices.npz"
  [little_owl]="results/individual_id_knn_graph_metrics/bird_knn_matrix_laplacian_cap30/little_owl/knn_attribution_matrices.npz"
  [ovenbird]="results/individual_id_knn_graph_metrics/bird_knn_matrix_laplacian_cap30/ovenbird/knn_attribution_matrices.npz"
  [tree_pipit]="results/individual_id_knn_graph_metrics/bird_knn_matrix_laplacian_cap30/tree_pipit/knn_attribution_matrices.npz"
)

if [[ -n "${KNN_NPZ_ROOT:-}" ]]; then
  for species in "${!knn_npz[@]}"; do
    candidate="$KNN_NPZ_ROOT/$species/knn_attribution_matrices.npz"
    if [[ -f "$candidate" ]]; then
      knn_npz[$species]="$candidate"
    fi
  done
fi

declare -A songs_per_bird=(
  [zf]=30
  [bf]=30
  [canary]=30
  [chiffchaff]=30
  [european_starling]=30
  [little_owl]=30
  [ovenbird]=0
  [tree_pipit]=30
)

windows=(
  "w10_h2 10 2"
  "w30_h5 30 5"
  "w70_h10 70 10"
)

recording_features=(
  "recsvd15 svd"
  "recaffrow affinity_row"
)

run_one() {
  local species="$1"
  local window_tag="$2"
  local pool_window="$3"
  local pool_hop="$4"
  local feature_tag="$5"
  local feature_mode="$6"

  local out_dir="$OUT_ROOT/${species}_${window_tag}_stats_pca${PCA_DIM}_${feature_tag}_nn${UMAP_NEIGHBORS}_${UMAP_METRIC}"
  local log="$OUT_ROOT/logs/${species}_${window_tag}_${feature_tag}.log"

  if [[ "$FORCE" != "1" && -f "$out_dir/summary.json" ]]; then
    printf "%s\t%s\t%s\tskipped\t0\t%s\t%s\n" "$species" "$window_tag" "$feature_tag" "$out_dir" "$log" | tee -a "$STATUS_FILE"
    return 0
  fi

  mkdir -p "$out_dir"
  echo "[$(date -Is)] running $species $window_tag $feature_tag" | tee "$log"

  MPLBACKEND=Agg python individual_id/run_individual_id_umap.py \
    --encoder SongMAE \
    --species "$species" \
    --annotation_json "${annotation_json[$species]}" \
    --spec_dir "${spec_dir[$species]}" \
    --out_dir "$out_dir" \
    --run_dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --recording_mode events \
    --songs_per_bird "${SONGS_PER_BIRD_OVERRIDE:-${songs_per_bird[$species]}}" \
    --seed 42 \
    --pool_window "$pool_window" \
    --pool_hop "$pool_hop" \
    --pool_mode stats \
    --pool_layout sliding \
    --max_points "$MAX_POINTS" \
    --feature_postprocess pca_whiten_l2 \
    --feature_postprocess_dim "$PCA_DIM" \
    --recording_svd_npz "${knn_npz[$species]}" \
    --recording_feature_mode "$feature_mode" \
    --recording_svd_dim 15 \
    --recording_svd_alpha 1.0 \
    --recording_svd_append post \
    --audio_params_stats_dir "${spec_dir[$species]}" \
    --umap_neighbors "$UMAP_NEIGHBORS" \
    --umap_min_dist "$UMAP_MIN_DIST" \
    --umap_metric "$UMAP_METRIC" \
    >> "$log" 2>&1

  local code=$?
  local status="ok"
  if [[ "$code" -ne 0 ]]; then
    status="failed"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$species" "$window_tag" "$feature_tag" "$status" "$code" "$out_dir" "$log" | tee -a "$STATUS_FILE"
  return 0
}

for species in "${species_list[@]}"; do
  for window_spec in "${windows[@]}"; do
    read -r window_tag pool_window pool_hop <<< "$window_spec"
    for feature_spec in "${recording_features[@]}"; do
      read -r feature_tag feature_mode <<< "$feature_spec"
      run_one "$species" "$window_tag" "$pool_window" "$pool_hop" "$feature_tag" "$feature_mode"
    done
  done
done

if [[ "$NO_COLLAGE" != "1" ]]; then
  python individual_id/plotting/make_stats_affinity_grid_collages.py --out_root "$OUT_ROOT"
fi
