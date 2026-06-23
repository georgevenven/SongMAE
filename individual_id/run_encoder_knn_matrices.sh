#!/usr/bin/env bash

set -uo pipefail

ROOT="/home/george-vengrovski/Documents/projects/TinyBird"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="$ROOT/individual_id/knn_bird_matrix.py"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/individual_id_knn_graph_metrics/encoder_knn_matrices}"
STATUS_PATH="$OUT_ROOT/status.tsv"
SONGMAE_RUN_DIR="${SONGMAE_RUN_DIR:-$ROOT/github_assets/xcl_voronoi_mask_no_normalize_32h_10w_5s_fp8}"
SONGMAE_CHECKPOINT="${SONGMAE_CHECKPOINT:-model_step_499999.pth}"
AVES_MODEL_PATH="${AVES_MODEL_PATH:-$ROOT/files/aves-base-bio.torchaudio.pt}"
AVES_CONFIG_PATH="${AVES_CONFIG_PATH:-$ROOT/files/aves-base-bio.torchaudio.model_config.json}"
PERCH_ENV_BIN="${PERCH_ENV_BIN:-/home/george-vengrovski/anaconda3/envs/perch/bin/python}"
PERCH_CUDNN_DIR="${PERCH_CUDNN_DIR:-/home/george-vengrovski/anaconda3/envs/perch/lib/python3.11/site-packages/nvidia/cudnn/lib}"

MAX_POINTS_PER_RECORDING="${MAX_POINTS_PER_RECORDING:-400}"
MAX_TOTAL_POINTS="${MAX_TOTAL_POINTS:-50000}"
K_VALUES="${K_VALUES:-1,2,5,10,20,50,100}"
MATRIX_K="${MATRIX_K:-50}"
KNN_CHUNK_SIZE="${KNN_CHUNK_SIZE:-128}"
FEATURE_POSTPROCESS="${FEATURE_POSTPROCESS:-pca_whiten_l2}"
FEATURE_POSTPROCESS_DIM="${FEATURE_POSTPROCESS_DIM:-1024}"
AUDIO_CONTEXT_SECONDS="${AUDIO_CONTEXT_SECONDS:-2.0}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-1800}"
MAX_AS_GB="${MAX_AS_GB:-48}"
KNN_CPU="${KNN_CPU:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

mkdir -p "$OUT_ROOT/logs"
printf "encoder\tspecies\tstatus\texit_code\tout_dir\tlog\n" > "$STATUS_PATH"

species_list=(${SPECIES_LIST:-zf bf canary chiffchaff european_starling tree_pipit little_owl ovenbird})
encoder_list=(${ENCODER_LIST:-SongMAE AVES Perch HuBERT BirdMAE})

declare -A wav_root=(
  [zf]="/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae"
  [bf]="/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae"
  [canary]="/media/george-vengrovski/disk2/raw_data/canary/individual_identification"
  [chiffchaff]="/media/george-vengrovski/disk2/raw_data/chiffchaff"
  [european_starling]="/media/george-vengrovski/disk2/raw_data/european_starling"
  [tree_pipit]="/media/george-vengrovski/disk2/raw_data/tree_pipit"
  [little_owl]="/media/george-vengrovski/disk2/raw_data/little_owl"
  [ovenbird]="/media/george-vengrovski/disk2/raw_data/ovenbird_lapp_sample"
)

declare -A songs_per_bird=(
  [zf]=30
  [bf]=30
  [canary]=30
  [chiffchaff]=30
  [european_starling]=30
  [tree_pipit]=30
  [little_owl]=30
  [ovenbird]=30
)

declare -A pool_window=(
  [zf]=30
  [bf]=30
  [canary]=100
  [chiffchaff]=300
  [european_starling]=100
  [tree_pipit]=100
  [little_owl]=30
  [ovenbird]=30
)

declare -A pool_hop=(
  [zf]=5
  [bf]=5
  [canary]=10
  [chiffchaff]=50
  [european_starling]=10
  [tree_pipit]=10
  [little_owl]=5
  [ovenbird]=5
)

encoder_tag() {
  case "$1" in
    SongMAE) printf "%s" "${SONGMAE_TAG:-songmae_xcl_ckpt499999}" ;;
    AVES) printf "birdaves_base" ;;
    Perch) printf "perch_v2" ;;
    HuBERT) printf "hubert_large_ll60k" ;;
    BirdMAE) printf "bird_mae_base" ;;
    *) return 1 ;;
  esac
}

run_dir_for_encoder() {
  case "$1" in
    SongMAE) printf "%s" "$SONGMAE_RUN_DIR" ;;
    *) printf "%s" "$ROOT" ;;
  esac
}

checkpoint_for_encoder() {
  case "$1" in
    SongMAE) printf "%s" "$SONGMAE_CHECKPOINT" ;;
    *) printf "" ;;
  esac
}

python_for_encoder() {
  case "$1" in
    Perch) printf "%s" "$PERCH_ENV_BIN" ;;
    *) printf "%s" "$PYTHON_BIN" ;;
  esac
}

window_for_encoder() {
  local encoder="$1"
  local species="$2"
  case "$encoder" in
    BirdMAE) printf "%s" "${BIRDMAE_POOL_WINDOW:-4}" ;;
    Perch) printf "%s" "${PERCH_POOL_WINDOW:-1}" ;;
    *) printf "%s" "${pool_window[$species]}" ;;
  esac
}

hop_for_encoder() {
  local encoder="$1"
  local species="$2"
  case "$encoder" in
    BirdMAE) printf "%s" "${BIRDMAE_POOL_HOP:-1}" ;;
    Perch) printf "%s" "${PERCH_POOL_HOP:-1}" ;;
    *) printf "%s" "${pool_hop[$species]}" ;;
  esac
}

run_one() {
  local encoder="$1"
  local species="$2"
  local tag out_dir log_path py_bin run_dir checkpoint window hop

  tag="$(encoder_tag "$encoder")"
  out_dir="$OUT_ROOT/$tag/$species"
  log_path="$OUT_ROOT/logs/${tag}_${species}.log"
  py_bin="$(python_for_encoder "$encoder")"
  run_dir="$(run_dir_for_encoder "$encoder")"
  checkpoint="$(checkpoint_for_encoder "$encoder")"
  window="$(window_for_encoder "$encoder" "$species")"
  hop="$(hop_for_encoder "$encoder" "$species")"

  if [[ -f "$out_dir/summary.json" && "${FORCE:-0}" != "1" ]]; then
    printf "%s\t%s\tskipped\t0\t%s\t%s\n" "$encoder" "$species" "$out_dir" "$log_path" | tee -a "$STATUS_PATH"
    return 0
  fi

  echo "[$(date --iso-8601=seconds)] running $encoder $species"
  local cmd=(
    "$py_bin" "$SCRIPT_PATH" "$species"
    --encoder "$encoder"
    --run_dir "$run_dir"
    --checkpoint "$checkpoint"
    --out_dir "$OUT_ROOT/$tag"
    --songs_per_bird "${SONGS_PER_BIRD_OVERRIDE:-${songs_per_bird[$species]}}"
    --pool_window "$window"
    --pool_hop "$hop"
    --max_points_per_recording "$MAX_POINTS_PER_RECORDING"
    --max_total_points "$MAX_TOTAL_POINTS"
    --k_values "$K_VALUES"
    --matrix_k "$MATRIX_K"
    --knn_chunk_size "$KNN_CHUNK_SIZE"
    --feature_postprocess "$FEATURE_POSTPROCESS"
    --feature_postprocess_dim "$FEATURE_POSTPROCESS_DIM"
    --wav_root "${wav_root[$species]}"
    --audio_context_seconds "$AUDIO_CONTEXT_SECONDS"
  )
  if [[ "$KNN_CPU" == "1" ]]; then
    cmd+=(--cpu)
  fi

  if [[ "$encoder" == "SongMAE" ]]; then
    cmd+=(--songmae_affinity_features tokens)
  fi
  if [[ "$encoder" == "AVES" ]]; then
    cmd+=(--aves_model_path "$AVES_MODEL_PATH" --aves_config_path "$AVES_CONFIG_PATH")
  fi
  if [[ "$species" == "european_starling" && "$encoder" != "SongMAE" ]]; then
    cmd+=(--annotation_json_override "$ROOT/files/annotation jsons/european_starling_annotations.json")
  fi
  if [[ "$encoder" == "Perch" ]]; then
    (
      ulimit -v $((MAX_AS_GB * 1024 * 1024))
      LD_LIBRARY_PATH="$PERCH_CUDNN_DIR:${LD_LIBRARY_PATH:-}" timeout --kill-after=60s "$RUN_TIMEOUT_SECONDS" "${cmd[@]}"
    ) > "$log_path" 2>&1
  else
    (
      ulimit -v $((MAX_AS_GB * 1024 * 1024))
      timeout --kill-after=60s "$RUN_TIMEOUT_SECONDS" "${cmd[@]}"
    ) > "$log_path" 2>&1
  fi
  local code=$?
  local status="ok"
  if [[ "$code" != "0" ]]; then
    status="fail"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$encoder" "$species" "$status" "$code" "$out_dir" "$log_path" | tee -a "$STATUS_PATH"
  return 0
}

for encoder in "${encoder_list[@]}"; do
  for species in "${species_list[@]}"; do
    run_one "$encoder" "$species"
  done
done
