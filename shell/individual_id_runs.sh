#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/george-vengrovski/Documents/projects/TinyBird"
UMAP_SCRIPT="$ROOT/shell/individual_id_umap_runs.sh"
LINEAR_SCRIPT="$ROOT/shell/individual_id_linear_probe_runs.sh"
CLUSTER_SCRIPT="$ROOT/shell/individual_id_cluster_runs.sh"

RUN_UMAP="${RUN_UMAP:-1}"
RUN_LINEAR_PROBE="${RUN_LINEAR_PROBE:-1}"
RUN_CLUSTER="${RUN_CLUSTER:-1}"

IID_ENCODER="${IID_ENCODER:-SongMAE}"
IID_RECORDING_MODE="${IID_RECORDING_MODE:-events}"
IID_POOL_WINDOW="${IID_POOL_WINDOW:-50}"
IID_POOL_HOP="${IID_POOL_HOP:-10}"
IID_POOL_MODE="${IID_POOL_MODE:-mean}"
IID_POOL_LAYOUT="${IID_POOL_LAYOUT:-sliding}"
IID_UMAP_MAX_POINTS="${IID_UMAP_MAX_POINTS:-0}"
IID_MAX_BIRDS="${IID_MAX_BIRDS:-0}"
IID_SEED="${IID_SEED:-42}"
IID_SONGS_PER_BIRD_OVERRIDE="${IID_SONGS_PER_BIRD:-}"

IID_LINEAR_VAL_FRACTION="${IID_LINEAR_VAL_FRACTION:-0.2}"
IID_LINEAR_C="${IID_LINEAR_C:-1.0}"
IID_LINEAR_MAX_ITER="${IID_LINEAR_MAX_ITER:-2000}"
IID_LINEAR_FEATURE_POSTPROCESS="${IID_LINEAR_FEATURE_POSTPROCESS:-none}"
IID_LINEAR_FEATURE_POSTPROCESS_DIM="${IID_LINEAR_FEATURE_POSTPROCESS_DIM:-256}"
IID_LINEAR_POOL_HOP="${IID_LINEAR_POOL_HOP:-$IID_POOL_WINDOW}"
IID_UMAP_FEATURE_POSTPROCESS="${IID_UMAP_FEATURE_POSTPROCESS:-whiten_l2}"
IID_UMAP_FEATURE_POSTPROCESS_DIM="${IID_UMAP_FEATURE_POSTPROCESS_DIM:-256}"
IID_UMAP_FEATURE_POSTPROCESS_LOAD="${IID_UMAP_FEATURE_POSTPROCESS_LOAD:-}"
IID_UMAP_FEATURE_POSTPROCESS_SAVE="${IID_UMAP_FEATURE_POSTPROCESS_SAVE:-}"
IID_UMAP_NORMALIZATION_PRESET="${IID_UMAP_NORMALIZATION_PRESET:-zscore_rescaled}"
IID_UMAP_AUDIO_PARAMS_STATS_DIR="${IID_UMAP_AUDIO_PARAMS_STATS_DIR:-}"
IID_LINEAR_NORMALIZATION_PRESET="${IID_LINEAR_NORMALIZATION_PRESET:-zscore_rescaled}"
IID_LINEAR_AUDIO_PARAMS_STATS_DIR="${IID_LINEAR_AUDIO_PARAMS_STATS_DIR:-}"
IID_LINEAR_SONGMAE_EMBEDDING_VARIANT="${IID_LINEAR_SONGMAE_EMBEDDING_VARIANT:-before}"
IID_LINEAR_WINDOW_MEAN_POOL="${IID_LINEAR_WINDOW_MEAN_POOL:-0}"
IID_LINEAR_WINDOW_CONCAT="${IID_LINEAR_WINDOW_CONCAT:-0}"
IID_LINEAR_WINDOW_TOKEN_PROBE="${IID_LINEAR_WINDOW_TOKEN_PROBE:-0}"
IID_LINEAR_TRAIN_AUDIO_SPEED_MIN_PCT="${IID_LINEAR_TRAIN_AUDIO_SPEED_MIN_PCT:-0.0}"
IID_LINEAR_TRAIN_AUDIO_SPEED_MAX_PCT="${IID_LINEAR_TRAIN_AUDIO_SPEED_MAX_PCT:-0.5}"
IID_AVES_MODEL_PATH="${IID_AVES_MODEL_PATH:-$ROOT/files/aves-base-bio.torchaudio.pt}"
IID_AVES_CONFIG_PATH="${IID_AVES_CONFIG_PATH:-$ROOT/files/aves-base-bio.torchaudio.model_config.json}"
IID_PERCH_MODEL_NAME="${IID_PERCH_MODEL_NAME:-perch_v2}"
IID_PERCH_AUDIO_SR="${IID_PERCH_AUDIO_SR:-32000}"
IID_PERCH_WINDOW_SECONDS="${IID_PERCH_WINDOW_SECONDS:-5.0}"
IID_WAV_ROOT="${IID_WAV_ROOT:-${IID_AVES_WAV_ROOT:-}}"
IID_WAV_MANIFEST="${IID_WAV_MANIFEST:-${IID_AVES_WAV_MANIFEST:-}}"
IID_WAV_EXTS="${IID_WAV_EXTS:-${IID_AVES_WAV_EXTS:-.wav,.flac,.ogg,.mp3}}"
IID_AVES_AUDIO_SR="${IID_AVES_AUDIO_SR:-16000}"
IID_AUDIO_CONTEXT_SECONDS="${IID_AUDIO_CONTEXT_SECONDS:-2.0}"

IID_CLUSTER_EMBEDDING_VARIANT="${IID_CLUSTER_EMBEDDING_VARIANT:-before}"
IID_CLUSTER_MIN_CLUSTER_SIZE="${IID_CLUSTER_MIN_CLUSTER_SIZE:-100}"
IID_CLUSTER_MIN_CLUSTER_HITS="${IID_CLUSTER_MIN_CLUSTER_HITS:-1}"
IID_CLUSTER_OVERLAP_THRESHOLD="${IID_CLUSTER_OVERLAP_THRESHOLD:-0.3}"

usage() {
  echo "Usage: $0 <species_default|data2vec_40k|birdaves_base|perch_2_0> [all|zf|bf|canary|chiffchaff|european_starling|tree_pipit|little_owl|orangutan|ovenbird ...]"
}

latest_checkpoint() {
  local run_dir="$1"
  find "$run_dir/weights" -maxdepth 1 -type f -name 'model_step_*.pth' | sort | tail -n 1 | xargs -r basename
}

checkpoint_suffix() {
  local checkpoint="$1"
  if [[ "$checkpoint" =~ ^model_step_0*([0-9]+)\.pth$ ]]; then
    printf 'step%s' "${BASH_REMATCH[1]}"
    return
  fi
  printf '%s' "${checkpoint%.pth}"
}

set_bird_config() {
  local bird="$1"
  case "$bird" in
    zf)
      IID_BIRD_KEY="zf"
      IID_SPECIES="Zebra_Finch"
      IID_SPEC_DIR="/media/george-vengrovski/disk2/specs/zf_64hop_32khz"
      IID_ANNOTATION_JSON="$ROOT/files/zf_annotations.json"
      IID_WAV_ROOT="/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae"
      IID_SONGS_PER_BIRD="${IID_SONGS_PER_BIRD_OVERRIDE:-30}"
      ;;
    bf)
      IID_BIRD_KEY="bf"
      IID_SPECIES="bf"
      IID_SPEC_DIR="/media/george-vengrovski/disk2/specs/bf_64hop_32khz"
      IID_ANNOTATION_JSON="$ROOT/files/bf_annotations.json"
      IID_WAV_ROOT="/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae"
      IID_SONGS_PER_BIRD="${IID_SONGS_PER_BIRD_OVERRIDE:-30}"
      ;;
    canary)
      IID_BIRD_KEY="canary"
      IID_SPECIES="canary"
      IID_SPEC_DIR="/media/george-vengrovski/disk2/specs/canary_individual_identification_64hop_32khz"
      IID_ANNOTATION_JSON="$ROOT/files/canary_annotations_for_individual_id.json"
      IID_WAV_ROOT="/media/george-vengrovski/disk2/raw_data/canary/individual_identification"
      IID_SONGS_PER_BIRD="${IID_SONGS_PER_BIRD_OVERRIDE:-30}"
      ;;
    chiffchaff)
      IID_BIRD_KEY="chiffchaff"
      IID_SPECIES="chiffchaff"
      IID_SPEC_DIR="/media/george-vengrovski/disk2/specs/chiffchaff_64hop_32khz"
      IID_ANNOTATION_JSON="$ROOT/files/chiffchaff_annotations.json"
      IID_WAV_ROOT="/media/george-vengrovski/disk2/raw_data/chiffchaff"
      IID_SONGS_PER_BIRD="${IID_SONGS_PER_BIRD_OVERRIDE:-30}"
      ;;
    european_starling|starling)
      IID_BIRD_KEY="european_starling"
      IID_SPECIES="european_starling"
      IID_SPEC_DIR="/media/george-vengrovski/disk2/specs/european_starling_64hop_32khz"
      IID_ANNOTATION_JSON="$ROOT/files/european_starling_annotations_unprefixed.json"
      IID_WAV_ROOT="/media/george-vengrovski/disk2/raw_data/european_starling"
      IID_SONGS_PER_BIRD="${IID_SONGS_PER_BIRD_OVERRIDE:-30}"
      ;;
    tree_pipit)
      IID_BIRD_KEY="tree_pipit"
      IID_SPECIES="tree_pipit"
      IID_SPEC_DIR="/media/george-vengrovski/disk2/specs/tree_pipit_64hop_32khz"
      IID_ANNOTATION_JSON="$ROOT/files/tree_pipit_annotations.json"
      IID_WAV_ROOT="/media/george-vengrovski/disk2/raw_data/tree_pipit"
      IID_SONGS_PER_BIRD="${IID_SONGS_PER_BIRD_OVERRIDE:-30}"
      ;;
    little_owl)
      IID_BIRD_KEY="little_owl"
      IID_SPECIES="little_owl"
      IID_SPEC_DIR="/media/george-vengrovski/disk2/specs/little_owl_64hop_32khz"
      IID_ANNOTATION_JSON="$ROOT/files/little_owl_annotations.json"
      IID_WAV_ROOT="/media/george-vengrovski/disk2/raw_data/little_owl"
      IID_SONGS_PER_BIRD="${IID_SONGS_PER_BIRD_OVERRIDE:-30}"
      ;;
    orangutan)
      IID_BIRD_KEY="orangutan"
      IID_SPECIES="orangutan"
      IID_SPEC_DIR="/media/george-vengrovski/disk2/specs/orangutan_64hop_32khz"
      IID_ANNOTATION_JSON="$ROOT/files/orangutan_annotations.json"
      IID_WAV_ROOT="/media/george-vengrovski/disk2/raw_data/orangutan"
      IID_SONGS_PER_BIRD="${IID_SONGS_PER_BIRD_OVERRIDE:-30}"
      ;;
    ovenbird|lapp_ovenbird)
      IID_BIRD_KEY="ovenbird"
      IID_SPECIES="ovenbird"
      IID_SPEC_DIR="/media/george-vengrovski/disk2/specs/ovenbird_lapp_sample_64hop_32khz"
      IID_ANNOTATION_JSON="$ROOT/files/lapp_ovenbird.json"
      IID_WAV_ROOT="/media/george-vengrovski/disk2/raw_data/ovenbird_lapp_sample"
      IID_SONGS_PER_BIRD="${IID_SONGS_PER_BIRD_OVERRIDE:-30}"
      ;;
    *)
      echo "Unknown bird target: $bird" >&2
      exit 1
      ;;
  esac

  if [[ -n "${IID_SPEC_DIR_OVERRIDE:-}" ]]; then
    IID_SPEC_DIR="$IID_SPEC_DIR_OVERRIDE"
  fi
  if [[ -n "${IID_ANNOTATION_JSON_OVERRIDE:-}" ]]; then
    IID_ANNOTATION_JSON="$IID_ANNOTATION_JSON_OVERRIDE"
  fi
  if [[ -n "${IID_WAV_ROOT_OVERRIDE:-}" ]]; then
    IID_WAV_ROOT="$IID_WAV_ROOT_OVERRIDE"
  fi
}

set_model_config() {
  local model_preset="$1"

  IID_CHECKPOINT="${IID_CHECKPOINT_OVERRIDE:-}"
  case "$model_preset" in
    species_default)
      IID_MODEL_TAG="${IID_MODEL_TAG_OVERRIDE:-species_continue10k_32h10w}"
      if [[ -n "${IID_RUN_DIR_OVERRIDE:-}" ]]; then
        IID_RUN_DIR="$IID_RUN_DIR_OVERRIDE"
        return
      fi
      case "$IID_BIRD_KEY" in
        zf)
          IID_RUN_DIR="$ROOT/runs/xcm_voronoi_mask_no_normalize_32h_10w_zf_continue10k_bs24_20260302_133416"
          ;;
        bf)
          IID_RUN_DIR="$ROOT/runs/xcm_voronoi_mask_no_normalize_32h_10w_bf_continue10k_bs24_20260311_131742"
          ;;
        canary)
          IID_RUN_DIR="$ROOT/runs/xcm_voronoi_mask_no_normalize_32h_10w_canary_continue10k_bs24_20260312_141845"
          ;;
        *)
          echo "No default species run configured for '$IID_BIRD_KEY'. Set IID_RUN_DIR_OVERRIDE." >&2
          exit 1
          ;;
      esac
      ;;
    data2vec_40k)
      IID_RUN_DIR="${IID_RUN_DIR_OVERRIDE:-$ROOT/runs/merged_data2vec_from_xcm_40k}"
      if [[ -z "$IID_CHECKPOINT" ]]; then
        IID_CHECKPOINT="$(latest_checkpoint "$IID_RUN_DIR")"
      fi
      if [[ -z "$IID_CHECKPOINT" ]]; then
        echo "No model_step_*.pth checkpoint found in $IID_RUN_DIR/weights" >&2
        exit 1
      fi
      IID_MODEL_TAG="${IID_MODEL_TAG_OVERRIDE:-data2vec_$(checkpoint_suffix "$IID_CHECKPOINT")}"
      ;;
    birdaves_base|aves_base|birdaves)
      IID_RUN_DIR="${IID_RUN_DIR_OVERRIDE:-$ROOT}"
      IID_MODEL_TAG="${IID_MODEL_TAG_OVERRIDE:-birdaves_base}"
      ;;
    perch_2_0|perch_v2|perch)
      IID_RUN_DIR="${IID_RUN_DIR_OVERRIDE:-$ROOT}"
      IID_MODEL_TAG="${IID_MODEL_TAG_OVERRIDE:-perch_v2}"
      ;;
    *)
      echo "Unknown model preset: $model_preset" >&2
      usage
      exit 1
      ;;
  esac
}

set_output_dirs() {
  local linear_encoder_tag linear_aug_tag linear_norm_tag

  local pool_layout_tag=""
  if [[ "$IID_POOL_LAYOUT" != "sliding" ]]; then
    pool_layout_tag="_${IID_POOL_LAYOUT}"
  fi
  local max_points_tag=""
  if [[ "$IID_UMAP_MAX_POINTS" != "0" ]]; then
    max_points_tag="_maxpts${IID_UMAP_MAX_POINTS}"
  fi

  local umap_postprocess_tag=""
  if [[ "$IID_UMAP_FEATURE_POSTPROCESS" != "none" ]]; then
    umap_postprocess_tag="_whitened_${IID_UMAP_FEATURE_POSTPROCESS}${IID_UMAP_FEATURE_POSTPROCESS_DIM}"
  fi
  IID_UMAP_OUT_DIR="$ROOT/results/individual_id_umap/${IID_BIRD_KEY}_${IID_MODEL_TAG}${umap_postprocess_tag}${pool_layout_tag}${max_points_tag}_w${IID_POOL_WINDOW}_h${IID_POOL_HOP}"
  linear_encoder_tag="$(printf '%s' "$IID_ENCODER" | tr '[:upper:]' '[:lower:]')"
  if [[ "$IID_ENCODER" == "SongMAE" ]]; then
    linear_encoder_tag="${linear_encoder_tag}_${IID_LINEAR_SONGMAE_EMBEDDING_VARIANT}"
  fi
  linear_norm_tag=""
  if [[ "$IID_ENCODER" == "Spec" ]]; then
    linear_norm_tag="_${IID_LINEAR_NORMALIZATION_PRESET}"
  fi
  local linear_postprocess_tag=""
  if [[ "$IID_LINEAR_FEATURE_POSTPROCESS" != "none" ]]; then
    linear_postprocess_tag="_${IID_LINEAR_FEATURE_POSTPROCESS}${IID_LINEAR_FEATURE_POSTPROCESS_DIM}"
  fi
  linear_aug_tag="clean"
  if [[ "$IID_LINEAR_WINDOW_MEAN_POOL" == "1" ]]; then
    linear_aug_tag="${linear_aug_tag}_windowmean"
  fi
  if [[ "$IID_LINEAR_WINDOW_CONCAT" == "1" ]]; then
    linear_aug_tag="${linear_aug_tag}_windowconcat"
  fi
  if [[ "$IID_LINEAR_WINDOW_TOKEN_PROBE" == "1" ]]; then
    linear_aug_tag="${linear_aug_tag}_windowtoken"
  fi
  if [[ "$IID_LINEAR_TRAIN_AUDIO_SPEED_MAX_PCT" != "0.0" ]]; then
    linear_aug_tag="speed$(printf '%s' "$IID_LINEAR_TRAIN_AUDIO_SPEED_MAX_PCT" | tr '.' 'p')_trainonly"
    if [[ "$IID_LINEAR_WINDOW_MEAN_POOL" == "1" ]]; then
      linear_aug_tag="${linear_aug_tag}_windowmean"
    fi
    if [[ "$IID_LINEAR_WINDOW_CONCAT" == "1" ]]; then
      linear_aug_tag="${linear_aug_tag}_windowconcat"
    fi
    if [[ "$IID_LINEAR_WINDOW_TOKEN_PROBE" == "1" ]]; then
      linear_aug_tag="${linear_aug_tag}_windowtoken"
    fi
  fi
  IID_LINEAR_OUT_DIR="$ROOT/results/individual_id_linear_probe/${IID_BIRD_KEY}_${linear_encoder_tag}${linear_norm_tag}_${IID_MODEL_TAG}_${linear_aug_tag}${linear_postprocess_tag}_w${IID_POOL_WINDOW}_h${IID_LINEAR_POOL_HOP}"
  IID_CLUSTER_OUT_DIR="$ROOT/results/individual_id_cluster/${IID_BIRD_KEY}_${IID_MODEL_TAG}_cluster_mcs${IID_CLUSTER_MIN_CLUSTER_SIZE}_w${IID_POOL_WINDOW}_h${IID_POOL_HOP}"
  IID_UMAP_AUDIO_PARAMS_STATS_DIR="${IID_UMAP_AUDIO_PARAMS_STATS_DIR_OVERRIDE:-$IID_SPEC_DIR}"
  IID_LINEAR_AUDIO_PARAMS_STATS_DIR="${IID_LINEAR_AUDIO_PARAMS_STATS_DIR_OVERRIDE:-$IID_SPEC_DIR}"
}

run_suite() {
  local bird="$1"

  set_bird_config "$bird"
  set_model_config "$MODEL_PRESET"
  set_output_dirs
  IID_AVES_WAV_ROOT="$IID_WAV_ROOT"
  IID_AVES_WAV_MANIFEST="$IID_WAV_MANIFEST"
  IID_AVES_WAV_EXTS="$IID_WAV_EXTS"

  export IID_BIRD_KEY IID_SPECIES IID_SPEC_DIR IID_ANNOTATION_JSON IID_RUN_DIR IID_CHECKPOINT IID_MODEL_TAG
  export IID_ENCODER IID_RECORDING_MODE IID_SONGS_PER_BIRD IID_MAX_BIRDS IID_SEED IID_POOL_WINDOW IID_POOL_HOP IID_POOL_MODE IID_POOL_LAYOUT IID_UMAP_MAX_POINTS
  export IID_UMAP_OUT_DIR IID_UMAP_FEATURE_POSTPROCESS IID_UMAP_FEATURE_POSTPROCESS_DIM IID_UMAP_FEATURE_POSTPROCESS_LOAD IID_UMAP_FEATURE_POSTPROCESS_SAVE IID_UMAP_NORMALIZATION_PRESET IID_UMAP_AUDIO_PARAMS_STATS_DIR
  export IID_LINEAR_OUT_DIR IID_LINEAR_VAL_FRACTION IID_LINEAR_C IID_LINEAR_MAX_ITER IID_LINEAR_FEATURE_POSTPROCESS IID_LINEAR_FEATURE_POSTPROCESS_DIM IID_LINEAR_POOL_HOP IID_LINEAR_NORMALIZATION_PRESET IID_LINEAR_SONGMAE_EMBEDDING_VARIANT IID_LINEAR_WINDOW_MEAN_POOL IID_LINEAR_WINDOW_CONCAT IID_LINEAR_WINDOW_TOKEN_PROBE
  export IID_LINEAR_AUDIO_PARAMS_STATS_DIR
  export IID_LINEAR_TRAIN_AUDIO_SPEED_MIN_PCT IID_LINEAR_TRAIN_AUDIO_SPEED_MAX_PCT
  export IID_AVES_MODEL_PATH IID_AVES_CONFIG_PATH IID_PERCH_MODEL_NAME IID_PERCH_AUDIO_SR IID_PERCH_WINDOW_SECONDS IID_WAV_ROOT IID_WAV_MANIFEST IID_WAV_EXTS IID_AVES_AUDIO_SR IID_AUDIO_CONTEXT_SECONDS
  export IID_CLUSTER_OUT_DIR IID_CLUSTER_EMBEDDING_VARIANT IID_CLUSTER_MIN_CLUSTER_SIZE IID_CLUSTER_MIN_CLUSTER_HITS IID_CLUSTER_OVERLAP_THRESHOLD

  echo "[individual_id] bird=$IID_BIRD_KEY model=$MODEL_PRESET run_dir=$IID_RUN_DIR"
  if [[ "$RUN_UMAP" == "1" ]]; then
    if [[ "$IID_ENCODER" == "Perch" ]]; then
      echo "[individual_id] skipping umap for encoder=Perch"
    else
      bash "$UMAP_SCRIPT"
    fi
  fi
  if [[ "$RUN_LINEAR_PROBE" == "1" ]]; then
    bash "$LINEAR_SCRIPT"
  fi
  if [[ "$RUN_CLUSTER" == "1" ]]; then
    if [[ "$IID_ENCODER" == "AVES" ]] || [[ "$IID_ENCODER" == "Perch" ]]; then
      echo "[individual_id] skipping cluster for encoder=$IID_ENCODER"
    else
      bash "$CLUSTER_SCRIPT"
    fi
  fi
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_PRESET="$1"
shift

targets=("$@")
if [[ ${#targets[@]} -eq 0 ]]; then
  targets=(all)
fi

birds=()
for target in "${targets[@]}"; do
  case "$target" in
    all)
      birds+=(zf bf canary chiffchaff european_starling tree_pipit little_owl orangutan ovenbird)
      ;;
    zf|bf|canary|chiffchaff|european_starling|starling|tree_pipit|little_owl|orangutan|ovenbird|lapp_ovenbird)
      birds+=("$target")
      ;;
    *)
      echo "Unknown bird target: $target" >&2
      usage
      exit 1
      ;;
  esac
done

for bird in "${birds[@]}"; do
  run_suite "$bird"
done
