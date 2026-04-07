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
IID_POOL_WINDOW="${IID_POOL_WINDOW:-30}"
IID_POOL_HOP="${IID_POOL_HOP:-5}"
IID_POOL_MODE="${IID_POOL_MODE:-mean}"
IID_MAX_BIRDS="${IID_MAX_BIRDS:-0}"
IID_SEED="${IID_SEED:-42}"
IID_SONGS_PER_BIRD_OVERRIDE="${IID_SONGS_PER_BIRD:-}"

IID_LINEAR_VAL_FRACTION="${IID_LINEAR_VAL_FRACTION:-0.2}"
IID_LINEAR_C="${IID_LINEAR_C:-1.0}"
IID_LINEAR_MAX_ITER="${IID_LINEAR_MAX_ITER:-2000}"
IID_LINEAR_NORMALIZATION_PRESET="${IID_LINEAR_NORMALIZATION_PRESET:-vanilla}"
IID_LINEAR_SONGMAE_EMBEDDING_VARIANT="${IID_LINEAR_SONGMAE_EMBEDDING_VARIANT:-before}"

IID_CLUSTER_EMBEDDING_VARIANT="${IID_CLUSTER_EMBEDDING_VARIANT:-before}"
IID_CLUSTER_MIN_CLUSTER_SIZE="${IID_CLUSTER_MIN_CLUSTER_SIZE:-100}"
IID_CLUSTER_MIN_CLUSTER_HITS="${IID_CLUSTER_MIN_CLUSTER_HITS:-1}"
IID_CLUSTER_OVERLAP_THRESHOLD="${IID_CLUSTER_OVERLAP_THRESHOLD:-0.3}"

IID_UMAP_SONGMAE_INPUT_NORMALIZATION="${IID_UMAP_SONGMAE_INPUT_NORMALIZATION:-audio_params}"

usage() {
  echo "Usage: $0 <species_default|data2vec_40k> [all|zf|bf|canary ...]"
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
      IID_SONGS_PER_BIRD="${IID_SONGS_PER_BIRD_OVERRIDE:-30}"
      ;;
    bf)
      IID_BIRD_KEY="bf"
      IID_SPECIES="bf"
      IID_SPEC_DIR="/media/george-vengrovski/disk2/specs/bf_64hop_32khz"
      IID_ANNOTATION_JSON="$ROOT/files/bf_annotations.json"
      IID_SONGS_PER_BIRD="${IID_SONGS_PER_BIRD_OVERRIDE:-30}"
      ;;
    canary)
      IID_BIRD_KEY="canary"
      IID_SPECIES="canary"
      IID_SPEC_DIR="/media/george-vengrovski/disk2/specs/canary_individual_identification_64hop_32khz"
      IID_ANNOTATION_JSON="$ROOT/files/canary_annotations_for_individual_id.json"
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
    *)
      echo "Unknown model preset: $model_preset" >&2
      usage
      exit 1
      ;;
  esac
}

set_output_dirs() {
  local linear_encoder_tag

  IID_UMAP_OUT_DIR="$ROOT/results/individual_id_umap/${IID_BIRD_KEY}_${IID_MODEL_TAG}_w${IID_POOL_WINDOW}_h${IID_POOL_HOP}"
  linear_encoder_tag="$(printf '%s' "$IID_ENCODER" | tr '[:upper:]' '[:lower:]')"
  if [[ "$IID_ENCODER" == "SongMAE" ]]; then
    linear_encoder_tag="${linear_encoder_tag}_${IID_LINEAR_SONGMAE_EMBEDDING_VARIANT}"
  fi
  IID_LINEAR_OUT_DIR="$ROOT/results/individual_id_linear_probe/${IID_BIRD_KEY}_${linear_encoder_tag}_${IID_LINEAR_NORMALIZATION_PRESET}_${IID_MODEL_TAG}_w${IID_POOL_WINDOW}_h${IID_POOL_HOP}"
  IID_CLUSTER_OUT_DIR="$ROOT/results/individual_id_cluster/${IID_BIRD_KEY}_${IID_MODEL_TAG}_cluster_mcs${IID_CLUSTER_MIN_CLUSTER_SIZE}_w${IID_POOL_WINDOW}_h${IID_POOL_HOP}"
  IID_UMAP_SONGMAE_INPUT_NORMALIZATION_STATS_DIR="${IID_UMAP_SONGMAE_INPUT_NORMALIZATION_STATS_DIR_OVERRIDE:-$IID_SPEC_DIR}"
}

run_suite() {
  local bird="$1"

  set_bird_config "$bird"
  set_model_config "$MODEL_PRESET"
  set_output_dirs

  export IID_BIRD_KEY IID_SPECIES IID_SPEC_DIR IID_ANNOTATION_JSON IID_RUN_DIR IID_CHECKPOINT IID_MODEL_TAG
  export IID_ENCODER IID_RECORDING_MODE IID_SONGS_PER_BIRD IID_MAX_BIRDS IID_SEED IID_POOL_WINDOW IID_POOL_HOP IID_POOL_MODE
  export IID_UMAP_OUT_DIR IID_UMAP_SONGMAE_INPUT_NORMALIZATION IID_UMAP_SONGMAE_INPUT_NORMALIZATION_STATS_DIR
  export IID_LINEAR_OUT_DIR IID_LINEAR_VAL_FRACTION IID_LINEAR_C IID_LINEAR_MAX_ITER IID_LINEAR_NORMALIZATION_PRESET IID_LINEAR_SONGMAE_EMBEDDING_VARIANT
  export IID_CLUSTER_OUT_DIR IID_CLUSTER_EMBEDDING_VARIANT IID_CLUSTER_MIN_CLUSTER_SIZE IID_CLUSTER_MIN_CLUSTER_HITS IID_CLUSTER_OVERLAP_THRESHOLD

  echo "[individual_id] bird=$IID_BIRD_KEY model=$MODEL_PRESET run_dir=$IID_RUN_DIR"
  if [[ "$RUN_UMAP" == "1" ]]; then
    bash "$UMAP_SCRIPT"
  fi
  if [[ "$RUN_LINEAR_PROBE" == "1" ]]; then
    bash "$LINEAR_SCRIPT"
  fi
  if [[ "$RUN_CLUSTER" == "1" ]]; then
    bash "$CLUSTER_SCRIPT"
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
      birds+=(zf bf canary)
      ;;
    zf|bf|canary)
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
