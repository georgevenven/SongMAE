#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/syllable_knn}"
MODELS="${MODELS:-xcl_large_500k_p32x4_c010 birdaves_biox_base hubert_base_ls960}"
LAYERS="${LAYERS:--1}"
K_VALUES="${K_VALUES:-1,5,10,20,50,100}"
MAX_REF_POINTS="${MAX_REF_POINTS:-200000}"
REF_MIN_PER_CLASS="${REF_MIN_PER_CLASS:-1000}"
MAX_QUERIES="${MAX_QUERIES:-5000}"
QUERY_PER_CLASS="${QUERY_PER_CLASS:-200}"
SEARCH_K="${SEARCH_K:-1000}"
NUM_TIMEBINS="${NUM_TIMEBINS:-0}"
OVERWRITE="${OVERWRITE:-0}"
TARGET_FEATURE_TYPE="end_of_block"
BIRDAVES_MODEL_PATH="${BIRDAVES_MODEL_PATH:-$ROOT/files/birdaves-biox-base.torchaudio.pt}"
BIRDAVES_CONFIG_PATH="${BIRDAVES_CONFIG_PATH:-$ROOT/files/birdaves-biox-base.torchaudio.model_config.json}"
HUBERT_MODEL_NAME="${HUBERT_MODEL_NAME:-facebook/hubert-base-ls960}"
WAV_EXTS="${WAV_EXTS:-.wav,.flac,.ogg,.mp3}"

DATASETS=(
  "zf|files/annotation jsons/zf_annotations.json|/media/george-vengrovski/disk2/specs/zebra_finch_5ms|/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae"
  "bf|files/annotation jsons/bf_annotations.json|/media/george-vengrovski/disk2/specs/bengalese_finch_5ms|/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae"
  "canary|files/annotation jsons/canary_annotations.json|/media/george-vengrovski/disk2/specs/canary_5ms|/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae"
)

birds_in_json() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
birds = {r.get("recording", {}).get("bird_id", "") for r in data["recordings"]}
print("\n".join(sorted(x for x in birds if x)))
PY
}

selected_dataset() {
  [[ "$#" -eq 1 ]] && return 0
  local dataset="$1"; shift
  for target in "$@"; do [[ "$dataset" == "$target" ]] && return 0; done
  return 1
}

songmae_run_dir() {
  for path in "$ROOT/runs/$1" "$ROOT/runs/dirty_runs/$1"; do
    [[ -d "$path" ]] && { echo "$path"; return 0; }
  done
  return 1
}

read -r -a MODEL_LIST <<< "$MODELS"
read -r -a LAYER_LIST <<< "$LAYERS"
mkdir -p "$OUT_ROOT"

for row in "${DATASETS[@]}"; do
  IFS="|" read -r dataset json spec_dir wav_dir <<< "$row"
  selected_dataset "$dataset" "$@" || continue
  while IFS= read -r bird; do
    for model in "${MODEL_LIST[@]}"; do
      for layer in "${LAYER_LIST[@]}"; do
        out_dir="$OUT_ROOT/$dataset/$bird/$model/layer_${layer}"
        [[ "$OVERWRITE" == "1" ]] && rm -rf "$out_dir"
        [[ -f "$out_dir/summary.json" ]] && { echo "exists: $out_dir/summary.json"; continue; }
        args=( "$PYTHON_BIN" src/embeddings/syllable_knn.py --spec_dir "$spec_dir" --annotation_file "$json" --out_dir "$out_dir" --bird "$bird" --encoder_layer_idx "$layer" --target_feature_type "$TARGET_FEATURE_TYPE" --num_timebins "$NUM_TIMEBINS" --k_values "$K_VALUES" --max_ref_points "$MAX_REF_POINTS" --ref_min_per_class "$REF_MIN_PER_CLASS" --max_queries "$MAX_QUERIES" --query_per_class "$QUERY_PER_CLASS" --search_k "$SEARCH_K" --wav_exts "$WAV_EXTS" )
        case "$model" in
          birdaves_biox_base) args+=(--model aves --name birdaves_biox_base --wav_dir "$wav_dir" --aves_model_path "$BIRDAVES_MODEL_PATH" --aves_config_path "$BIRDAVES_CONFIG_PATH") ;;
          hubert_base_ls960) args+=(--model hubert --name hubert_base_ls960 --wav_dir "$wav_dir" --hubert_model_name "$HUBERT_MODEL_NAME") ;;
          *) run_dir="$(songmae_run_dir "$model")"; [[ -d "$run_dir" ]] || { echo "missing run_dir: $run_dir"; continue; }; args+=(--model songmae --name "$model" --songmae_run_dir "$run_dir") ;;
        esac
        echo "running: dataset=$dataset bird=$bird model=$model layer=$layer"
        "${args[@]}"
      done
    done
  done < <(birds_in_json "$json")
done
