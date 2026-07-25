#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

K_VALUES="${K_VALUES:-1,5,10}"
NUM_TIMEBINS="${NUM_TIMEBINS:-200000}"
REFERENCE_OCCURRENCES_PER_CLASS="${REFERENCE_OCCURRENCES_PER_CLASS:-100}"
QUERY_OCCURRENCES_PER_CLASS="${QUERY_OCCURRENCES_PER_CLASS:-20}"
SEARCH_K="${SEARCH_K:-1000}"
PCA_COMPONENTS="${PCA_COMPONENTS:-0}"
OVERWRITE="${OVERWRITE:-0}"
BIRDAVES_MODEL_PATH="${BIRDAVES_MODEL_PATH:-$ROOT/files/birdaves-biox-base.torchaudio.pt}"
BIRDAVES_CONFIG_PATH="${BIRDAVES_CONFIG_PATH:-$ROOT/files/birdaves-biox-base.torchaudio.model_config.json}"
HUBERT_MODEL_NAME="${HUBERT_MODEL_NAME:-facebook/hubert-base-ls960}"
WAV_EXTS="${WAV_EXTS:-.wav,.flac,.ogg,.mp3}"
SPEC_ROOT="${SPEC_ROOT:-/media/george-vengrovski/disk2/specs}"
WAV_ROOT="${WAV_ROOT:-/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae}"

DATASETS=(
  "zf|files/annotation jsons/zf_annotations.json|$SPEC_ROOT/zebra_finch_5ms|$WAV_ROOT"
  "bf|files/annotation jsons/bf_annotations.json|$SPEC_ROOT/bengalese_finch_5ms|$WAV_ROOT"
  "canary|files/annotation jsons/canary_annotations.json|$SPEC_ROOT/canary_5ms|$WAV_ROOT"
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

songmae_run_dir() {
  for path in "$ROOT/runs/$1" "$ROOT/runs/dirty_runs/$1"; do
    [[ -d "$path" ]] && { echo "$path"; return; }
  done
  echo "missing run dir for $1" >&2
  return 1
}

knn_out_dir() {
  local root="$1" species="$2" bird="$3" model="$4" layer="$5" target="$6"
  case "$model" in
    birdaves_biox_base|hubert_base_ls960) echo "$root/$species/$bird/$model/layer_$layer" ;;
    *) echo "$root/$species/$bird/$model/layer_$layer/$target" ;;
  esac
}

cleanup_embeddings() {
  rm -rf "$1/embeddings" "$1/embeddings.tmp"
}

run_knn() {
  local model="$1" out_dir="$2" json="$3" spec_dir="$4" wav_dir="$5" bird="$6" layer="$7" target="$8" embedding_dir="${9:-}"
  [[ "$OVERWRITE" == "1" ]] && rm -rf "$out_dir"
  if [[ -f "$out_dir/summary.json" ]]; then
    cleanup_embeddings "$out_dir"
    echo "exists $out_dir/summary.json"
    return
  fi

  mkdir -p "$out_dir"
  local args=(
    "$PYTHON_BIN" src/embeddings/syllable_knn.py
    --spec_dir "$spec_dir" --annotation_file "$json" --out_dir "$out_dir"
    --bird "$bird" --encoder_layer_idx "$layer"
    --num_timebins "$NUM_TIMEBINS"
    --k_values "$K_VALUES"
    --reference_occurrences_per_class "$REFERENCE_OCCURRENCES_PER_CLASS"
    --query_occurrences_per_class "$QUERY_OCCURRENCES_PER_CLASS"
    --search_k "$SEARCH_K"
    --pca_components "$PCA_COMPONENTS"
    --wav_exts "$WAV_EXTS"
  )
  [[ -n "$embedding_dir" ]] && args+=(--embedding_dir "$embedding_dir")

  case "$model" in
    birdaves_biox_base)
      args+=(--model aves --name birdaves_biox_base --wav_dir "$wav_dir" --aves_model_path "$BIRDAVES_MODEL_PATH" --aves_config_path "$BIRDAVES_CONFIG_PATH")
      ;;
    hubert_base_ls960)
      args+=(--model hubert --name hubert_base_ls960 --wav_dir "$wav_dir" --hubert_model_name "$HUBERT_MODEL_NAME")
      ;;
    *)
      args+=(--model songmae --name "${KNN_NAME:-$model}" --songmae_run_dir "$(songmae_run_dir "$model")" --target_feature_type "$target")
      [[ -n "${SONGMAE_CHECKPOINT:-}" ]] && args+=(--checkpoint "$SONGMAE_CHECKPOINT")
      ;;
  esac

  echo "running $out_dir"
  "${args[@]}" > "$out_dir/run.log" 2>&1 || { tail -80 "$out_dir/run.log"; return 1; }
}

run_knn_pair() {
  local raw_out="$1" pca_out="$2" model="$3" json="$4" specs="$5"
  local wavs="$6" bird="$7" layer="$8" target="$9" embeddings=""
  if [[ ! -f "$raw_out/summary.json" ]]; then
    PCA_COMPONENTS=0 run_knn "$model" "$raw_out" "$json" "$specs" "$wavs" "$bird" "$layer" "$target" || return
    embeddings="$raw_out/embeddings"
  fi
  if [[ ! -f "$pca_out/summary.json" ]]; then
    PCA_COMPONENTS=128 run_knn "$model" "$pca_out" "$json" "$specs" "$wavs" "$bird" "$layer" "$target" "$embeddings" || return
  fi
  cleanup_embeddings "$raw_out"
  cleanup_embeddings "$pca_out"
}
