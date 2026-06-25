#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/syllable_linear_probe}"
MODELS="${MODELS:-xcl_micro_500k_p16x1_default xcl_micro_500k_p32x1_default xcl_micro_500k_p32x4_default xcl_micro_500k_p128x1_default}"
PROBE_MODEL="${PROBE_MODEL:-logreg}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
SEED="${SEED:-42}"
OVERWRITE="${OVERWRITE:-0}"
SAVE_PLOTS="${SAVE_PLOTS:-0}"
CLEAN_EMBEDDINGS="${CLEAN_EMBEDDINGS:-1}"
MAX_PROBE_SECONDS="${MAX_PROBE_SECONDS:-3600}"
PROBE_TIMEBINS_PER_SECOND="${PROBE_TIMEBINS_PER_SECOND:-200}"
PROBE_NUM_TIMEBINS="${PROBE_NUM_TIMEBINS:-$((MAX_PROBE_SECONDS * PROBE_TIMEBINS_PER_SECOND))}"

SONGMAE_RUN_DIR="${SONGMAE_RUN_DIR:-$ROOT/runs/xcl_full_500k_bs256_5s_p32x10}"
XCL_TINY_P32X4_RUN_DIR="${XCL_TINY_P32X4_RUN_DIR:-$ROOT/runs/xcl_tiny_500k_p32x4_default}"
XCL_BASE_P32X4_RUN_DIR="${XCL_BASE_P32X4_RUN_DIR:-$ROOT/runs/xcl_base_500k_p32x4_default}"
XCL_MICRO_P128X1_RUN_DIR="${XCL_MICRO_P128X1_RUN_DIR:-$ROOT/runs/xcl_micro_500k_p128x1_default}"
XCL_MICRO_P16X1_RUN_DIR="${XCL_MICRO_P16X1_RUN_DIR:-$ROOT/runs/xcl_micro_500k_p16x1_default}"
XCL_MICRO_P32X1_RUN_DIR="${XCL_MICRO_P32X1_RUN_DIR:-$ROOT/runs/xcl_micro_500k_p32x1_default}"
XCL_MICRO_P32X4_RUN_DIR="${XCL_MICRO_P32X4_RUN_DIR:-$ROOT/runs/xcl_micro_500k_p32x4_default}"
AVES_MODEL_PATH="${AVES_MODEL_PATH:-$ROOT/files/aves-base-bio.torchaudio.pt}"
AVES_CONFIG_PATH="${AVES_CONFIG_PATH:-$ROOT/files/aves-base-bio.torchaudio.model_config.json}"
AVES_AUDIO_SR="${AVES_AUDIO_SR:-16000}"
HUBERT_MODEL_NAME="${HUBERT_MODEL_NAME:-facebook/hubert-large-ll60k}"
HUBERT_AUDIO_SR="${HUBERT_AUDIO_SR:-16000}"
WAV_EXTS="${WAV_EXTS:-.wav,.flac,.ogg,.mp3}"

# Hardcoded for now because these evals need matching annotation, spec, and wav roots.
# Positional args filter this list, e.g. `bash shell/linear_probe_across_models.sh cassins_vireo`.
DATASETS=(
  # "zf|files/annotation jsons/zf_annotations.json|/media/george-vengrovski/disk2/specs/zebra_finch_5ms|/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae|events"
  # "bf|files/annotation jsons/bf_annotations.json|/media/george-vengrovski/disk2/specs/bengalese_finch_5ms|/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae|events"
  # "canary|files/annotation jsons/canary_annotations.json|/media/george-vengrovski/disk2/specs/canary_5ms|/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae|events"
  "cassins_vireo|files/annotation jsons/cassins_vireo_annotations.json|/media/george-vengrovski/disk2/specs/cassins_vireo_5ms|/media/george-vengrovski/disk2/raw_data/cassins_vireo/data/figshare_3081814/wav_files|events"
  "american_robin|files/annotation jsons/american_robin_annotations.json|/media/george-vengrovski/disk2/specs/american_robin_5ms|/media/george-vengrovski/disk2/raw_data/american_robin/data/rmbl_robin/RMBL-Robin/data|events"
)

usage() {
  echo "Usage: $0 [zf|bf|canary|cassins_vireo|american_robin ...]" 1>&2
}

selected_dataset() {
  local dataset="$1"
  if [[ "$#" -eq 1 && "${#TARGETS[@]}" -eq 0 ]]; then
    return 0
  fi
  for target in "${TARGETS[@]}"; do
    [[ "$dataset" == "$target" ]] && return 0
  done
  return 1
}

birds_in_json() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
birds = {
    recording.get("recording", {}).get("bird_id", "")
    for recording in data["recordings"]
}
for bird in sorted(bird for bird in birds if bird):
    print(bird)
PY
}

strict_split_dataset() {
  case "$1" in
    cassins_vireo|american_robin) return 1 ;;
    *) return 0 ;;
  esac
}

has_train_coverage() {
  "$PYTHON_BIN" - "$1" "$2" "$PROBE_NUM_TIMEBINS" "$PROBE_TIMEBINS_PER_SECOND" <<'PY'
import json
import sys
from pathlib import Path

path, bird, max_timebins, bins_per_second = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
data = json.loads(Path(path).read_text())
counts = {}
groups = 0
used = 0
for recording in data["recordings"]:
    if recording.get("recording", {}).get("bird_id", "") != bird:
        continue
    for event in recording.get("detected_events", []):
        duration_ms = float(event["offset_ms"]) - float(event["onset_ms"])
        duration_bins = max(1, int(round(duration_ms / 1000.0 * bins_per_second)))
        if max_timebins > 0 and used >= max_timebins:
            break
        used += duration_bins
        classes = {int(unit["id"]) for unit in event.get("units", [])}
        if classes:
            groups += 1
            for label in classes:
                counts[label] = counts.get(label, 0) + 1
    if max_timebins > 0 and used >= max_timebins:
        break

ok = groups >= 2 and counts
if not ok:
    print(f"skip: bird={bird} groups={groups} classes={len(counts)}", file=sys.stderr)
sys.exit(0 if ok else 1)
PY
}

songmae_run_dir() {
  case "$1" in
    songmae|songmae_random) echo "$SONGMAE_RUN_DIR" ;;
    xcl_tiny_500k_p32x4_default) echo "$XCL_TINY_P32X4_RUN_DIR" ;;
    xcl_base_500k_p32x4_default) echo "$XCL_BASE_P32X4_RUN_DIR" ;;
    xcl_micro_500k_p128x1_default) echo "$XCL_MICRO_P128X1_RUN_DIR" ;;
    xcl_micro_500k_p16x1_default) echo "$XCL_MICRO_P16X1_RUN_DIR" ;;
    xcl_micro_500k_p32x1_default) echo "$XCL_MICRO_P32X1_RUN_DIR" ;;
    xcl_micro_500k_p32x4_default) echo "$XCL_MICRO_P32X4_RUN_DIR" ;;
    *) return 1 ;;
  esac
}

extract_embeddings() {
  local model="$1" json="$2" spec_dir="$3" wav_dir="$4" mode="$5" bird="$6" out_dir="$7"
  case "$model" in
    songmae|songmae_random|xcl_tiny_500k_p32x4_default|xcl_base_500k_p32x4_default|xcl_micro_500k_p128x1_default|xcl_micro_500k_p16x1_default|xcl_micro_500k_p32x1_default|xcl_micro_500k_p32x4_default)
      cmd=(
        "$PYTHON_BIN" -m src.core.extract_embedding
        --run_dir "$(songmae_run_dir "$model")" \
        --spec_dir "$spec_dir" \
        --json_path "$json" \
        --bird "$bird" \
        --recording_mode "$mode" \
        --out_dir "$out_dir" \
        --minimal \
        --num_timebins "$PROBE_NUM_TIMEBINS"
      )
      if [[ -n "${SONGMAE_CHECKPOINT:-}" ]]; then cmd+=(--checkpoint "$SONGMAE_CHECKPOINT"); fi
      if [[ "$model" == "songmae_random" ]]; then cmd+=(--random_init); fi
      "${cmd[@]}"
      ;;
    aves)
      "$PYTHON_BIN" src/external_models/aves.py \
        --spec_dir "$spec_dir" \
        --wav_dir "$wav_dir" \
        --annotation_file "$json" \
        --out_dir "$out_dir" \
        --bird "$bird" \
        --recording_mode "$mode" \
        --aves_model_path "$AVES_MODEL_PATH" \
        --aves_config_path "$AVES_CONFIG_PATH" \
        --audio_sr "$AVES_AUDIO_SR" \
        --wav_exts "$WAV_EXTS" \
        --num_timebins "$PROBE_NUM_TIMEBINS"
      ;;
    hubert)
      "$PYTHON_BIN" src/external_models/hubert.py \
        --spec_dir "$spec_dir" \
        --wav_dir "$wav_dir" \
        --annotation_file "$json" \
        --out_dir "$out_dir" \
        --bird "$bird" \
        --recording_mode "$mode" \
        --model_name "$HUBERT_MODEL_NAME" \
        --audio_sr "$HUBERT_AUDIO_SR" \
        --wav_exts "$WAV_EXTS" \
        --num_timebins "$PROBE_NUM_TIMEBINS"
      ;;
    *)
      echo "Unknown model: $model" 1>&2
      return 1
      ;;
  esac
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TARGETS=("$@")
read -r -a MODEL_LIST <<< "$MODELS"
mkdir -p "$OUT_ROOT"

for row in "${DATASETS[@]}"; do
  IFS="|" read -r dataset json spec_dir wav_dir recording_mode <<< "$row"
  selected_dataset "$dataset" || continue

  while IFS= read -r bird; do
    if ! has_train_coverage "$json" "$bird"; then
      continue
    fi
    for model in "${MODEL_LIST[@]}"; do
      run_dir="$OUT_ROOT/$dataset/$bird/$model"
      embed_dir="$run_dir/embeddings"
      metrics_path="$run_dir/metrics.json"
      metrics_tmp="$run_dir/metrics.tmp"
      split_path="$run_dir/split.json"
      if [[ -f "$metrics_path" && "$OVERWRITE" != "1" ]]; then
        echo "exists: $metrics_path"
        continue
      fi

      rm -rf "$run_dir"
      mkdir -p "$embed_dir"
      echo "running: dataset=$dataset bird=$bird model=$model"
      if ! extract_embeddings "$model" "$json" "$spec_dir" "$wav_dir" "$recording_mode" "$bird" "$embed_dir"; then
        echo "extract failed: dataset=$dataset bird=$bird model=$model" 1>&2
        continue
      fi
      plot_args=()
      if [[ "$SAVE_PLOTS" == "1" ]]; then
        plot_args=(--save_plots --plot_dir "$run_dir/prediction_plots")
      fi
      split_args=(--split_json "$split_path")
      if ! strict_split_dataset "$dataset"; then
        split_args+=(--allow_unmatched_val_classes)
      fi
      if ! "$PYTHON_BIN" src/evals/syllable_classification.py \
        --embeddings "$embed_dir" \
        --annotations "$json" \
        --model "$PROBE_MODEL" \
        --val_fraction "$VAL_FRACTION" \
        --seed "$SEED" \
        "${split_args[@]}" \
        "${plot_args[@]}" > "$metrics_tmp"; then
        rm -f "$metrics_tmp"
        if [[ "$CLEAN_EMBEDDINGS" == "1" ]]; then
          rm -rf "$embed_dir"
          echo "cleaned: $embed_dir"
        fi
        echo "probe failed: dataset=$dataset bird=$bird model=$model" 1>&2
      else
        mv "$metrics_tmp" "$metrics_path"
        if [[ "$CLEAN_EMBEDDINGS" == "1" ]]; then
          rm -rf "$embed_dir"
          echo "cleaned: $embed_dir"
        fi
      fi
    done
  done < <(birds_in_json "$json")
done
