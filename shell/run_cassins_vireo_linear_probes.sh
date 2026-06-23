#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/syllable_linear_probe}"
MODELS="${MODELS:-xcl_tiny_500k_p32x4_default xcl_base_500k_p32x4_default xcl_micro_500k_p16x1_default xcl_micro_500k_p32x1_default xcl_micro_500k_p32x4_default aves hubert}"
PROBE_MODEL="${PROBE_MODEL:-logreg}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
SEED="${SEED:-42}"
OVERWRITE="${OVERWRITE:-0}"
SAVE_PLOTS="${SAVE_PLOTS:-0}"
CLEAN_EMBEDDINGS="${CLEAN_EMBEDDINGS:-1}"
MAX_PROBE_SECONDS="${MAX_PROBE_SECONDS:-3600}"
PROBE_TIMEBINS_PER_SECOND="${PROBE_TIMEBINS_PER_SECOND:-200}"
PROBE_NUM_TIMEBINS="${PROBE_NUM_TIMEBINS:-$((MAX_PROBE_SECONDS * PROBE_TIMEBINS_PER_SECOND))}"

XCL_TINY_P32X4_RUN_DIR="${XCL_TINY_P32X4_RUN_DIR:-$ROOT/runs/xcl_tiny_500k_p32x4_default}"
XCL_BASE_P32X4_RUN_DIR="${XCL_BASE_P32X4_RUN_DIR:-$ROOT/runs/xcl_base_500k_p32x4_default}"
XCL_MICRO_P16X1_RUN_DIR="${XCL_MICRO_P16X1_RUN_DIR:-$ROOT/runs/xcl_micro_500k_p16x1_default}"
XCL_MICRO_P32X1_RUN_DIR="${XCL_MICRO_P32X1_RUN_DIR:-$ROOT/runs/xcl_micro_500k_p32x1_default}"
XCL_MICRO_P32X4_RUN_DIR="${XCL_MICRO_P32X4_RUN_DIR:-$ROOT/runs/xcl_micro_500k_p32x4_default}"
AVES_MODEL_PATH="${AVES_MODEL_PATH:-$ROOT/files/aves-base-bio.torchaudio.pt}"
AVES_CONFIG_PATH="${AVES_CONFIG_PATH:-$ROOT/files/aves-base-bio.torchaudio.model_config.json}"
AVES_AUDIO_SR="${AVES_AUDIO_SR:-16000}"
HUBERT_MODEL_NAME="${HUBERT_MODEL_NAME:-facebook/hubert-large-ll60k}"
HUBERT_AUDIO_SR="${HUBERT_AUDIO_SR:-16000}"
WAV_EXTS="${WAV_EXTS:-.wav,.flac,.ogg,.mp3}"

DATASET="cassins_vireo"
ANNOTATIONS="${CASSINS_VIREO_JSON:-files/annotation jsons/cassins_vireo_annotations.json}"
SPEC_DIR="${CASSINS_VIREO_SPEC_DIR:-/media/george-vengrovski/disk2/specs/cassins_vireo_5ms}"
WAV_DIR="${CASSINS_VIREO_WAV_DIR:-/media/george-vengrovski/disk2/raw_data/cassins_vireo/data/figshare_3081814/wav_files}"
RECORDING_MODE="${CASSINS_VIREO_RECORDING_MODE:-events}"

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

has_split_coverage() {
  "$PYTHON_BIN" - "$ANNOTATIONS" "$1" "$PROBE_NUM_TIMEBINS" "$PROBE_TIMEBINS_PER_SECOND" <<'PY'
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

ok = groups >= 2 and counts and all(count > 1 for count in counts.values())
if not ok:
    rare = sorted(label for label, count in counts.items() if count <= 1)
    print(f"skip: bird={bird} groups={groups} classes={len(counts)} rare_classes={rare[:12]}", file=sys.stderr)
sys.exit(0 if ok else 1)
PY
}

songmae_run_dir() {
  case "$1" in
    xcl_tiny_500k_p32x4_default) echo "$XCL_TINY_P32X4_RUN_DIR" ;;
    xcl_base_500k_p32x4_default) echo "$XCL_BASE_P32X4_RUN_DIR" ;;
    xcl_micro_500k_p16x1_default) echo "$XCL_MICRO_P16X1_RUN_DIR" ;;
    xcl_micro_500k_p32x1_default) echo "$XCL_MICRO_P32X1_RUN_DIR" ;;
    xcl_micro_500k_p32x4_default) echo "$XCL_MICRO_P32X4_RUN_DIR" ;;
    *)
      echo "Unknown SongMAE/XCL model: $1" 1>&2
      return 1
      ;;
  esac
}

extract_embeddings() {
  local model="$1" bird="$2" out_dir="$3"
  case "$model" in
    xcl_tiny_500k_p32x4_default|xcl_base_500k_p32x4_default|xcl_micro_500k_p16x1_default|xcl_micro_500k_p32x1_default|xcl_micro_500k_p32x4_default)
      cmd=(
        "$PYTHON_BIN" -m src.core.extract_embedding
        --run_dir "$(songmae_run_dir "$model")"
        --spec_dir "$SPEC_DIR"
        --json_path "$ANNOTATIONS"
        --bird "$bird"
        --recording_mode "$RECORDING_MODE"
        --out_dir "$out_dir"
        --num_timebins "$PROBE_NUM_TIMEBINS"
      )
      if [[ -n "${SONGMAE_CHECKPOINT:-}" ]]; then cmd+=(--checkpoint "$SONGMAE_CHECKPOINT"); fi
      "${cmd[@]}"
      ;;
    aves)
      "$PYTHON_BIN" src/external_models/aves.py \
        --spec_dir "$SPEC_DIR" \
        --wav_dir "$WAV_DIR" \
        --annotation_file "$ANNOTATIONS" \
        --out_dir "$out_dir" \
        --bird "$bird" \
        --recording_mode "$RECORDING_MODE" \
        --aves_model_path "$AVES_MODEL_PATH" \
        --aves_config_path "$AVES_CONFIG_PATH" \
        --audio_sr "$AVES_AUDIO_SR" \
        --wav_exts "$WAV_EXTS" \
        --num_timebins "$PROBE_NUM_TIMEBINS"
      ;;
    hubert)
      "$PYTHON_BIN" src/external_models/hubert.py \
        --spec_dir "$SPEC_DIR" \
        --wav_dir "$WAV_DIR" \
        --annotation_file "$ANNOTATIONS" \
        --out_dir "$out_dir" \
        --bird "$bird" \
        --recording_mode "$RECORDING_MODE" \
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

run_probe() {
  local bird="$1" model="$2"
  local run_dir="$OUT_ROOT/$DATASET/$bird/$model"
  local embed_dir="$run_dir/embeddings"
  local metrics_path="$run_dir/metrics.json"
  local metrics_tmp="$run_dir/metrics.tmp"

  if [[ -f "$metrics_path" && "$OVERWRITE" != "1" ]]; then
    echo "exists: $metrics_path"
    return 0
  fi

  rm -rf "$run_dir"
  mkdir -p "$embed_dir"
  echo "running: dataset=$DATASET bird=$bird model=$model"
  if ! extract_embeddings "$model" "$bird" "$embed_dir"; then
    echo "extract failed: dataset=$DATASET bird=$bird model=$model" 1>&2
    return 1
  fi

  plot_args=()
  if [[ "$SAVE_PLOTS" == "1" ]]; then
    plot_args=(--save_plots --plot_dir "$run_dir/prediction_plots")
  fi

  if "$PYTHON_BIN" src/evals/syllable_classification.py \
    --embeddings "$embed_dir" \
    --annotations "$ANNOTATIONS" \
    --model "$PROBE_MODEL" \
    --val_fraction "$VAL_FRACTION" \
    --seed "$SEED" \
    "${plot_args[@]}" > "$metrics_tmp"; then
    mv "$metrics_tmp" "$metrics_path"
    if [[ "$CLEAN_EMBEDDINGS" == "1" ]]; then
      rm -rf "$embed_dir"
      echo "cleaned: $embed_dir"
    fi
  else
    rm -f "$metrics_tmp"
    echo "probe failed: dataset=$DATASET bird=$bird model=$model" 1>&2
    return 1
  fi
}

read -r -a MODEL_LIST <<< "$MODELS"
mkdir -p "$OUT_ROOT"

failed=0
while IFS= read -r bird; do
  if ! has_split_coverage "$bird"; then
    continue
  fi
  for model in "${MODEL_LIST[@]}"; do
    run_probe "$bird" "$model" || failed=1
  done
done < <(birds_in_json "$ANNOTATIONS")
exit "$failed"
