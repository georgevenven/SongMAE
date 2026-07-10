#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/syllable_classification_train_sweep}"smoke
CLASSIFIER="${CLASSIFIER:-mlp}"
FEATURE_KEY="${FEATURE_KEY:-encoded_embeddings}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
LR="${LR:-0.001}"
TRAIN_SECONDS="${TRAIN_SECONDS:-32 64 128 256 512 MAX}"
OVERWRITE="${OVERWRITE:-0}"
CLEAN_EMBEDDINGS="${CLEAN_EMBEDDINGS:-1}"
MAX_PROBE_SECONDS="${MAX_PROBE_SECONDS:-3600}"
PROBE_TIMEBINS_PER_SECOND="${PROBE_TIMEBINS_PER_SECOND:-200}"
PROBE_NUM_TIMEBINS="${PROBE_NUM_TIMEBINS:-$((MAX_PROBE_SECONDS * PROBE_TIMEBINS_PER_SECOND))}"

BIRDAVES_MODEL_PATH="${BIRDAVES_MODEL_PATH:-$ROOT/files/birdaves-biox-base.torchaudio.pt}"
BIRDAVES_CONFIG_PATH="${BIRDAVES_CONFIG_PATH:-$ROOT/files/birdaves-biox-base.torchaudio.model_config.json}"
BIRDAVES_AUDIO_SR="${BIRDAVES_AUDIO_SR:-16000}"
HUBERT_MODEL_NAME="${HUBERT_MODEL_NAME:-facebook/hubert-base-ls960}"
HUBERT_AUDIO_SR="${HUBERT_AUDIO_SR:-16000}"
WAV_EXTS="${WAV_EXTS:-.wav,.flac,.ogg,.mp3}"

# Hardcoded for now because these evals need matching annotation, spec, and wav roots.
# Positional args filter this list, e.g. `bash shell/syllable_classification_train_sweep.sh zf`.
DATASETS=(
  "zf|files/annotation jsons/zf_annotations.json|/media/george-vengrovski/disk2/specs/zebra_finch_5ms|/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae|events"
  "bf|files/annotation jsons/bf_annotations.json|/media/george-vengrovski/disk2/specs/bengalese_finch_5ms|/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae|events"
  "canary|files/annotation jsons/canary_annotations.json|/media/george-vengrovski/disk2/specs/canary_5ms|/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae|events"
)

usage() {
  echo "Usage: $0 [zf|bf|canary ...]" 1>&2
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

default_models() {
  local run
  for run in "$ROOT"/runs/*; do
    [[ -f "$run/config.json" ]] || continue
    compgen -G "$run/weights/model_step_*.pth" > /dev/null || continue
    basename "$run"
  done
  printf '%s\n' birdaves_biox_base hubert_base_ls960
}

model_slug() {
  case "$1" in
    birdaves|BirdAVES|birdaves_biox_base) echo birdaves_biox_base ;;
    hubert_base|hubert_base_ls960|HuBERT) echo hubert_base_ls960 ;;
    *) basename "$1" ;;
  esac
}

extract_embeddings() {
  local model="$1" json="$2" spec_dir="$3" wav_dir="$4" mode="$5" bird="$6" out_dir="$7"
  local cmd run_dir
  case "$model" in
    birdaves|BirdAVES|birdaves_biox_base)
      "$PYTHON_BIN" src/external_models/aves.py \
        --spec_dir "$spec_dir" \
        --wav_dir "$wav_dir" \
        --annotation_file "$json" \
        --out_dir "$out_dir" \
        --bird "$bird" \
        --recording_mode "$mode" \
        --aves_model_path "$BIRDAVES_MODEL_PATH" \
        --aves_config_path "$BIRDAVES_CONFIG_PATH" \
        --model_name birdaves_biox_base \
        --audio_sr "$BIRDAVES_AUDIO_SR" \
        --wav_exts "$WAV_EXTS" \
        --num_timebins "$PROBE_NUM_TIMEBINS"
      ;;
    hubert_base|hubert_base_ls960|HuBERT)
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
      run_dir="$model"
      [[ -f "$run_dir/config.json" ]] || run_dir="$ROOT/runs/$model"
      if [[ ! -f "$run_dir/config.json" ]]; then
        echo "Unknown model: $model" 1>&2
        return 1
      fi
      cmd=(
        "$PYTHON_BIN" -m src.core.extract_embedding
        --run_dir "$run_dir"
        --spec_dir "$spec_dir"
        --json_path "$json"
        --bird "$bird"
        --recording_mode "$mode"
        --out_dir "$out_dir"
        --minimal
        --num_timebins "$PROBE_NUM_TIMEBINS"
      )
      if [[ -n "${SONGMAE_CHECKPOINT:-}" ]]; then cmd+=(--checkpoint "$SONGMAE_CHECKPOINT"); fi
      "${cmd[@]}"
      ;;
  esac
}

all_metrics_exist() {
  local model_dir="$1"
  local budget
  for budget in "${BUDGETS[@]}"; do
    [[ -f "$model_dir/train_${budget}s/metrics.json" ]] || return 1
  done
  return 0
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TARGETS=("$@")
if [[ -n "${MODELS:-}" ]]; then
  read -r -a MODEL_LIST <<< "$MODELS"
else
  mapfile -t MODEL_LIST < <(default_models)
fi
read -r -a BUDGETS <<< "$TRAIN_SECONDS"
mkdir -p "$OUT_ROOT"

for row in "${DATASETS[@]}"; do
  IFS="|" read -r dataset json spec_dir wav_dir recording_mode <<< "$row"
  selected_dataset "$dataset" || continue

  while IFS= read -r bird; do
    if ! has_train_coverage "$json" "$bird"; then
      continue
    fi
    for model in "${MODEL_LIST[@]}"; do
      model_name="$(model_slug "$model")"
      model_dir="$OUT_ROOT/$dataset/$bird/$model_name"
      embed_dir="$model_dir/embeddings"
      if [[ "$OVERWRITE" != "1" ]] && all_metrics_exist "$model_dir"; then
        for budget in "${BUDGETS[@]}"; do
          echo "exists: $model_dir/train_${budget}s/metrics.json"
        done
        continue
      fi
      if [[ "$OVERWRITE" == "1" ]]; then
        rm -rf "$model_dir"
      fi
      mkdir -p "$embed_dir"
      echo "extracting: dataset=$dataset bird=$bird model=$model_name"
      if ! extract_embeddings "$model" "$json" "$spec_dir" "$wav_dir" "$recording_mode" "$bird" "$embed_dir"; then
        echo "extract failed: dataset=$dataset bird=$bird model=$model_name" 1>&2
        if [[ "$CLEAN_EMBEDDINGS" == "1" ]]; then
          rm -rf "$embed_dir"
        fi
        continue
      fi

      for budget in "${BUDGETS[@]}"; do
        run_dir="$model_dir/train_${budget}s"
        metrics="$run_dir/metrics.json"
        tmp="$run_dir/metrics.tmp"
        split_json="$run_dir/split.json"
        if [[ -f "$metrics" && "$OVERWRITE" != "1" ]]; then
          echo "exists: $metrics"
          continue
        fi

        mkdir -p "$run_dir"
        split_args=(--split_json "$split_json")
        echo "running: dataset=$dataset bird=$bird model=$model_name train=$budget"
        if "$PYTHON_BIN" src/evals/syllable_classification.py \
          --embeddings "$embed_dir" \
          --annotations "$json" \
          --model "$CLASSIFIER" \
          --feature_key "$FEATURE_KEY" \
          --val_fraction "$VAL_FRACTION" \
          --seed "$SEED" \
          --epochs "$EPOCHS" \
          --batch_size "$BATCH_SIZE" \
          --lr "$LR" \
          --max_train_seconds "$budget" \
          "${split_args[@]}" > "$tmp"; then
          mv "$tmp" "$metrics"
        else
          rm -f "$tmp"
          echo "failed: dataset=$dataset bird=$bird model=$model_name train=$budget" 1>&2
        fi
      done

      if [[ "$CLEAN_EMBEDDINGS" == "1" ]]; then
        rm -rf "$embed_dir"
        echo "cleaned: $embed_dir"
      fi
    done
  done < <(birds_in_json "$json")
done
