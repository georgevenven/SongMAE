#!/usr/bin/env bash
# Three-fold capped-K=50 ranking for every micro model and annotated bird.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)

PYTHON_BIN=${PYTHON_BIN:-/home/george-vengrovski/anaconda3/envs/mae/bin/python}
OUT_ROOT=${OUT_ROOT:-$ROOT/results/micro_model_capped_probe_k50}
LABEL_CAP=${LABEL_CAP:-50}
NUM_TIMEBINS=${NUM_TIMEBINS:-200000}
PCA_COMPONENTS=${PCA_COMPONENTS:-128}
STEPS=${STEPS:-1000}
BATCH_SIZE=${BATCH_SIZE:-256}
SEED=${SEED:-42}
CLEAN_EMBEDDINGS=${CLEAN_EMBEDDINGS:-1}
DATASET_FILTER=${DATASET_FILTER:-}
BIRD_FILTER=${BIRD_FILTER:-}
MODEL_FILTER=${MODEL_FILTER:-}

DATASETS=(
  "zf|files/annotation jsons/zf_annotations.json|/media/george-vengrovski/disk2/specs/zebra_finch_5ms"
  "bf|files/annotation jsons/bf_annotations.json|/media/george-vengrovski/disk2/specs/bengalese_finch_5ms"
  "canary|files/annotation jsons/canary_annotations.json|/media/george-vengrovski/disk2/specs/canary_5ms"
)

selected() {
  [[ -z "$2" || " $2 " == *" $1 "* ]]
}

birds() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
print("\n".join(sorted({row["recording"]["bird_id"] for row in data["recordings"]})))
PY
}

mapfile -t MODELS < <("$PYTHON_BIN" - <<'PY'
from src.evals.micro_model_linear_probe_table_aggregator import SECTIONS

models = dict.fromkeys(model for _, rows in SECTIONS for _, model in rows)
print("\n".join(models))
PY
)

ACTIVE_MODELS=()
for model in "${MODELS[@]}"; do
  checkpoint="$ROOT/runs/$model/weights/model_step_099999.pth"
  if [[ -f "$checkpoint" ]]; then
    ACTIVE_MODELS+=("$model")
  else
    echo "missing checkpoint, skipping: $checkpoint" >&2
  fi
done

mkdir -p "$OUT_ROOT"
for row in "${DATASETS[@]}"; do
  IFS="|" read -r dataset annotations specs <<< "$row"
  selected "$dataset" "$DATASET_FILTER" || continue
  while read -r bird; do
    selected "$bird" "$BIRD_FILTER" || continue
    manifest="$OUT_ROOT/manifests/$dataset/$bird.json"
    for model in "${ACTIVE_MODELS[@]}"; do
      selected "$model" "$MODEL_FILTER" || continue
      model_dir="$OUT_ROOT/$dataset/$bird/$model"
      embeddings="$model_dir/embeddings"
      metrics="$model_dir/metrics.json"
      [[ -f "$metrics" ]] && { echo "exists: $metrics"; continue; }

      if [[ ! -f "$embeddings/metadata.json" ]]; then
        rm -rf "$embeddings" "$embeddings.tmp"
        mkdir -p "$model_dir"
        started=$SECONDS
        echo "extracting: dataset=$dataset bird=$bird model=$model"
        if ! "$PYTHON_BIN" -m src.core.extract_embedding \
          --spec_dir "$specs" --run_dir "runs/$model" --checkpoint model_step_099999.pth \
          --out_dir "$embeddings" --json_path "$annotations" --num_timebins "$NUM_TIMEBINS" \
          --recording_mode events --bird "$bird" --minimal --target_feature_type end_of_block \
          --balanced_events 3 --event_seed "$SEED"; then
          echo "extraction failed: dataset=$dataset bird=$bird model=$model" >&2
          continue
        fi
        printf '%s\n' "$((SECONDS - started))" > "$model_dir/extraction_seconds.txt"
      fi

      manifest_args=(--manifest_in "$manifest")
      [[ -f "$manifest" ]] || manifest_args=(--manifest_out "$manifest")
      echo "probing: dataset=$dataset bird=$bird model=$model"
      if "$PYTHON_BIN" src/evals/syllable_classification_capped.py \
        --embeddings "$embeddings" --annotations "$annotations" --label_cap "$LABEL_CAP" --folds 3 \
        "${manifest_args[@]}" --pca_components "$PCA_COMPONENTS" \
        --pca_cache "$embeddings/pca_${PCA_COMPONENTS}_seed${SEED}.npy" \
        --steps "$STEPS" --batch_size "$BATCH_SIZE" --seed "$SEED" > "$model_dir/metrics.tmp"; then
        mv "$model_dir/metrics.tmp" "$metrics"
        [[ "$CLEAN_EMBEDDINGS" == 1 ]] && rm -rf "$embeddings"
      else
        rm -f "$model_dir/metrics.tmp"
        echo "probe failed: dataset=$dataset bird=$bird model=$model" >&2
      fi
    done
  done < <(birds "$annotations")
done
