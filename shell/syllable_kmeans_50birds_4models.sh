#!/usr/bin/env bash
# Post-hoc rasterized K-means on the retained 50-bird UMAP embeddings.
set -u

cd "$(dirname "$0")/.."
PYTHON_BIN=${PYTHON_BIN:-/home/george-vengrovski/anaconda3/envs/mae/bin/python}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-/media/george-vengrovski/disk1/tinybird_umap_50birds_4models_250k_20260720}
PCA_COMPONENTS=${PCA_COMPONENTS:-8}
INCLUDE_SILENCE=${INCLUDE_SILENCE:-0}
MODEL_GROUP=${MODEL_GROUP:-large_comparison}
SILENCE_ARGS=()
SILENCE_SUFFIX=
if ((INCLUDE_SILENCE)); then
  SILENCE_ARGS+=(--include_silence)
  SILENCE_SUFFIX=_including_silence
fi
OUT_ROOT=${OUT_ROOT:-results/syllable_kmeans_50birds_4models_raster_pca${PCA_COMPONENTS}_250k${SILENCE_SUFFIX}}

DATASETS=(
  "zf|files/annotation jsons/zf_annotations.json"
  "bf|files/annotation jsons/bf_annotations.json"
  "canary|files/annotation jsons/canary_annotations.json"
)

case "$MODEL_GROUP" in
  large_comparison)
    MODELS=(
      "songmae_32x1|songmae_32x1/songmae/embeddings"
      "songmae_32x4|songmae_32x4/songmae/embeddings"
      "birdaves|birdaves/aves/embeddings"
      "hubert|hubert/hubert/embeddings"
    )
    ;;
  micro_base)
    MODELS=(
      "micro_32x1|micro_32x1/songmae/embeddings"
      "micro_32x4|micro_32x4/songmae/embeddings"
      "base_32x1|base_32x1/songmae/embeddings"
      "base_32x4|base_32x4/songmae/embeddings"
    )
    ;;
  *) echo "unknown MODEL_GROUP: $MODEL_GROUP" >&2; exit 2 ;;
esac

birds() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
print("\n".join(sorted({row["recording"]["bird_id"] for row in data["recordings"]})))
PY
}

failures=0
for dataset_row in "${DATASETS[@]}"; do
  IFS="|" read -r dataset annotations <<< "$dataset_row"
  while read -r bird; do
    out=$OUT_ROOT/$dataset/$bird
    if [[ -f "$out/metrics.csv" ]]; then
      echo "exists: dataset=$dataset bird=$bird"
      continue
    fi
    base=$EMBEDDING_ROOT/$dataset/$bird
    embeddings=()
    for model_row in "${MODELS[@]}"; do
      IFS="|" read -r name path <<< "$model_row"
      embeddings+=("$name=$base/$path")
    done
    echo ">> dataset=$dataset bird=$bird"
    if ! "$PYTHON_BIN" src/evals/syllable_kmeans.py "$out" "${embeddings[@]}" \
      --annotations "$annotations" --pca_components "$PCA_COMPONENTS" "${SILENCE_ARGS[@]}"; then
      echo "failed: dataset=$dataset bird=$bird" >&2
      failures=$((failures + 1))
    fi
  done < <(birds "$annotations")
done

echo "finished: failures=$failures"
((failures == 0))
