#!/usr/bin/env bash
# Post-hoc rasterized PCA-8 K-means on the retained 50-bird UMAP embeddings.
set -u

cd "$(dirname "$0")/.."
PYTHON_BIN=${PYTHON_BIN:-/home/george-vengrovski/anaconda3/envs/mae/bin/python}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-/media/george-vengrovski/disk1/tinybird_umap_50birds_4models_250k_20260720}
OUT_ROOT=${OUT_ROOT:-results/syllable_kmeans_50birds_4models_raster_pca8_250k}

DATASETS=(
  "zf|files/annotation jsons/zf_annotations.json"
  "bf|files/annotation jsons/bf_annotations.json"
  "canary|files/annotation jsons/canary_annotations.json"
)

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
    echo ">> dataset=$dataset bird=$bird"
    if ! "$PYTHON_BIN" src/evals/syllable_kmeans.py "$out" \
      songmae_32x1="$base/songmae_32x1/songmae/embeddings" \
      songmae_32x4="$base/songmae_32x4/songmae/embeddings" \
      birdaves="$base/birdaves/aves/embeddings" \
      hubert="$base/hubert/hubert/embeddings" \
      --annotations "$annotations" --pca_components 8; then
      echo "failed: dataset=$dataset bird=$bird" >&2
      failures=$((failures + 1))
    fi
  done < <(birds "$annotations")
done

echo "finished: failures=$failures"
((failures == 0))
