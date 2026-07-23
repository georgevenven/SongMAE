#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
WAIT_UNIT=${WAIT_UNIT:-avex-micro-base-500k.service}
UMAP_ROOT=/media/george-vengrovski/disk1/tinybird_umap_50birds_micro_base_500k_250k_20260722
KMEANS_ROOT=results/syllable_kmeans_50birds_micro_base_500k_raster_pca128_250k_including_silence

echo "waiting for $WAIT_UNIT"
while systemctl --user is-active --quiet "$WAIT_UNIT"; do
  sleep 30
done

if ! MODEL_GROUP=micro_base OUT_ROOT="$UMAP_ROOT" \
    bash shell/syllable_umap_50birds_4models.sh; then
  echo "UMAP extraction failed; K-means not started" >&2
  exit 1
fi

MODEL_GROUP=micro_base EMBEDDING_ROOT="$UMAP_ROOT" OUT_ROOT="$KMEANS_ROOT" \
  PCA_COMPONENTS=128 INCLUDE_SILENCE=1 bash shell/syllable_kmeans_50birds_4models.sh
