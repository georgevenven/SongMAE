#!/usr/bin/env bash
# Table 4 re-run (P4): K-means (K = annotated classes + silence) on PCA-128 embeddings of up to ~20 min per bird, for
# every condition at every selected layer (union over selection variants). Extracts one layer, clusters, deletes it.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN=${PYTHON_BIN:-python}
SPEC_ROOT=${SPEC_ROOT:-/media/george-vengrovski/disk2/specs}
WAV_ROOT=${WAV_ROOT:-/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae}
OUT_ROOT=${OUT_ROOT:-results/kmeans}
LAYERS_JSON=${LAYERS_JSON:-SongMAE_TMLR_Revision/results/layer_selection.json}
DATASET_FILTER=${DATASET_FILTER:-}
BIRD_FILTER=${BIRD_FILTER:-}
CONDITIONS=${CONDITIONS:-$("$PYTHON_BIN" -c "import json,sys;print(' '.join(k for k in json.load(open(sys.argv[1])) if '_100k_' not in k))" "$LAYERS_JSON")}
SELECTION=(--recording_mode events --num_timebins 250000)
source SongMAE_TMLR_Revision/scripts/extract_lib.sh

for row in "canary|canary_5ms" "zf|zebra_finch_5ms" "bf|bengalese_finch_5ms"; do
  IFS='|' read -r dataset specs <<< "$row"
  selected "$dataset" "$DATASET_FILTER" || continue
  ann="files/annotation jsons/${dataset}_annotations.json"
  for bird in $(birds_of "$ann"); do
    selected "$bird" "$BIRD_FILTER" || continue
    for c in $CONDITIONS; do
      for layer in $(layers_of "$c"); do
        dir=$OUT_ROOT/$dataset/$bird/$c/layer_$layer
        [[ -f $dir/metrics.csv ]] && { echo "done: $dir"; continue; }
        mkdir -p "$dir"; rm -rf "$dir"/embeddings*
        echo "clustering: $dir"
        extract "$c" "$layer" "$SPEC_ROOT/$specs" "$ann" "$bird" "$dir/embeddings" > "$dir/extract.log" 2>&1 \
          || { echo "extraction failed: $dir" >&2; continue; }
        "$PYTHON_BIN" src/evals/syllable_kmeans.py "$dir" "$c=$dir/embeddings" --annotations "$ann" \
          --pca_components 128 --include_silence > "$dir/kmeans.log" 2>&1 || echo "kmeans failed: $dir" >&2
        rm -rf "$dir/embeddings"
      done
    done
  done
done
echo "kmeans complete"
