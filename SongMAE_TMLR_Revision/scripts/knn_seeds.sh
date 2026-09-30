#!/usr/bin/env bash
# R1 evaluation-side spread: kNN purity with several reference/query sampling seeds, for every condition at every
# selected layer (same data selection as the layer sweep: first 200k timebins of events). One extraction per layer.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN=${PYTHON_BIN:-python}
SPEC_ROOT=${SPEC_ROOT:-/media/george-vengrovski/disk2/specs}
WAV_ROOT=${WAV_ROOT:-/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae}
OUT_ROOT=${OUT_ROOT:-results/knn/seeds}
LAYERS_JSON=${LAYERS_JSON:-SongMAE_TMLR_Revision/results/layer_selection.json}
SEEDS=${SEEDS:-"42 43 44 45 46"}
KNN_ARGS=${KNN_ARGS:-}
DATASET_FILTER=${DATASET_FILTER:-}
BIRD_FILTER=${BIRD_FILTER:-}
CONDITIONS=${CONDITIONS:-$("$PYTHON_BIN" -c "import json,sys;print(' '.join(k for k in json.load(open(sys.argv[1])) if '_100k_' not in k))" "$LAYERS_JSON")}
SELECTION=(--recording_mode events --num_timebins 200000)
source SongMAE_TMLR_Revision/scripts/extract_lib.sh
model_of() { case $1 in xcl_*|Xcl_*) echo songmae ;; birdaves_*) echo aves ;; hubert_*) echo hubert ;; *) echo "${1%%_*}" ;; esac; }

for row in "canary|canary_5ms" "zf|zebra_finch_5ms" "bf|bengalese_finch_5ms"; do
  IFS='|' read -r dataset specs <<< "$row"
  selected "$dataset" "$DATASET_FILTER" || continue
  ann="files/annotation jsons/${dataset}_annotations.json"
  for bird in $(birds_of "$ann"); do
    selected "$bird" "$BIRD_FILTER" || continue
    for c in $CONDITIONS; do
      for layer in $(layers_of "$c"); do
        dir=$OUT_ROOT/$dataset/$bird/$c/layer_$layer
        todo=""; for seed in $SEEDS; do [[ -f $dir/seed_$seed/summary.json ]] || todo="$todo $seed"; done
        [[ -z $todo ]] && { echo "done: $dir"; continue; }
        mkdir -p "$dir"; rm -rf "$dir"/embeddings*
        echo "knn:$todo $dir"
        extract "$c" "$layer" "$SPEC_ROOT/$specs" "$ann" "$bird" "$dir/embeddings" > "$dir/extract.log" 2>&1 \
          || { echo "extraction failed: $dir" >&2; continue; }
        for seed in $todo; do
          mkdir -p "$dir/seed_$seed"
          "$PYTHON_BIN" src/embeddings/syllable_knn.py --model "$(model_of "$c")" --playback_speed "$(speed_of "$c")" \
            --embedding_dir "$dir/embeddings" --spec_dir "$SPEC_ROOT/$specs" --annotation_file "$ann" --bird "$bird" \
            --out_dir "$dir/seed_$seed" --encoder_layer_idx "$layer" --num_timebins 200000 --k_values 1,5,10,50,100 \
            --seed "$seed" $KNN_ARGS > "$dir/seed_$seed/run.log" 2>&1 || echo "knn failed: $dir/seed_$seed" >&2
        done
        rm -rf "$dir/embeddings"
      done
    done
  done
done
echo "knn seeds complete"
