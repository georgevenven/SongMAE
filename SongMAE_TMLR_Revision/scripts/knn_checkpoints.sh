#!/usr/bin/env bash
# Figure 4 re-run: kNN purity (corrected: same-recording neighbours excluded) across pretraining checkpoints for SongMAE
# Micro/Base/Large at 32x5 and 32x20 ms, last encoder layer, every bird. Same data selection as the layer sweep.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN=${PYTHON_BIN:-python}
SPEC_ROOT=${SPEC_ROOT:-/media/george-vengrovski/disk2/specs}
OUT_ROOT=${OUT_ROOT:-results/knn/checkpoints}
BIRD_FILTER=${BIRD_FILTER:-}
RUNS=${RUNS:-"xcl_micro_500k_p32x1_c005 xcl_base_500k_p32x1_c005 xcl_large_500k_p32x1_c005 xcl_micro_500k_p32x4_c010 xcl_base_500k_p32x4_c010 xcl_large_500k_p32x4_c010"}
STEPS=${STEPS:-"000000 010000 020000 050000 100000 499999"}
selected() { [[ -z "$2" || " $2 " == *" $1 "* ]]; }

for row in "canary|canary_5ms" "zf|zebra_finch_5ms" "bf|bengalese_finch_5ms"; do
  IFS='|' read -r dataset specs <<< "$row"
  ann="files/annotation jsons/${dataset}_annotations.json"
  for bird in $("$PYTHON_BIN" -c "import json,sys;print(' '.join(sorted({r['recording']['bird_id'] for r in json.load(open(sys.argv[1]))['recordings']})))" "$ann"); do
    selected "$bird" "$BIRD_FILTER" || continue
    for run in $RUNS; do
      layer=$(( $("$PYTHON_BIN" -c "import json;print(json.load(open('runs/$run/model.json'))['enc_n_layer'])") - 1 ))
      for step in $STEPS; do
        dir=$OUT_ROOT/$dataset/$bird/${run}_step_$step/layer_$layer
        [[ -f $dir/summary.json ]] && { echo "done: $dir"; continue; }
        mkdir -p "$dir"; rm -rf "$dir"/embeddings*
        echo "knn: $dir"
        "$PYTHON_BIN" -m src.core.extract_embedding --spec_dir "$SPEC_ROOT/$specs" --run_dir "runs/$run" \
          --checkpoint "model_step_$step.pth" --out_dir "$dir/embeddings" --json_path "$ann" --bird "$bird" \
          --recording_mode events --minimal --target_feature_type end_of_block --num_timebins 200000 \
          --encoder_layer_idx "$layer" > "$dir/extract.log" 2>&1 || { echo "extraction failed: $dir" >&2; continue; }
        "$PYTHON_BIN" src/embeddings/syllable_knn.py --model songmae --embedding_dir "$dir/embeddings" \
          --spec_dir "$SPEC_ROOT/$specs" --annotation_file "$ann" --bird "$bird" --out_dir "$dir" \
          --encoder_layer_idx "$layer" --num_timebins 200000 --k_values 1,5,10,50,100 > "$dir/run.log" 2>&1 \
          || echo "knn failed: $dir" >&2
        rm -rf "$dir/embeddings"
      done
    done
  done
done
echo "checkpoint knn complete"
