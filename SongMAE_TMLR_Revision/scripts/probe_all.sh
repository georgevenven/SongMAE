#!/usr/bin/env bash
# Corrected linear probes (recording folds, per-fold PCA, all ground-truth classes scored) for every condition at every
# layer in results/layer_selection.json (union over selection variants). Extracts one layer, probes, deletes embeddings.
# Usage (repo root): SPEC_ROOT=... WAV_ROOT=... PYTHON_BIN=... SongMAE_TMLR_Revision/scripts/probe_all.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON_BIN=${PYTHON_BIN:-python}
SPEC_ROOT=${SPEC_ROOT:-/media/george-vengrovski/disk2/specs}
WAV_ROOT=${WAV_ROOT:-/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae}
OUT_ROOT=${OUT_ROOT:-results/probes}
LAYERS_JSON=${LAYERS_JSON:-SongMAE_TMLR_Revision/results/layer_selection.json}
LOGREG_C=${LOGREG_C:-0.001}
CAPS=${CAPS:-"1 5 10 20 50 100"}
# Label-budget probes (Fig 6) for these; the finest model goes first because it creates the shared cap manifests.
CAP_CONDITIONS=${CAP_CONDITIONS:-"xcl_large_500k_p32x1_c005 xcl_large_500k_p32x4_c010 birdaves_biox_base birdaves_biox_base_speed0p5 birdaves_biox_base_speed0p25 hubert_base_ls960"}
DATASET_FILTER=${DATASET_FILTER:-}
BIRD_FILTER=${BIRD_FILTER:-}
# Default: every selected condition except the Table 2 Micro sweep runs (kNN only), finest SongMAE first.
CONDITIONS=${CONDITIONS:-$("$PYTHON_BIN" -c "
import json, sys
c = [k for k in json.load(open(sys.argv[1])) if '_100k_' not in k]
first = 'xcl_large_500k_p32x1_c005'
print(' '.join(([first] if first in c else []) + [k for k in c if k != first]))" "$LAYERS_JSON")}
COMMON=(--recording_mode events --num_timebins 720000 --balanced_events 3 --event_seed 42)
SELECTION=("${COMMON[@]}")
source SongMAE_TMLR_Revision/scripts/extract_lib.sh

for row in "canary|canary_5ms" "zf|zebra_finch_5ms" "bf|bengalese_finch_5ms"; do
  IFS='|' read -r dataset specs <<< "$row"
  selected "$dataset" "$DATASET_FILTER" || continue
  ann="files/annotation jsons/${dataset}_annotations.json"
  for bird in $(birds_of "$ann"); do
    selected "$bird" "$BIRD_FILTER" || continue
    manifest=$OUT_ROOT/manifests/$dataset/$bird.json
    for c in $CONDITIONS; do
      caps=""; selected "$c" "$CAP_CONDITIONS" && caps=$CAPS
      for layer in $(layers_of "$c"); do
        dir=$OUT_ROOT/$dataset/$bird/$c/layer_$layer
        todo=0; [[ -f $dir/metrics.json ]] || todo=1
        for cap in $caps; do [[ -f $dir/cap_$(printf %03d "$cap")/metrics.json ]] || todo=1; done
        ((todo)) || { echo "done: $dir"; continue; }
        mkdir -p "$dir"; rm -rf "$dir/embeddings" "$dir/embeddings.tmp" "$dir"/embeddings.*
        echo "probing: $dir"
        extract "$c" "$layer" "$SPEC_ROOT/$specs" "$ann" "$bird" "$dir/embeddings" > "$dir/extract.log" 2>&1 \
          || { echo "extraction failed: $dir" >&2; continue; }
        if [[ ! -f $dir/metrics.json ]]; then
          margs=(--manifest_in "$manifest"); [[ -f $manifest ]] || { mkdir -p "${manifest%/*}"; margs=(--manifest_out "$manifest"); }
          "$PYTHON_BIN" src/evals/syllable_classification.py --embeddings "$dir/embeddings" --annotations "$ann" \
            "${margs[@]}" --logreg_c "$LOGREG_C" > "$dir/metrics.tmp" 2> "$dir/probe.log" \
            && mv "$dir/metrics.tmp" "$dir/metrics.json" || echo "probe failed: $dir" >&2
        fi
        missing=""; for cap in $caps; do [[ -f $dir/cap_$(printf %03d "$cap")/metrics.json ]] || missing="$missing,$cap"; done
        if [[ -n $missing ]]; then  # all missing budgets in one process: per-fold PCA is fit once and shared
          "$PYTHON_BIN" src/evals/syllable_classification_capped.py --embeddings "$dir/embeddings" --annotations "$ann" \
            --label_caps "${missing#,}" --manifest_dir "$OUT_ROOT/manifests/$dataset/$bird" --out_dir "$dir" \
            --logreg_c "$LOGREG_C" > "$dir/caps.log" 2>&1 || echo "cap probes failed: $dir" >&2
        fi
        rm -rf "$dir/embeddings"
      done
    done
  done
done
echo "probes complete"
