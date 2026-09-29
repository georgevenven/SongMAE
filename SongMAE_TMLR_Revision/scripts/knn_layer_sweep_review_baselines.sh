#!/usr/bin/env bash
# kNN purity at every encoder layer (review baselines, BirdAVES, HuBERT, SongMAE runs), for protocol layer selection.
# Model field: beats | birdmae | aves | hubert | songmae:<run dir name>.
# One all-layer extraction per bird/condition (same data selection as the other kNN runs), then kNN per layer.
# Usage (repo root): SPEC_ROOT=... WAV_ROOT=... PYTHON_BIN=... SongMAE_TMLR_Revision/scripts/knn_layer_sweep_review_baselines.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON_BIN=${PYTHON_BIN:-python}
SPEC_ROOT=${SPEC_ROOT:-/media/george-vengrovski/disk2/specs}
WAV_ROOT=${WAV_ROOT:-/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae}
OUT_ROOT=${OUT_ROOT:-results/knn/review_baselines_all_layers}
NUM_TIMEBINS=${NUM_TIMEBINS:-200000}
DATASET_FILTER=${DATASET_FILTER:-}
MODEL_FILTER=${MODEL_FILTER:-}
BIRD_FILTER=${BIRD_FILTER:-}
DATASETS=(
  "canary|files/annotation jsons/canary_annotations.json|$SPEC_ROOT/canary_5ms"
  "zf|files/annotation jsons/zf_annotations.json|$SPEC_ROOT/zebra_finch_5ms"
  "bf|files/annotation jsons/bf_annotations.json|$SPEC_ROOT/bengalese_finch_5ms"
)
MODELS=(
  "beats_iter3_plus_as2m|beats|1.0"
  "beats_iter3_plus_as2m_speed0p5|beats|0.5"
  "beats_iter3_plus_as2m_speed0p25|beats|0.25"
  "birdmae_base_speed1|birdmae|1.0"
  "birdmae_base_speed0p5|birdmae|0.5"
  "birdmae_base_speed0p25|birdmae|0.25"
  "birdmae_base_speed0p125|birdmae|0.125"
  "birdmae_base_speed0p0625|birdmae|0.0625"
  "birdmae_base_speed0p03125|birdmae|0.03125"
  "birdaves_biox_base|aves|1.0"
  "birdaves_biox_base_speed0p5|aves|0.5"
  "birdaves_biox_base_speed0p25|aves|0.25"
  "hubert_base_ls960|hubert|1.0"
  "xcl_large_500k_p32x1_c005|songmae:xcl_large_500k_p32x1_c005|1.0"
  "xcl_large_500k_p32x4_c010|songmae:xcl_large_500k_p32x4_c010|1.0"
  "xcl_large_500k_p32x4_c0025|songmae:xcl_large_500k_p32x4_c0025|1.0"
  "xcl_base_500k_p32x1_c005|songmae:xcl_base_500k_p32x1_c005|1.0"
  "xcl_base_500k_p32x4_c010|songmae:xcl_base_500k_p32x4_c010|1.0"
  "xcl_micro_500k_p32x1_c005|songmae:xcl_micro_500k_p32x1_c005|1.0"
  "xcl_micro_500k_p32x4_c010|songmae:xcl_micro_500k_p32x4_c010|1.0"
)
# Table 2 Micro sweep runs (100k steps): scored at every layer like the others.
for run in Xcl_micro_100k_p128x1_default Xcl_micro_100k_p16x1_default Xcl_micro_100k_p32x1_c0025 Xcl_micro_100k_p32x1_c005 \
  Xcl_micro_100k_p32x1_c010 xcl_micro_100k_p32x1_random xcl_micro_100k_p32x4_c0025 xcl_micro_100k_p32x4_c005 \
  xcl_micro_100k_p32x4_c010 Xcl_micro_100k_p4x4_default ${EXTRA_SONGMAE_RUNS:-}; do
  MODELS+=("$run|songmae:$run|1.0")
done
selected() { [[ -z "$2" || " $2 " == *" $1 "* ]]; }

for dataset_row in "${DATASETS[@]}"; do
  IFS='|' read -r dataset annotations specs <<< "$dataset_row"
  selected "$dataset" "$DATASET_FILTER" || continue
  for bird in $("$PYTHON_BIN" -c "import json,sys;print(' '.join(sorted({r['recording']['bird_id'] for r in json.load(open(sys.argv[1]))['recordings']})))" "$annotations"); do
    selected "$bird" "$BIRD_FILTER" || continue
    for model_row in "${MODELS[@]}"; do
      IFS='|' read -r name model speed <<< "$model_row"
      selected "$name" "$MODEL_FILTER" || continue
      out=$OUT_ROOT/$dataset/$bird/$name
      layers=12
      [[ $model == songmae:* ]] && layers=$("$PYTHON_BIN" -c "import json;print(json.load(open('runs/${model#songmae:}/model.json'))['enc_n_layer'])")
      [[ -f $out/layer_$((layers - 1))/summary.json ]] && { echo "done: $out"; continue; }
      embeddings=$out/embeddings
      rm -rf "$embeddings" "$embeddings.tmp"
      mkdir -p "$out"
      echo "extracting: $out"
      common=(--spec_dir "$specs" --wav_dir "$WAV_ROOT" --annotation_file "$annotations" --bird "$bird"
        --recording_mode events --out_dir "$embeddings" --num_timebins "$NUM_TIMEBINS" --all_layers)
      if [[ $model == songmae:* ]]; then
        run=runs/${model#songmae:}
        "$PYTHON_BIN" -m src.core.extract_embedding --spec_dir "$specs" --run_dir "$run" \
          --checkpoint "$(ls "$run/weights" | sort -V | tail -1)" --out_dir "$embeddings" --json_path "$annotations" \
          --bird "$bird" --recording_mode events --minimal --target_feature_type end_of_block \
          --num_timebins "$NUM_TIMEBINS" --all_layers > "$out/extract.log" 2>&1
      elif [[ $model == hubert ]]; then
        "$PYTHON_BIN" src/external_models/hubert.py --model_name facebook/hubert-base-ls960 --audio_sr 16000 \
          --chunk_timebins 1000 "${common[@]}" > "$out/extract.log" 2>&1
      elif [[ $model == aves ]]; then  # same 5 s model input as the other baselines: 5 s x speed of original audio
        "$PYTHON_BIN" src/external_models/aves.py --speed "$speed" --chunk_timebins "$("$PYTHON_BIN" -c "print(int(1000 * $speed))")" \
          --aves_model_path files/birdaves-biox-base.torchaudio.pt --aves_config_path files/birdaves-biox-base.torchaudio.model_config.json \
          "${common[@]}" > "$out/extract.log" 2>&1
      else
        "$PYTHON_BIN" src/external_models/review_baselines.py --model "$model" --speed "$speed" --chunk_timebins 1000 \
          "${common[@]}" > "$out/extract.log" 2>&1
      fi
      for layer in $(seq 0 $((layers - 1))); do
        mkdir -p "$out/layer_$layer"
        "$PYTHON_BIN" src/embeddings/syllable_knn.py --model "${model%%:*}" --playback_speed "$speed" --embedding_dir "$embeddings" \
          --spec_dir "$specs" --annotation_file "$annotations" --bird "$bird" --out_dir "$out/layer_$layer" \
          --encoder_layer_idx "$layer" --num_timebins "$NUM_TIMEBINS" --k_values 1,5,10,50,100 > "$out/layer_$layer/run.log" 2>&1
      done
      rm -rf "$embeddings"
    done
  done
done
echo "sweep complete"
