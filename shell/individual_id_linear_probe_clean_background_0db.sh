#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/george-vengrovski/Documents/projects/TinyBird
PYTHON=/home/george-vengrovski/anaconda3/envs/mae/bin/python
DATA="$ROOT/results/individual_id_linear_probe/multispecies_background_robustness"
WAV=/media/george-vengrovski/disk2/raw_data/individual_id_multispecies_background_robustness
SPEC=/media/george-vengrovski/disk2/specs/individual_id_multispecies_background_robustness_5ms
WORK=/media/george-vengrovski/disk2/individual_id_clean_background_0db
OUT="$ROOT/results/individual_id_linear_probe/clean_background_0db"

SPECIES=(american_robin bengalese_finch canary cassins_vireo chiffchaff european_starling ovenbird little_owl tree_pipit zebra_finch)
MODELS=(hubert_base_ls960 xcl_large_500k_p32x1_c005 birdaves_biox_base)
CONDITIONS=(clean test_0)

extract() {
  local species=$1 model=$2 condition=$3 destination=$4
  [[ -f "$destination/metadata.json" ]] && return
  mkdir -p "$destination"
  case "$model" in
    hubert_base_ls960)
      "$PYTHON" "$ROOT/src/external_models/hubert.py" \
        --spec_dir "$SPEC/$species/$condition" --wav_dir "$WAV/$species/$condition" \
        --annotation_file "$DATA/$species/${condition}_annotations.json" --out_dir "$destination" \
        --recording_mode events --model_name facebook/hubert-base-ls960 \
        --audio_sr 16000 --chunk_timebins 1000 --num_timebins 0
      ;;
    xcl_large_500k_p32x1_c005)
      "$PYTHON" -m src.core.extract_embedding \
        --spec_dir "$SPEC/$species/$condition" --run_dir "$ROOT/runs/$model" \
        --checkpoint model_step_499999.pth --out_dir "$destination" \
        --json_path "$DATA/$species/${condition}_annotations.json" \
        --recording_mode events --minimal --target_feature_type end_of_block --num_timebins 0
      ;;
    birdaves_biox_base)
      "$PYTHON" "$ROOT/src/external_models/aves.py" \
        --spec_dir "$SPEC/$species/$condition" --wav_dir "$WAV/$species/$condition" \
        --annotation_file "$DATA/$species/${condition}_annotations.json" --out_dir "$destination" \
        --recording_mode events --aves_model_path "$ROOT/files/birdaves-biox-base.torchaudio.pt" \
        --aves_config_path "$ROOT/files/birdaves-biox-base.torchaudio.model_config.json" \
        --model_name birdaves_biox_base --audio_sr 16000 --chunk_timebins 1000 --num_timebins 0
      ;;
  esac
}

for species in "${SPECIES[@]}"; do
  for model in "${MODELS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
      name=$condition
      [[ $condition == test_0 ]] && name=background_0db
      embeddings="$WORK/embeddings/$species/$model/$condition"
      destination="$OUT/$species/$model/$name"
      metrics="$destination/metrics.json"
      [[ -s "$metrics" ]] && continue
      extract "$species" "$model" "$condition" "$embeddings"
      mkdir -p "$destination"
      cp "$embeddings/metadata.json" "$destination/embedding_metadata.json"
      "$PYTHON" "$ROOT/src/evals/individual_id_classification.py" \
        --embeddings "$embeddings" \
        --annotations "$DATA/$species/${condition}_annotations.json" \
        --clip_map "$DATA/$species/clip_map.json" \
        --audio_scope events --folds 3 --pca_components 128 --logreg_c 0.001 \
        --manifest_out "$destination/manifest.json" > "$metrics.partial"
      mv "$metrics.partial" "$metrics"
    done
  done
done
