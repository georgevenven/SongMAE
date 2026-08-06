#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/george-vengrovski/Documents/projects/TinyBird
PYTHON=/home/george-vengrovski/anaconda3/envs/mae/bin/python
WAV=/media/george-vengrovski/disk2/raw_data/individual_id_multispecies_background_robustness
SPEC=/media/george-vengrovski/disk2/specs/individual_id_multispecies_background_robustness_5ms
EMBEDDINGS=/media/george-vengrovski/disk2/individual_id_clean_background_0db/embeddings
DATA="$ROOT/results/individual_id_linear_probe/multispecies_background_robustness"
OUT="$ROOT/results/individual_id_open_set/background_robustness"
CONDITIONS=(clean train_aug test_p10 test_0 test_m10)
SPECIES=(american_robin bengalese_finch canary cassins_vireo chiffchaff european_starling ovenbird little_owl tree_pipit zebra_finch)
MODELS=(hubert_base_ls960 birdaves_biox_base xcl_large_500k_p32x1_c005)

extract() {
  local model=$1 species=$2 condition=$3 destination=$4
  [[ -f "$destination/metadata.json" ]] && return
  case "$model" in
    hubert_base_ls960)
      "$PYTHON" "$ROOT/src/external_models/hubert.py" \
        --spec_dir "$SPEC/$species/$condition" --wav_dir "$WAV/$species/$condition" \
        --annotation_file "$DATA/$species/${condition}_annotations.json" --out_dir "$destination" \
        --recording_mode events --model_name facebook/hubert-base-ls960 \
        --audio_sr 16000 --chunk_timebins 1000 --num_timebins 0
      ;;
    birdaves_biox_base)
      "$PYTHON" "$ROOT/src/external_models/aves.py" \
        --spec_dir "$SPEC/$species/$condition" --wav_dir "$WAV/$species/$condition" \
        --annotation_file "$DATA/$species/${condition}_annotations.json" --out_dir "$destination" \
        --recording_mode events --aves_model_path "$ROOT/files/birdaves-biox-base.torchaudio.pt" \
        --aves_config_path "$ROOT/files/birdaves-biox-base.torchaudio.model_config.json" \
        --model_name birdaves_biox_base --audio_sr 16000 --chunk_timebins 1000 --num_timebins 0
      ;;
    xcl_large_500k_p32x1_c005)
      "$PYTHON" -m src.core.extract_embedding \
        --spec_dir "$SPEC/$species/$condition" --run_dir "$ROOT/runs/$model" \
        --checkpoint model_step_499999.pth --out_dir "$destination" \
        --json_path "$DATA/$species/${condition}_annotations.json" \
        --recording_mode events --minimal --target_feature_type end_of_block --num_timebins 0
      ;;
    *) exit 2 ;;
  esac
}

for species in "${SPECIES[@]}"; do
  for model in "${MODELS[@]}"; do
    embeddings="$EMBEDDINGS/$species/$model"
    output="$OUT/$species/$model/summary.json"
    [[ -s "$output" ]] && continue
    for condition in "${CONDITIONS[@]}"; do
      extract "$model" "$species" "$condition" "$embeddings/$condition"
    done
    "$PYTHON" "$ROOT/src/evals/individual_id_open_set_background.py" \
      --embedding_root "$embeddings" --clip_map "$DATA/$species/clip_map.json" \
      --species "$species" --model "$model" --output "$output"
  done
done
