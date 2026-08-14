#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/george-vengrovski/Documents/projects/TinyBird
PYTHON=/home/george-vengrovski/anaconda3/envs/mae/bin/python
PAPER="$ROOT/Individual_Id_paper_materials"
DATA="$ROOT/results/individual_id/individual_id_linear_probe/multispecies_background_robustness"
CLEAN_WAV=/media/george-vengrovski/disk2/raw_data/individual_id_multispecies_background_robustness
CLEAN_SPEC=/media/george-vengrovski/disk2/specs/individual_id_multispecies_background_robustness_5ms
PINK=/media/george-vengrovski/disk2/individual_id_pink_noise_same_condition_0db
WORK=/media/george-vengrovski/disk2/individual_id_all_layers_10_species
OUT="$PAPER/results"
K_VALUES="${K_VALUES:-1,5,10,50,100}"

SPECIES=(american_robin bengalese_finch canary cassins_vireo chiffchaff european_starling ovenbird little_owl tree_pipit zebra_finch)
MODELS=(xcl_large_500k_p32x4_c010 xcl_large_500k_p32x1_c005 hubert_base_ls960 birdaves_biox_base)
CONDITIONS=(clean pink_0db)
METHODS=(centroid token)
LAYERS=(0 1 2 3 4 5 6 7 8 9 10 11)

log_failure() {
  local stage=$1 species=$2 model=$3 condition=$4 layer=$5 method=$6
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date -Is)" "$stage" "$species" "$model" "$condition" "$layer" "$method" \
    >> "$FAILURES" || true
}

prepare_pink() {
  local species=$1
  local destination="$PINK/$species"
  [[ -f "$destination/.complete" ]] && return
  "$PYTHON" "$ROOT/src/dataset_utils/individual_id/pink_noise_augmentation.py" \
    --audio_dir "$CLEAN_WAV/$species/clean" \
    --audio_output_dir "$destination/wav" \
    --output_dir "$destination/spec" \
    --stats_dir "$CLEAN_SPEC/$species/clean" \
    --snr_db 0 || return 1
  touch "$destination/.complete" || return 1
}

extract() {
  local species=$1 model=$2 condition=$3 destination=$4
  local annotation="$DATA/$species/clean_annotations.json"
  local spec_dir="$CLEAN_SPEC/$species/clean"
  local wav_dir="$CLEAN_WAV/$species/clean"
  [[ $condition == pink_0db ]] && spec_dir="$PINK/$species/spec"
  [[ $condition == pink_0db ]] && wav_dir="$PINK/$species/wav"
  [[ -f "$destination/metadata.json" ]] && return

  case "$model" in
    xcl_large_500k_p32x4_c010|xcl_large_500k_p32x1_c005)
      "$PYTHON" -m src.core.extract_embedding \
        --spec_dir "$spec_dir" --run_dir "$ROOT/runs/$model" \
        --checkpoint model_step_499999.pth --out_dir "$destination" \
        --json_path "$annotation" --recording_mode full_recordings \
        --max_segment_timebins 1000 --per_segment_normalize \
        --all_layers --minimal --target_feature_type end_of_block --num_timebins 0 || return 1
      ;;
    hubert_base_ls960)
      "$PYTHON" "$ROOT/src/external_models/hubert.py" \
        --spec_dir "$spec_dir" --wav_dir "$wav_dir" \
        --annotation_file "$annotation" --out_dir "$destination" \
        --recording_mode full_recordings --model_name facebook/hubert-base-ls960 \
        --audio_sr 16000 --chunk_timebins 1000 --all_layers --num_timebins 0 || return 1
      ;;
    birdaves_biox_base)
      "$PYTHON" "$ROOT/src/external_models/aves.py" \
        --spec_dir "$spec_dir" --wav_dir "$wav_dir" \
        --annotation_file "$annotation" --out_dir "$destination" \
        --recording_mode full_recordings \
        --aves_model_path "$ROOT/files/birdaves-biox-base.torchaudio.pt" \
        --aves_config_path "$ROOT/files/birdaves-biox-base.torchaudio.model_config.json" \
        --model_name birdaves_biox_base --audio_sr 16000 \
        --chunk_timebins 1000 --all_layers --num_timebins 0 || return 1
      ;;
    *)
      return 1
  esac
}

FAILURES="$OUT/failures.tsv"
mkdir -p "$OUT"
if [[ ! -f "$FAILURES" ]]; then
  printf "timestamp\tstage\tspecies\tmodel\tcondition\tlayer\tmethod\n" > "$FAILURES"
fi

for species in "${SPECIES[@]}"; do
  pink_ready=1
  if ! prepare_pink "$species"; then
    pink_ready=0
    log_failure pink_preparation "$species" - pink_0db - -
  fi
  manifest="$OUT/manifests/$species.json"
  for model in "${MODELS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
      if [[ $condition == pink_0db && $pink_ready -eq 0 ]]; then
        log_failure skipped_missing_pink "$species" "$model" "$condition" - -
        continue
      fi
      embeddings="$WORK/embeddings/$species/$model/$condition"
      if ! extract "$species" "$model" "$condition" "$embeddings"; then
        log_failure embedding_extraction "$species" "$model" "$condition" - -
        continue
      fi
      for layer in "${LAYERS[@]}"; do
        layer_name=$(printf "layer_%02d" "$layer")
        for method in "${METHODS[@]}"; do
          destination="$OUT/logistic/$method/$condition/$species/$model/$layer_name"
          metrics="$destination/metrics.json"
          [[ -s "$metrics" && -s "$manifest" ]] && continue
          if ! mkdir -p "$destination" "$(dirname "$manifest")"; then
            log_failure logistic_mkdir "$species" "$model" "$condition" "$layer" "$method"
            continue
          fi
          manifest_args=(--manifest_out "$manifest")
          [[ -s "$manifest" ]] && manifest_args=(--manifest_in "$manifest")
          if ! cp "$embeddings/metadata.json" "$destination/embedding_metadata.json"; then
            log_failure logistic_metadata "$species" "$model" "$condition" "$layer" "$method"
            continue
          fi
          if ! "$PYTHON" "$ROOT/src/evals/individual_id_classification.py" \
            --embeddings "$embeddings" --layer "$layer" \
            --annotations "$DATA/$species/clean_annotations.json" \
            --condition "$condition" --method "$method" \
            --audio_scope song_and_non_song --folds 3 \
            --pca_components 128 --logreg_c 0.001 \
            "${manifest_args[@]}" > "$metrics.partial"; then
            rm -f "$metrics.partial"
            log_failure logistic "$species" "$model" "$condition" "$layer" "$method"
            continue
          fi
          if ! mv "$metrics.partial" "$metrics"; then
            log_failure logistic_finalize "$species" "$model" "$condition" "$layer" "$method"
            continue
          fi
        done

        destination="$OUT/knn/$condition/$species/$model/$layer_name"
        [[ -s "$destination/summary.json" ]] && continue
        if [[ ! -s "$manifest" ]]; then
          log_failure knn_missing_manifest "$species" "$model" "$condition" "$layer" dn4
          continue
        fi
        if ! mkdir -p "$destination"; then
          log_failure knn_mkdir "$species" "$model" "$condition" "$layer" dn4
          continue
        fi
        if ! cp "$embeddings/metadata.json" "$destination/embedding_metadata.json"; then
          log_failure knn_metadata "$species" "$model" "$condition" "$layer" dn4
          continue
        fi
        if ! "$PYTHON" "$ROOT/src/evals/individual_id_knn_purity.py" \
          --embeddings "$embeddings" --layer "$layer" \
          --annotations "$DATA/$species/clean_annotations.json" \
          --out_dir "$destination" --condition "$condition" \
          --audio_scope song_and_non_song --folds 3 \
          --pca_components 128 --k_values "$K_VALUES" \
          --manifest_in "$manifest" > "$destination/stdout.json.partial"; then
          rm -f "$destination/stdout.json.partial"
          log_failure knn "$species" "$model" "$condition" "$layer" dn4
          continue
        fi
        if ! mv "$destination/stdout.json.partial" "$destination/stdout.json"; then
          log_failure knn_finalize "$species" "$model" "$condition" "$layer" dn4
          continue
        fi
      done
    done
  done
done
