#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/george-vengrovski/Documents/projects/TinyBird
PYTHON=/home/george-vengrovski/anaconda3/envs/mae/bin/python
EMBEDDINGS=/media/george-vengrovski/disk2/individual_id_clean_background_0db/embeddings
DATA="$ROOT/results/individual_id_linear_probe/multispecies_background_robustness"
OUT="$ROOT/results/individual_id_open_set/clean"

SPECIES=(american_robin bengalese_finch canary cassins_vireo chiffchaff european_starling ovenbird little_owl tree_pipit zebra_finch)
MODELS=(hubert_base_ls960 birdaves_biox_base xcl_large_500k_p32x1_c005)

for species in "${SPECIES[@]}"; do
  for model in "${MODELS[@]}"; do
    output="$OUT/$species/$model/summary.json"
    [[ -s "$output" ]] && continue
    mkdir -p "$(dirname "$output")"
    "$PYTHON" "$ROOT/src/evals/individual_id_open_set.py" \
      --embeddings "$EMBEDDINGS/$species/$model/clean" \
      --clip_map "$DATA/$species/clip_map.json" \
      --species "$species" --model "$model" --output "$output"
  done
done
