#!/usr/bin/env bash
set -euo pipefail

SPEC_ROOT="/media/george-vengrovski/disk2/specs"
ANN_ROOT="files/annotation jsons"
WAV_ROOT="/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae"
MODEL_DIR="runs/xcl_micro_500k_p4x4_default"
MODELS="${MODELS:-songmae,aves,hubert}"
NUM_TIMEBINS="${NUM_TIMEBINS:-0}"
MAX_POINTS="${MAX_POINTS:-100000}"

# spec_dir annotation_file bird1 bird2 bird3
SPECIES=(
  "zebra_finch_5ms zf_annotations.json Y389 S389 Y453"
  "bengalese_finch_5ms bf_annotations.json bird1 bird3 bird4"
  "canary_5ms canary_annotations.json llb11_annot llb3_annot llb16_annot"
)

for entry in "${SPECIES[@]}"; do
  read -r spec_dir anno_file bird1 bird2 bird3 <<< "$entry"
  for bird in "$bird1" "$bird2" "$bird3"; do
    out_dir="results/syllable_umap_${spec_dir}_${bird}"
    echo ">> ${spec_dir} | ${bird} -> ${out_dir}"
    python src/embeddings/syllable_umap.py \
      --spec_dir "${SPEC_ROOT}/${spec_dir}" \
      --annotation_file "${ANN_ROOT}/${anno_file}" \
      --out_dir "${out_dir}" \
      --models "${MODELS}" \
      --wav_dir "${WAV_ROOT}" \
      --bird "${bird}" \
      --num_timebins ${NUM_TIMEBINS} \
      --max_points ${MAX_POINTS} \
      --songmae_run_dir "${MODEL_DIR}"
  done
done
