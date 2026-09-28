#!/usr/bin/env bash
# Copy syllable-eval inputs from the work desktop to the Mac Studio (resumable). Run from the repo root.
# Mac layout: data in ~/Documents/songmae_data/{specs,wavs}; files/ and runs/ inside ~/Documents/SongMAE-reviews.
set -euo pipefail
cd "$(dirname "$0")/../.."

HOST=${HOST:-macstudio}
REPO=Documents/SongMAE-reviews
DATA=Documents/songmae_data
SPECS=/media/george-vengrovski/disk2/specs
WAVS=/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae
RUNS="xcl_large_500k_p32x1_c005 xcl_large_500k_p32x4_c010 xcl_large_500k_p32x4_c0025
  xcl_base_500k_p32x1_c005 xcl_base_500k_p32x4_c010 xcl_micro_500k_p32x1_c005 xcl_micro_500k_p32x4_c010
  Xcl_micro_100k_p128x1_default Xcl_micro_100k_p16x1_default Xcl_micro_100k_p32x1_c0025 Xcl_micro_100k_p32x1_c005
  Xcl_micro_100k_p32x1_c010 xcl_micro_100k_p32x1_random xcl_micro_100k_p32x4_c0025 xcl_micro_100k_p32x4_c005
  xcl_micro_100k_p32x4_c010 Xcl_micro_100k_p4x4_default"

sync() { rsync -a --partial "$@"; }

ssh "$HOST" "mkdir -p $REPO/runs $REPO/files/annotation\ jsons $DATA/specs $DATA/wavs"
sync files/birdaves-biox-base.torchaudio.pt files/birdaves-biox-base.torchaudio.model_config.json files/review_baselines "$HOST:$REPO/files/"
sync "files/annotation jsons/"{zf,bf,canary}_annotations.json "$HOST:$REPO/files/annotation jsons/"  # Mac rsync 2.6.9 takes remote paths literally
for run in $RUNS; do
  last=$(ls "runs/$run/weights" | sort -V | tail -1)
  sync --exclude imgs --exclude wandb --include weights/ --include "weights/$last" --exclude 'weights/*' "runs/$run" "$HOST:$REPO/runs/"
done
sync "$SPECS"/{zebra_finch_5ms,bengalese_finch_5ms,canary_5ms} "$HOST:$DATA/specs/"
sync "$WAVS"/ "$HOST:$DATA/wavs/"
echo "sync complete"
