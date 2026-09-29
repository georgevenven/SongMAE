#!/usr/bin/env bash
# Copy finished revision runs (final checkpoint + configs) to the Mac: Twins runs go through the work desktop's runs/.
# Usage (repo root, work desktop): SongMAE_TMLR_Revision/scripts/sync_runs_to_mac.sh RUN [RUN ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
TWINS=${TWINS:-george-vengrovski@163.41.128.66:Documents/SongMAE-reviews/runs}
MAC=${MAC:-macstudio:Documents/SongMAE-reviews/runs}
final=model_step_099999.pth
for run in "$@"; do
  [[ -f runs/$run/weights/$final ]] || rsync -a --exclude imgs --exclude wandb --include weights/ --include "weights/$final" \
    --exclude 'weights/*' "$TWINS/$run" runs/
  [[ -f runs/$run/weights/$final ]] || { echo "not finished: $run" >&2; exit 1; }
  rsync -a --exclude imgs --exclude wandb --include weights/ --include "weights/$final" --exclude 'weights/*' "runs/$run" "$MAC/"
  echo "synced: $run"
done
