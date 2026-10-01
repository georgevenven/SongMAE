#!/usr/bin/env bash
# Work desktop: kNN layer sweep for Table 2 seed runs as soon as they finish (Twins runs are pulled here), birds split
# over two streams, then re-select layers and refresh the staged tables. Replaces watch_seed_batch2.sh's Mac hand-off.
# Usage (repo root): SongMAE_TMLR_Revision/scripts/seed_knn_desktop.sh RUN [RUN ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
P=/home/george-vengrovski/anaconda3/envs/mae/bin/python
TWINS=george-vengrovski@163.41.128.66:Documents/SongMAE-reviews/runs
final=model_step_099999.pth

for run; do
  until [[ -f runs/$run/weights/$final ]]; do
    rsync -a --exclude imgs --exclude wandb --include weights/ --include "weights/$final" --exclude 'weights/*' "$TWINS/$run" runs/ 2>/dev/null || true
    [[ -f runs/$run/weights/$final ]] || sleep 300
  done
done
read -ra birds <<< "$(for d in canary zf bf; do $P -c "import json,sys;print(' '.join(sorted({r['recording']['bird_id'] for r in json.load(open(sys.argv[1]))['recordings']})))" "files/annotation jsons/${d}_annotations.json"; done | tr "\n" " ")"
for s in 0 1; do
  filter=$(for i in "${!birds[@]}"; do if (( i % 2 == s )); then printf '%s ' "${birds[$i]}"; fi; done)
  EXTRA_SONGMAE_RUNS="$*" MODEL_FILTER="$*" BIRD_FILTER=$filter PYTHON_BIN=$P \
    SongMAE_TMLR_Revision/scripts/knn_layer_sweep_review_baselines.sh > "logs/seed_knn_desktop_${1}_$s.log" 2>&1 &
done
wait
$P SongMAE_TMLR_Revision/scripts/select_layers.py
$P SongMAE_TMLR_Revision/scripts/export_results.py && $P SongMAE_TMLR_Revision/scripts/aggregate.py
echo "seed knn complete: $*"
