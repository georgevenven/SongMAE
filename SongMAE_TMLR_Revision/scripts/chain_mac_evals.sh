#!/usr/bin/env bash
# Work desktop: full evaluation of new baseline conditions on the Mac. kNN layer sweep there, layer selection here, then
# probes / k-means / kNN seeds there; pull everything back and refresh the staged tables.
# Usage (repo root): SongMAE_TMLR_Revision/scripts/chain_mac_evals.sh NAME CONDITION [CONDITION ...]  (Mac logs: logs/NAME_*.log)
set -euo pipefail
cd "$(dirname "$0")/../.."
P=/home/george-vengrovski/anaconda3/envs/mae/bin/python
MAC=macstudio R=Documents/SongMAE-reviews
name=$1; shift
ENV="cd $R && D=~/Documents/songmae_data && export SPEC_ROOT=\$D/specs WAV_ROOT=\$D/wavs PYTHON_BIN=~/miniforge3/envs/mae/bin/python"
pull() { rsync -a --ignore-existing "$@" 2>&1 | grep -v "^rsync: warning" || true; }
on_mac() { ssh -o BatchMode=yes $MAC "$ENV && nohup caffeinate -i bash -c '$2; echo $1 complete' > logs/${name}_$1.log 2>&1 < /dev/null &"; }
wait_mac() { until ssh -o BatchMode=yes $MAC "tail -1 $R/logs/${name}_$1.log" 2>/dev/null | grep -q "$1 complete"; do sleep 120; done; }

ssh -o BatchMode=yes $MAC "cd $R && git checkout -q -- SongMAE_TMLR_Revision/results && git pull -q"
on_mac sweep "MODEL_FILTER=\"$*\" SongMAE_TMLR_Revision/scripts/knn_layer_sweep_review_baselines.sh"; wait_mac sweep
pull "$MAC:$R/results/knn/review_baselines_all_layers/" results/knn/review_baselines_all_layers/
$P SongMAE_TMLR_Revision/scripts/select_layers.py
scp -q SongMAE_TMLR_Revision/results/layer_selection.json "$MAC:$R/SongMAE_TMLR_Revision/results/"
on_mac evals "export CONDITIONS=\"$*\"; SongMAE_TMLR_Revision/scripts/probe_all.sh; SongMAE_TMLR_Revision/scripts/kmeans_all.sh; SongMAE_TMLR_Revision/scripts/knn_seeds.sh"; wait_mac evals
pull --exclude manifests "$MAC:$R/results/probes/" results/probes/
for d in kmeans knn/seeds; do pull "$MAC:$R/results/$d/" "results/$d/"; done
$P SongMAE_TMLR_Revision/scripts/export_results.py && $P SongMAE_TMLR_Revision/scripts/aggregate.py
echo "chain complete: $*"
