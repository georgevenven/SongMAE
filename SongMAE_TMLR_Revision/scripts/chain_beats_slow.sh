#!/usr/bin/env bash
# Work desktop: after the Mac's BEATs 1/8, 1/16, 1/32 layer sweep, pull kNN results (plus the Mac's finished k-means and
# kNN-seed outputs), re-select layers with the full data, run probes / k-means / kNN seeds for the new conditions on the
# Mac, pull everything back and refresh the staged tables.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=/home/george-vengrovski/anaconda3/envs/mae/bin/python
MAC=macstudio R=Documents/SongMAE-reviews
NEW="beats_iter3_plus_as2m_speed0p125 beats_iter3_plus_as2m_speed0p0625 beats_iter3_plus_as2m_speed0p03125"
pull() { rsync -a --ignore-existing "$MAC:$R/results/$1/" "results/$1/" 2>&1 | grep -v "^rsync: warning" || true; }

until ssh -o BatchMode=yes $MAC "tail -1 $R/logs/knn_layer_sweep_beats_slow.log" 2>/dev/null | grep -q "sweep complete"; do sleep 120; done
pull knn/review_baselines_all_layers; pull kmeans; pull knn/seeds
$P SongMAE_TMLR_Revision/scripts/select_layers.py
scp -q SongMAE_TMLR_Revision/results/layer_selection.json "$MAC:$R/SongMAE_TMLR_Revision/results/"
ssh -o BatchMode=yes $MAC "cd $R && D=~/Documents/songmae_data && export SPEC_ROOT=\$D/specs WAV_ROOT=\$D/wavs PYTHON_BIN=~/miniforge3/envs/mae/bin/python CONDITIONS='$NEW' && nohup caffeinate -i bash -c 'SongMAE_TMLR_Revision/scripts/probe_all.sh; SongMAE_TMLR_Revision/scripts/kmeans_all.sh; SongMAE_TMLR_Revision/scripts/knn_seeds.sh; echo beats slow evals complete' > logs/beats_slow_evals.log 2>&1 < /dev/null &"
until ssh -o BatchMode=yes $MAC "tail -1 $R/logs/beats_slow_evals.log" 2>/dev/null | grep -q "beats slow evals complete"; do sleep 120; done
rsync -a --ignore-existing --exclude manifests "$MAC:$R/results/probes/" results/probes/ 2>&1 | grep -v "^rsync: warning" || true
pull kmeans; pull knn/seeds
$P SongMAE_TMLR_Revision/scripts/export_results.py && $P SongMAE_TMLR_Revision/scripts/aggregate.py
echo "chain complete"
