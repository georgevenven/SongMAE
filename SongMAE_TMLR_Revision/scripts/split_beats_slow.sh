#!/usr/bin/env bash
# Work desktop: BEATs 1/8, 1/16, 1/32 probes / k-means / kNN seeds split across machines (replaces the second half of
# chain_beats_slow.sh). Mac: canary. Desktop (RTX 4090): zebra and Bengalese finch in two streams. Then pull and refresh.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=/home/george-vengrovski/anaconda3/envs/mae/bin/python
MAC=macstudio R=Documents/SongMAE-reviews
export CONDITIONS="beats_iter3_plus_as2m_speed0p125 beats_iter3_plus_as2m_speed0p0625 beats_iter3_plus_as2m_speed0p03125"
birds() { $P -c "import json,sys;print(' '.join(sorted({r['recording']['bird_id'] for r in json.load(open(sys.argv[1]))['recordings']})))" "files/annotation jsons/$1_annotations.json"; }
evals='SongMAE_TMLR_Revision/scripts/probe_all.sh; SongMAE_TMLR_Revision/scripts/kmeans_all.sh; SongMAE_TMLR_Revision/scripts/knn_seeds.sh'

ssh -o BatchMode=yes $MAC "cd $R && D=~/Documents/songmae_data && export SPEC_ROOT=\$D/specs WAV_ROOT=\$D/wavs PYTHON_BIN=~/miniforge3/envs/mae/bin/python CONDITIONS='$CONDITIONS' BIRD_FILTER='$(birds canary)' && nohup caffeinate -i bash -c '$evals; echo beats slow evals complete' > logs/beats_slow_evals.log 2>&1 < /dev/null &"
read -ra local_birds <<< "$(birds zf) $(birds bf)"
for s in 0 1; do
  filter=$(for i in "${!local_birds[@]}"; do if (( i % 2 == s )); then printf '%s ' "${local_birds[$i]}"; fi; done)
  PYTHON_BIN=$P BIRD_FILTER=$filter bash -c "$evals" > logs/beats_slow_evals_$s.log 2>&1 &
done
wait
until ssh -o BatchMode=yes $MAC "tail -1 $R/logs/beats_slow_evals.log" 2>/dev/null | grep -q "beats slow evals complete"; do sleep 120; done
rsync -a --ignore-existing --exclude manifests "$MAC:$R/results/probes/" results/probes/ 2>&1 | grep -v "^rsync: warning" || true
for d in kmeans knn/seeds; do rsync -a --ignore-existing "$MAC:$R/results/$d/" "results/$d/" 2>&1 | grep -v "^rsync: warning" || true; done
$P SongMAE_TMLR_Revision/scripts/export_results.py && $P SongMAE_TMLR_Revision/scripts/aggregate.py
echo "split complete"
