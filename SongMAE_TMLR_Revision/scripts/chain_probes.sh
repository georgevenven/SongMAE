#!/usr/bin/env bash
# Run on the work desktop. Waits for the Mac kNN sweeps (bf; zf in two batches; canary was done on the desktop), merges
# the Mac's kNN results, selects layers, then probes: SongMAE/BirdAVES/HuBERT here, BEATs/Bird-MAE on the Mac. Mac probe
# results are pulled back at the end.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=/home/george-vengrovski/anaconda3/envs/mae/bin/python
MAC=macstudio MAC_REPO=Documents/SongMAE-reviews

for log in knn_layer_sweep_bf.log knn_layer_sweep_zf_mac.log knn_layer_sweep_zf_mac2.log; do
  until ssh -o BatchMode=yes $MAC "tail -1 $MAC_REPO/logs/$log" 2>/dev/null | grep -q "sweep complete"; do sleep 120; done
done
for dataset in bf zf; do
  rsync -a "$MAC:$MAC_REPO/results/knn/review_baselines_all_layers/$dataset/" "results/knn/review_baselines_all_layers/$dataset/"
done
$P SongMAE_TMLR_Revision/scripts/select_layers.py
scp -q SongMAE_TMLR_Revision/results/layer_selection.json "$MAC:$MAC_REPO/SongMAE_TMLR_Revision/results/"

all=$($P -c "import json;print(' '.join(k for k in json.load(open('SongMAE_TMLR_Revision/results/layer_selection.json')) if '_100k_' not in k))")
mac_conditions=$(tr ' ' '\n' <<< "$all" | grep -E '^(beats|birdmae)_' | tr '\n' ' ')
here_conditions="xcl_large_500k_p32x1_c005 $(tr ' ' '\n' <<< "$all" | grep -vE '^(beats|birdmae)_|^xcl_large_500k_p32x1_c005$' | tr '\n' ' ')"
ssh -o BatchMode=yes $MAC "cd $MAC_REPO && D=~/Documents/songmae_data && SPEC_ROOT=\$D/specs WAV_ROOT=\$D/wavs PYTHON_BIN=~/miniforge3/envs/mae/bin/python CONDITIONS='$mac_conditions' nohup caffeinate -i SongMAE_TMLR_Revision/scripts/probe_all.sh > logs/probes_mac.log 2>&1 < /dev/null &"
CONDITIONS="$here_conditions" PYTHON_BIN=$P SongMAE_TMLR_Revision/scripts/probe_all.sh
until ssh -o BatchMode=yes $MAC "tail -1 $MAC_REPO/logs/probes_mac.log" | grep -q "probes complete"; do sleep 120; done
rsync -a --exclude manifests "$MAC:$MAC_REPO/results/probes/" results/probes/
echo "chain complete"
