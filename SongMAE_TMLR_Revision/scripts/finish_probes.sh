#!/usr/bin/env bash
# Work desktop: probe SongMAE/BirdAVES/HuBERT here (resumes; finished results are skipped), then wait for the Mac's
# BEATs/Bird-MAE probes and pull them back. Use after chain_probes.sh has selected layers and started the Mac.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=/home/george-vengrovski/anaconda3/envs/mae/bin/python
all=$($P -c "import json;print(' '.join(k for k in json.load(open('SongMAE_TMLR_Revision/results/layer_selection.json')) if '_100k_' not in k))")
here="xcl_large_500k_p32x1_c005 $(tr ' ' '\n' <<< "$all" | grep -vE '^(beats|birdmae)_|^xcl_large_500k_p32x1_c005$' | tr '\n' ' ')"
CONDITIONS="$here" PYTHON_BIN=$P SongMAE_TMLR_Revision/scripts/probe_all.sh
until ssh -o BatchMode=yes macstudio "tail -1 Documents/SongMAE-reviews/logs/probes_mac.log" | grep -q "probes complete"; do sleep 120; done
rsync -a --exclude manifests macstudio:Documents/SongMAE-reviews/results/probes/ results/probes/
echo "all probes complete"
