#!/usr/bin/env bash
# Work desktop: probe SongMAE/BirdAVES/HuBERT here in PARTS parallel streams (birds split round-robin, canary and bf
# first; finished results are skipped), then wait for the Mac's BEATs/Bird-MAE probes and pull them back.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=/home/george-vengrovski/anaconda3/envs/mae/bin/python
PARTS=${PARTS:-3}
all=$($P -c "import json;print(' '.join(k for k in json.load(open('SongMAE_TMLR_Revision/results/layer_selection.json')) if '_100k_' not in k))")
here="xcl_large_500k_p32x1_c005 $(tr ' ' '\n' <<< "$all" | grep -vE '^(beats|birdmae)_|^xcl_large_500k_p32x1_c005$' | tr '\n' ' ')"
birds=$($P -c "
import json
out = []
for sp in ['canary', 'bf', 'zf']:
    out += sorted({r['recording']['bird_id'] for r in json.load(open(f'files/annotation jsons/{sp}_annotations.json'))['recordings']})
print(' '.join(out))")
read -r -a birds <<< "$birds"
for ((part = 0; part < PARTS; part++)); do
  mine=""; for ((i = part; i < ${#birds[@]}; i += PARTS)); do mine="$mine ${birds[$i]}"; done
  CONDITIONS="$here" BIRD_FILTER="$mine" PYTHON_BIN=$P SongMAE_TMLR_Revision/scripts/probe_all.sh > "logs/probes_part$part.log" 2>&1 &
done
wait
grep -h "failed" logs/probes_part*.log || true
until ssh -o BatchMode=yes macstudio "tail -1 Documents/SongMAE-reviews/logs/probes_mac.log" | grep -q "probes complete"; do sleep 120; done
rsync -a --exclude manifests macstudio:Documents/SongMAE-reviews/results/probes/ results/probes/
echo "all probes complete"
