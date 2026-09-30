#!/usr/bin/env bash
# Oracle Macro FER (R1) for every bird at 1/2/5/20/40/80/160 ms output grids, 8 birds in parallel (CPU only).
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN=${PYTHON_BIN:-/home/george-vengrovski/anaconda3/envs/mae/bin/python}
SPEC_ROOT=${SPEC_ROOT:-/media/george-vengrovski/disk2/specs}
OUT=SongMAE_TMLR_Revision/results/oracle
for row in "canary|canary_5ms" "zf|zebra_finch_5ms" "bf|bengalese_finch_5ms"; do
  IFS='|' read -r dataset specs <<< "$row"
  annotations="files/annotation jsons/${dataset}_annotations.json"
  mkdir -p "$OUT/$dataset"
  "$PYTHON_BIN" -c "import json,sys;print('\n'.join(sorted({r['recording']['bird_id'] for r in json.load(open(sys.argv[1]))['recordings']})))" "$annotations" \
    | sed "s|^|$dataset $specs |"
done | xargs -P 8 -L 1 bash -c '[[ -s '"$OUT"'/$0/$2.json ]] || '"$PYTHON_BIN"' src/evals/syllable_oracle.py --spec_dir '"$SPEC_ROOT"'/$1 --annotations "files/annotation jsons/$0_annotations.json" --bird $2 > '"$OUT"'/$0/$2.json'
echo "oracle complete: $(ls $OUT/*/*.json | wc -l) birds"
