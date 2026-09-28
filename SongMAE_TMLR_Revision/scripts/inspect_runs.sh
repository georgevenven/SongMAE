#!/usr/bin/env bash
# Gather the latest reconstruction image and losses of every revision run (work desktop + Twins) into
# SongMAE_TMLR_Revision/inspect/, and plot validation loss per configuration. Re-run any time to refresh.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=SongMAE_TMLR_Revision/inspect
TWINS=${TWINS:-george-vengrovski@163.41.128.66}
PYTHON_BIN=${PYTHON_BIN:-/home/george-vengrovski/anaconda3/envs/mae/bin/python}
mkdir -p "$OUT/recon" "$OUT/losses"

collect() {  # $1 = runs dir (local path or host:path), $2 = run names
  for run in $2; do
    rsync -a "$1/$run/losses.csv" "$OUT/losses/$run.csv" 2>/dev/null || continue
    last=$( [[ $1 == *:* ]] && ssh -o BatchMode=yes "${1%%:*}" "ls ${1#*:}/$run/imgs | grep recon | sort | tail -1" \
      || ls "$1/$run/imgs" | grep recon | sort | tail -1 )
    [[ -n $last ]] && rsync -a "$1/$run/imgs/$last" "$OUT/recon/${run}__${last}"
    ls "$OUT/recon/${run}__"*.png 2>/dev/null | sort | head -n -1 | xargs -r rm -f  # keep only the newest
  done
}

local_runs=$(ls runs | grep -E '^xcl_micro_100k_.*_seed[0-9]+$' || true)
twins_runs=$(ssh -o BatchMode=yes "$TWINS" 'ls ~/Documents/SongMAE-reviews/runs' | grep -E '_seed[0-9]+$' || true)
original="Xcl_micro_100k_p32x1_c0025 Xcl_micro_100k_p32x1_c005 Xcl_micro_100k_p32x1_c010 xcl_micro_100k_p32x1_random"
collect runs "$local_runs $original"
collect "$TWINS:Documents/SongMAE-reviews/runs" "$twins_runs"

"$PYTHON_BIN" - "$OUT" <<'EOF'
import csv, re, sys
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
groups = defaultdict(dict)
for path in sorted((out / "losses").glob("*.csv")):
    name = path.stem
    config = re.sub(r"^[Xx]cl_micro_100k_|_seed\d+$", "", name)
    seed = re.search(r"_seed(\d+)$", name)
    rows = [r for r in csv.DictReader(path.open()) if r["split"] == "val"]
    groups[config][f"seed {seed.group(1)}" if seed else "original (seed 0)"] = (
        [int(r["step"]) for r in rows], [float(r["loss"]) for r in rows])
fig, axes = plt.subplots(1, len(groups), figsize=(4.5 * len(groups), 3.6), squeeze=False, sharey=True)
for ax, (config, runs) in zip(axes[0], sorted(groups.items())):
    for label, (steps, loss) in sorted(runs.items()):
        ax.plot(steps, loss, label=f"{label} ({steps[-1] // 1000}k)")
    ax.set(title=config, xlabel="step", yscale="log")
    ax.legend(fontsize=7)
axes[0][0].set_ylabel("validation MSE (masked patches)")
fig.suptitle("Validation loss is only comparable within a masking type (harder masks give higher loss)", fontsize=9)
fig.tight_layout()
fig.savefig(out / "val_loss_by_config.png", dpi=150)
print(f"wrote {out / 'val_loss_by_config.png'}: " + ", ".join(f"{c} ({len(r)})" for c, r in sorted(groups.items())))
EOF
