#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p imgs/loss_plots

for csv in runs/*/losses.csv; do
  run="$(basename "$(dirname "$csv")")"
  for split in train val; do
    out="imgs/loss_plots/${run}_${split}_mse_loss.png"
    python - "$csv" "$out" "$run" "$split" <<'PY'
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path, out_path, run_name, split = sys.argv[1:]
steps = []
losses = []

with open(csv_path) as f:
    for row in csv.DictReader(f):
        step = int(row["step"])
        loss = float(row["loss"])
        if row["split"] == split and step > 0 and loss > 0:
            steps.append(step)
            losses.append(loss)

if not steps:
    raise SystemExit(f"no positive {split} rows: {csv_path}")

fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=300)
ax.plot(steps, losses, marker="o", markersize=2.5, linewidth=1.4)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title(f"{run_name} {split}", fontsize=9)
ax.set_xlabel("step")
ax.set_ylabel(f"{split} MSE loss")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(out_path)
PY
  done
done
