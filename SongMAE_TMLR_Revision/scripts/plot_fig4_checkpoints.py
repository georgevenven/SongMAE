#!/usr/bin/env python3
"""Figure 4: kNN purity (k = 100, last layer) across pretraining checkpoints, SongMAE 32 mels × 5 ms
-> latex/figures/main_figure_3_scaling.{png,pdf}. Same style as src/plotting_utils/plot_knn_purity_steps.py.

Source: results/knn/checkpoints (scripts/knn_checkpoints.sh; same-recording neighbours excluded). Classes in fewer
than two recordings of a bird's kNN selection are dropped (results/knn_rare_classes.json); birds averaged per species.
"""
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "SongMAE_TMLR_Revision"
KNN = ROOT / "results/knn/checkpoints"
RARE = json.loads((REV / "results/knn_rare_classes.json").read_text())
BIRDS = {"zf": 36, "bf": 11, "canary": 3}
SPECIES_LABELS = {"zf": "zebra finch", "bf": "Bengalese finch", "canary": "canary"}
CHECKPOINTS = (("000000", "0k"), ("020000", "20k"), ("050000", "50k"), ("100000", "100k"), ("499999", "500k"))
MODELS = {"large": ("Large", "#440154", "-"), "base": ("Base", "#21918C", "--"), "micro": ("Micro", "#FDE725", ":")}


def purity(species, size, step):
    values = []
    for path in KNN.glob(f"{species}/*/xcl_{size}_500k_p32x1_c005_step_{step}/layer_*/summary.json"):
        row = next(r for r in json.loads(path.read_text())["rows"] if r["k"] == 100)
        rare = {str(c) for c in RARE.get(f"{species}/{path.parts[-4]}", [])}
        values.append(100 * st.mean(v for c, v in row["per_class_same_purity"].items() if c not in rare))
    assert len(values) == BIRDS[species], (species, size, step, len(values))
    return st.mean(values)


fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8), dpi=200, sharex=True)
handles = [axes[0].plot([], [], marker="o", markersize=5, linewidth=2, color=c, linestyle=ls, label=label)[0]
           for label, c, ls in MODELS.values()]
for axis, species in zip(axes, BIRDS):
    present = []
    for size, (_, color, style) in MODELS.items():
        y = [purity(species, size, step) for step, _ in CHECKPOINTS]
        present += y
        axis.plot(range(len(y)), y, marker="o", markersize=4.5, linewidth=2, color=color, linestyle=style)
        print(species, size, " ".join(f"{v:.1f}" for v in y))
    axis.set_title(SPECIES_LABELS[species], fontsize=11)
    axis.set_xticks(range(len(CHECKPOINTS)), [label for _, label in CHECKPOINTS])
    axis.grid(alpha=0.18)
    axis.set_axisbelow(True)
    axis.set_ylim(max(0, min(present) - 5), min(100, max(present) + 5))
axes[0].set_ylabel("Macro kNN purity (%) ↑")
fig.supxlabel("Training step", y=0.035)
axes[-1].legend(handles=handles, loc="lower right", fontsize=8, frameon=True, framealpha=0.78, facecolor="white",
                edgecolor="none", borderpad=0.3, handlelength=1.5)
fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.92, wspace=0.22)
out = REV / "latex/figures/main_figure_3_scaling.png"
fig.savefig(out, dpi=600, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(out)
