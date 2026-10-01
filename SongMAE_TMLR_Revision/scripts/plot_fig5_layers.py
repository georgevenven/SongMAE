#!/usr/bin/env python3
"""Figure 5: kNN purity (k = 100) across encoder layers -> latex/figures/main_figure_4_layers.{png,pdf}.

Birds are averaged within species, then species equally (results/per_bird/knn_layers.csv).
Show native resolution and 20/5 ms only; coarser resolutions use greyer shades.
"""
import collections
import csv
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

REV = Path(__file__).resolve().parents[1]
LAYERS = range(12)
STYLES = {5: (0, "-"), 20: (0.4, "--"), 160: (0.7, ":")}
PANELS = [  # title, colour, (condition, resolution, legend label)
    ("SongMAE-Large", "#0072B2", [
        ("xcl_large_500k_p32x1_c005", 5, "5 ms"),
        ("xcl_large_500k_p32x4_c010", 20, "20 ms")]),
    ("BirdAVES", "#D55E00", [
        ("birdaves_biox_base_speed0p25", 5, "5 ms"),
        ("birdaves_biox_base", 20, "20 ms (native)")]),
    ("HuBERT", "#009E73", [("hubert_base_ls960", 20, "20 ms (native)")]),
    ("BEATs", "#CC79A7", [
        ("beats_iter3_plus_as2m_speed0p03125", 5, "5 ms"),
        ("beats_iter3_plus_as2m_speed0p125", 20, "20 ms"),
        ("beats_iter3_plus_as2m", 160, "160 ms (native)")]),
    ("Bird-MAE", "#444444", [
        ("birdmae_base_speed0p03125", 5, "5 ms"),
        ("birdmae_base_speed0p125", 20, "20 ms"),
        ("birdmae_base_speed1", 160, "160 ms (native)")]),
]

values = collections.defaultdict(list)
for r in csv.DictReader(open(REV / "results/per_bird/knn_layers.csv")):
    if r["k"] == "100":
        values[r["condition"], int(r["layer"]), r["species"]].append(float(r["purity"]))


def curve(condition):
    assert all(len(values[condition, l, s]) == n
               for l in LAYERS for s, n in (("canary", 3), ("zf", 36), ("bf", 11))), condition
    return [st.mean(st.mean(values[condition, l, s]) for s in ("canary", "zf", "bf")) for l in LAYERS]


def greyed(color, amount):
    return mcolors.to_hex([c + (0.78 - c) * amount for c in mcolors.to_rgb(color)])


fig, axes = plt.subplots(1, 5, figsize=(9.5, 3), dpi=200, sharex=True, sharey=True)
for i, (axis, (title, color, lines)) in enumerate(zip(axes, PANELS)):
    for condition, resolution, label in reversed(lines):
        amount, linestyle = STYLES[resolution]
        c = greyed(color, amount) if len(lines) > 1 else color
        y = curve(condition)
        axis.plot(LAYERS, y, color=c, linestyle=linestyle, marker="o", markersize=3.5,
                  linewidth=2, label=label)
        print(f"{condition:34s} best L{max(LAYERS, key=lambda l: y[l])} {max(y):.1f}")
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles[::-1], labels[::-1], fontsize=8, frameon=False, loc="lower right",
                handlelength=1.3, labelspacing=0.3)
    axis.set_title(title, fontsize=12)
    axis.set_xticks(LAYERS)
    axis.set_xticklabels([str(l) if l % 2 == 0 else "" for l in LAYERS])
    axis.grid(alpha=0.18)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=10, labelleft=i == 0)
fig.supylabel("Macro kNN purity (%) ↑", x=0.01, fontsize=12)
fig.supxlabel("Encoder layer", y=0.02, fontsize=12)
fig.subplots_adjust(left=0.07, right=0.995, bottom=0.20, top=0.85, wspace=0.08)
out = REV / "latex/figures/main_figure_4_layers.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(out)
