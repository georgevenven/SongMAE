#!/usr/bin/env python3
"""Figure 5: kNN purity (k = 100) across encoder layers -> latex/figures/main_figure_4_layers.{png,pdf}.

Birds are averaged within species, then species equally (results/per_bird/knn_layers.csv). Slowed playback of a
baseline is drawn in its panel in progressively greyed-out shades of the model's colour.
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
SPEEDS = {"1": "1×", "0p5": "½×", "0p25": "¼×", "0p125": "⅛×", "0p0625": "1/16×", "0p03125": "1/32×"}
PANELS = [  # title, colour, condition stem, slowed speeds
    ("SongMAE-Large\n(32 mels × 5 ms)", "#0072B2", "xcl_large_500k_p32x1_c005", []),
    ("SongMAE-Large\n(32 mels × 20 ms)", "#56B4E9", "xcl_large_500k_p32x4_c010", []),
    ("BirdAVES", "#D55E00", "birdaves_biox_base", ["0p5", "0p25"]),
    ("HuBERT", "#009E73", "hubert_base_ls960", []),
    ("BEATs", "#CC79A7", "beats_iter3_plus_as2m", ["0p5", "0p25"]),
    ("Bird-MAE", "#444444", "birdmae_base", ["0p5", "0p25", "0p125", "0p0625", "0p03125"]),
]

values = collections.defaultdict(list)
for r in csv.DictReader(open(REV / "results/per_bird/knn_layers.csv")):
    if r["k"] == "100":
        values[r["condition"], int(r["layer"]), r["species"]].append(float(r["purity"]))


def curve(condition):
    assert all(sum(len(values[condition, l, s]) for s in ("canary", "zf", "bf")) == 50 for l in LAYERS), condition
    return [st.mean(st.mean(values[condition, l, s]) for s in ("canary", "zf", "bf")) for l in LAYERS]


def greyed(color, amount):
    return mcolors.to_hex([c + (0.78 - c) * amount for c in mcolors.to_rgb(color)])


fig, axes = plt.subplots(1, len(PANELS), figsize=(14, 3.4), dpi=200, sharey=True)
for axis, (title, color, stem, slowed) in zip(axes, PANELS):
    lines = [(stem if stem != "birdmae_base" else "birdmae_base_speed1", "1", color)]
    lines += [(f"{stem}_speed{s}", s, greyed(color, 0.35 + 0.5 * i / max(1, len(slowed) - 1)))
              for i, s in enumerate(slowed)]
    for condition, speed, c in reversed(lines):
        y = curve(condition)
        axis.plot(LAYERS, y, color=c, marker="o", markersize=3.5 if speed != "1" else 4,
                  linewidth=1.6 if speed != "1" else 2.25, label=SPEEDS[speed])
        print(f"{condition:34s} best L{max(LAYERS, key=lambda l: y[l])} {max(y):.1f}")
    if slowed:
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(handles[::-1], labels[::-1], fontsize=9, frameon=False, loc="lower right",
                    ncol=2 if len(slowed) > 2 else 1, handlelength=1.2, columnspacing=0.8, labelspacing=0.3)
    axis.set_title(title, fontsize=14)
    axis.set_xticks(LAYERS)
    axis.set_xticklabels([str(l) if l % 2 == 0 else "" for l in LAYERS])
    axis.set_box_aspect(1)
    axis.grid(alpha=0.18)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=12)
axes[0].set_ylabel("Macro kNN purity (%) ↑", fontsize=14)
fig.supxlabel("Encoder layer", y=0.03, fontsize=14)
fig.subplots_adjust(left=0.06, right=0.995, bottom=0.2, top=0.86, wspace=0.08)
out = REV / "latex/figures/main_figure_4_layers.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(out)
