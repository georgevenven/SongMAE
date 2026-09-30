#!/usr/bin/env python3
"""Oracle Macro FER vs output resolution (R1), in the style of the BEANS figure."""
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "DejaVu Sans"

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results" / "oracle"
SPECIES = ("canary", "zf", "bf")
MARKS = [("SongMAE", 5, (3, 7)), ("BirdAVES", 20, (13, 11)), ("Bird-MAE", 160, (55, 21))]
STYLE = {"canary": ("Canary",), "zf": ("Zebra finch",), "bf": ("Bengalese finch",)}


def oracle_curves():
    """Macro-optimal oracle (the lower bound): bin widths, per-species Macro FER (birds averaged), and their mean."""
    rows = {s: [json.loads(p.read_text())["rows"] for p in sorted((RESULTS / s).glob("*.json"))] for s in SPECIES}
    idx = [i for i, r in enumerate(rows["zf"][0]) if r["oracle"] == "macro_optimal"]
    per = {s: [100 * st.mean(b[i]["macro_fer"] for b in rows[s]) for i in idx] for s in SPECIES}
    return [rows["zf"][0][i]["bin_ms"] for i in idx], per, [st.mean(v) for v in zip(*per.values())]


def main():
    x, per, mean = oracle_curves()
    curves = [(STYLE[s][0], per[s]) for s in SPECIES] + [("Mean", mean)]
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.6), dpi=200, sharey=True)
    for axis, (title, y) in zip(axes, curves):
        axis.plot(x, y, color="#222222", linewidth=2.2, marker="o", markersize=4.5)
        axis.set_xscale("log")
        axis.set_xticks([1, 2, 5, 20, 40, 80, 160])
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlim(0.8, 200)
        axis.set_ylim(0, 25)
        axis.set_title(title, fontsize=13)
        axis.set_box_aspect(1)
        axis.grid(alpha=0.18)
        axis.set_axisbelow(True)
    for label, ms, text_xy in MARKS:  # native output resolution of each model family
        axes[-1].annotate(label, (ms, mean[x.index(ms)]), xytext=text_xy, fontsize=10, ha="center",
                          arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))
    axes[0].set_ylabel("Macro FER (%) ↓", fontsize=12)
    fig.supxlabel("Output bin (ms)", y=0.02)
    for suffix, dpi in [(".png", 300), (".pdf", None), (".svg", None)]:
        fig.savefig(HERE / f"oracle_fer{suffix}", dpi=dpi, bbox_inches="tight")
    fig.savefig(HERE.parents[1] / "latex/figures/supplement_figure_oracle_fer.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(HERE / "oracle_fer.png")


if __name__ == "__main__":
    main()
