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
STYLE = {"canary": ("Canary",), "zf": ("Zebra finch",), "bf": ("Bengalese finch",)}


def oracle_curves():
    """Macro-optimal oracle (the lower bound): bin widths, per-species Macro FER (birds averaged), and their mean."""
    rows = {s: [json.loads(p.read_text())["rows"] for p in sorted((RESULTS / s).glob("*.json"))] for s in SPECIES}
    idx = [i for i, r in enumerate(rows["zf"][0]) if r["oracle"] == "macro_optimal"]
    per = {s: [100 * st.mean(b[i]["macro_fer"] for b in rows[s]) for i in idx] for s in SPECIES}
    return [rows["zf"][0][i]["bin_ms"] for i in idx], per, [st.mean(v) for v in zip(*per.values())]


def main():
    x, per, _ = oracle_curves()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), dpi=200, sharey=True)
    for axis, s in zip(axes, SPECIES):
        axis.plot(x, per[s], color="#222222", linewidth=2.2, marker="o", markersize=4.5)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks([5, 20, 40, 80, 160])
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20])
        axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set_xlim(4, 200)
        axis.set_ylim(0.2, 30)
        axis.set_title(STYLE[s][0], fontsize=13)
        axis.set_box_aspect(1)
        axis.grid(alpha=0.18)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Macro FER (%) ↓", fontsize=12)
    fig.supxlabel("Output bin (ms)", y=0.02)
    for suffix, dpi in [(".png", 300), (".pdf", None), (".svg", None)]:
        fig.savefig(HERE / f"oracle_fer{suffix}", dpi=dpi, bbox_inches="tight")
    fig.savefig(HERE.parents[1] / "latex/figures/supplement_figure_oracle_fer.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(HERE / "oracle_fer.png")


if __name__ == "__main__":
    main()
