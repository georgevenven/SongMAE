#!/usr/bin/env python3
"""Oracle Macro FER and parsing error vs output resolution (R1), in the style of the BEANS figure."""
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
# PROVISIONAL: submitted Table 3 (mean over species); replace with the corrected probe re-run.
MODELS = {  # name: (bin ms, Macro FER, parsing, color, marker, x nudge so 20 ms points don't overlap)
    "SongMAE 32 mels × 5 ms": (5, 4.91, 1.85, "#0072B2", "o", 1.0),
    "SongMAE 32 mels × 20 ms": (20, 7.13, 4.24, "#56B4E9", "o", 1.0),
    "BirdAVES": (20, 7.31, 4.16, "#009E73", "s", 0.86),
    "HuBERT": (20, 8.46, 4.32, "#E69F00", "s", 1.16),
}


def oracle_curves():
    rows = {s: [json.loads(p.read_text())["rows"] for p in sorted((RESULTS / s).glob("*.json"))] for s in SPECIES}
    curves = {}
    for oracle in ("majority", "macro_optimal"):
        idx = [i for i, r in enumerate(rows["zf"][0]) if r["oracle"] == oracle]
        mean = lambda i, k: st.mean(100 * st.mean(b[i][k] for b in rows[s]) for s in SPECIES)
        curves[oracle] = ([rows["zf"][0][i]["bin_ms"] for i in idx],
                          [mean(i, "macro_fer") for i in idx], [mean(i, "macro_parsing_error") for i in idx])
    return curves


def panel(axis, curves, column, title, ylabel):
    for oracle, label, style in [("macro_optimal", "Oracle lower bound", dict(color="#222222", linewidth=2.2)),
                                 ("majority", "Majority-label oracle", dict(color="#999999", linewidth=1.6, linestyle="--"))]:
        x, *values = curves[oracle]
        axis.plot(x, values[column], marker="o", markersize=4.5, label=label, zorder=3, **style)
    for name, (ms, fer, parsing, color, marker, nudge) in MODELS.items():
        axis.scatter(ms * nudge, (fer, parsing)[column], s=58, marker=marker, color=color, label=name, zorder=5)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks([5, 20, 40, 80, 160])
    axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_yticks([0.5, 1, 2, 5, 10, 20, 40])
    axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xlim(4, 200)
    axis.set_ylim(0.4, 50)
    axis.set_title(title, fontsize=13)
    axis.set_xlabel("Output bin (ms)")
    axis.set_ylabel(ylabel)
    axis.set_box_aspect(1)
    axis.grid(alpha=0.18)
    axis.set_axisbelow(True)


def main():
    curves = oracle_curves()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.1), dpi=200)
    panel(axes[0], curves, 0, "Macro FER", "Macro FER (%) ↓")
    panel(axes[1], curves, 1, "Parsing error", "Parsing error (%) ↓")
    axes[0].annotate("BirdMAE / BEATs\nat 1× speed", (160, curves["macro_optimal"][1][-1]), xytext=(-2, -95),
                     textcoords="offset points", fontsize=8, ha="right", va="top", color="#555555",
                     arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.65, "shrinkA": 2, "shrinkB": 4})
    axes[1].legend(frameon=False, loc="upper left", fontsize=7.5)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.15, top=0.91, wspace=0.22)
    for suffix, dpi in [(".png", 300), (".pdf", None), (".svg", None)]:
        fig.savefig(HERE / f"oracle_fer{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(HERE / "oracle_fer.png")


if __name__ == "__main__":
    main()
