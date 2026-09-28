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


def oracle_curves():
    rows = {s: [json.loads(p.read_text())["rows"] for p in sorted((RESULTS / s).glob("*.json"))] for s in SPECIES}
    curves = {}
    for oracle in ("majority", "macro_optimal"):
        idx = [i for i, r in enumerate(rows["zf"][0]) if r["oracle"] == oracle]
        mean = lambda i, k: st.mean(100 * st.mean(b[i][k] for b in rows[s]) for s in SPECIES)
        curves[oracle] = ([rows["zf"][0][i]["bin_ms"] for i in idx],
                          [mean(i, "macro_fer") for i in idx], [mean(i, "macro_parsing_error") for i in idx])
    return curves


def main():
    curves = oracle_curves()
    fig, axis = plt.subplots(figsize=(4.3, 4.1), dpi=200)
    x, fer, _ = curves["macro_optimal"]  # lowest Macro FER reachable on each output grid
    axis.plot(x, fer, color="#222222", linewidth=2.2, marker="o", markersize=4.5, zorder=3)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks([5, 20, 40, 80, 160])
    axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_yticks([0.5, 1, 2, 5, 10, 20])
    axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xlim(4, 200)
    axis.set_ylim(0.4, 25)
    axis.set_title("Oracle lower bound", fontsize=13)
    axis.set_xlabel("Output bin (ms)")
    axis.set_ylabel("Macro FER (%) ↓")
    axis.set_box_aspect(1)
    axis.grid(alpha=0.18)
    axis.set_axisbelow(True)
    for suffix, dpi in [(".png", 300), (".pdf", None), (".svg", None)]:
        fig.savefig(HERE / f"oracle_fer{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(HERE / "oracle_fer.png")


if __name__ == "__main__":
    main()
