#!/usr/bin/env python3
import argparse
import csv
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


COLORS = ("#0072B2", "#E69F00")
OUTPUTS = {
    "micro": Path("imgs/linear_probe/micro_ablations.png"),
    "models": Path("imgs/linear_probe/model_comparison.png"),
}
SECTION_TITLES = {
    "Masking strategy (32x1; Voronoi C=0.1)": "Masking Strategy",
    "Patch shape (Voronoi, C=0.1)": "Patch Shape",
    "Voronoi C parameter (32x1)": "Voronoi C (32×1)",
    "Voronoi C parameter (32x4)": "Voronoi C (32×4)",
}


def parse_cell(cell):
    if cell == "-":
        return None
    match = re.fullmatch(r"[\d.]+ \(([\d.]+)/([\d.]+)\)", cell)
    assert match, cell
    return tuple(map(float, match.groups()))


def read_micro(rows):
    assert rows[0][0] == "Linear Probe Ablations"
    panels = []
    for row in rows[2:]:
        if row[0] in SECTION_TITLES:
            panels.append((SECTION_TITLES[row[0]], []))
        else:
            panels[-1][1].append((row[0], parse_cell(row[-1])))
    return panels


def read_models(rows):
    assert rows[0][0] == "Model"
    titles = [header.removesuffix(" Macro FER (P/I)") for header in rows[0][1:]]
    titles[1:3] = ["Zebra Finch", "Bengalese Finch"]
    return [
        (title, [(row[0], parse_cell(row[column])) for row in rows[1:]])
        for column, title in enumerate(titles, 1)
    ]


def tick_label(label):
    return label.replace("x", "×").replace("Large ", "SongMAE\n").replace("HuBERT ", "HuBERT\n")


def plot(panels, output, ylabel):
    values = [value for _, rows in panels for _, value in rows if value is not None]
    assert values
    upper = 5 * math.ceil(max(sum(value) for value in values) / 5)
    fig, axes = plt.subplots(1, len(panels), figsize=(2.7 * len(panels), 4), sharey=True)

    for ax, (title, rows) in zip(axes, panels):
        for x, (_, value) in enumerate(rows):
            if value is None:
                ax.text(x, 0.5, "—", ha="center", va="bottom", color="#666666")
                continue
            parsing, identity = value
            ax.bar(x, parsing, width=0.68, color=COLORS[0])
            ax.bar(x, identity, width=0.68, bottom=parsing, color=COLORS[1])

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, upper)
        ax.set_xticks(range(len(rows)), [tick_label(label) for label, _ in rows], fontsize=8)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.set_box_aspect(1)

    axes[0].set_ylabel(ylabel)
    fig.legend(
        [Patch(color=COLORS[0]), Patch(color=COLORS[1])],
        ["Parsing error", "Identity error"],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.14, top=0.78, wspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output)
    print(output.with_suffix(".pdf"))


def main():
    parser = argparse.ArgumentParser(description="Plot linear-probe aggregator TSV from stdin.")
    parser.add_argument("table", choices=OUTPUTS)
    args = parser.parse_args()

    rows = list(csv.reader(sys.stdin, delimiter="\t"))
    assert rows
    panels = read_micro(rows) if args.table == "micro" else read_models(rows)
    ylabel = "Mean Macro FER (%)" if args.table == "micro" else "Macro FER (%)"
    plot(panels, OUTPUTS[args.table], ylabel)


if __name__ == "__main__":
    main()
