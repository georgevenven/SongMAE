#!/usr/bin/env python3
"""Plot checkpoint kNN purity in the linear-probe sweep style."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SPECIES = ("zf", "bf", "canary")
SPECIES_LABELS = {"zf": "zebra finch", "bf": "Bengalese finch", "canary": "canary"}
CHECKPOINTS = (
    ("000000", "0k"),
    ("020000", "20k"),
    ("050000", "50k"),
    ("100000", "100k"),
    ("499999", "500k"),
)
MODELS = {
    "large": ("Large", "#440154", "-"),
    "base": ("Base", "#21918C", "--"),
    "micro": ("Micro", "#FDE725", ":"),
}


def load_purity(root, k):
    values = defaultdict(list)
    for path in root.glob("*/*/*/layer_*/end_of_block/summary.json"):
        species = path.relative_to(root).parts[0]
        data = json.loads(path.read_text())
        row = next(row for row in data["rows"] if row["k"] == k)
        values[species, data["name"]].append(100 * row["macro_same_purity"])
    return {key: sum(rows) / len(rows) for key, rows in values.items()}


def plot_shape(values, shape, k, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8), dpi=200, sharex=True)
    handles = [
        axes[0].plot(
            [], [], marker="o", markersize=5, linewidth=2,
            color=color, linestyle=style, label=label,
        )[0]
        for label, color, style in MODELS.values()
    ]

    for axis, species in zip(axes, SPECIES):
        present = []
        for size, (_, color, style) in MODELS.items():
            points = [
                (i, values.get((species, f"{size}_{shape}_step_{step}")))
                for i, (step, _) in enumerate(CHECKPOINTS)
            ]
            x = [i for i, value in points if value is not None]
            y = [value for _, value in points if value is not None]
            present.extend(y)
            axis.plot(x, y, marker="o", markersize=4.5, linewidth=2, color=color, linestyle=style)
        axis.set_title(SPECIES_LABELS[species], fontsize=11)
        axis.set_xticks(range(len(CHECKPOINTS)), [label for _, label in CHECKPOINTS])
        axis.grid(alpha=0.18)
        axis.set_axisbelow(True)
        if present:
            axis.set_ylim(max(0, min(present) - 5), min(100, max(present) + 5))
        else:
            axis.set_ylim(0, 100)
            axis.text(0.5, 0.5, "No completed runs", ha="center", va="center", transform=axis.transAxes)

    axes[0].set_ylabel("Macro kNN purity (%) ↑")
    fig.supxlabel("Training step", y=0.035)
    axes[-1].legend(
        handles=handles,
        loc="upper left",
        fontsize=8,
        frameon=True,
        framealpha=0.78,
        facecolor="white",
        edgecolor="none",
        borderpad=0.3,
        handlelength=1.5,
        title=f"{shape.replace('x', '×')} · k={k}",
        title_fontsize=8,
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.92, wspace=0.22)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"checkpoint_purity_{shape}_k{k}.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output)
    print(output.with_suffix(".pdf"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_root",
        default="results/knn/songmae_checkpoints_k1_k5_k10_k50_k100_last_layer",
    )
    parser.add_argument("--output_dir", default="imgs/knn_purity")
    parser.add_argument("--k", type=int, default=100)
    args = parser.parse_args()
    values = load_purity(Path(args.results_root), args.k)
    assert values, f"no k={args.k} results under {args.results_root}"
    for shape in ("32x1", "32x4"):
        plot_shape(values, shape, args.k, Path(args.output_dir))


if __name__ == "__main__":
    main()
