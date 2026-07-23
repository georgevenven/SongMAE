#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


SPECIES = {
    "canary": ("Canary", "#D45087"),
    "zf": ("Zebra Finch", "#7A5195"),
    "bf": ("Bengalese Finch", "#2A9D8F"),
}


def load_rows(path):
    rows = list(csv.DictReader(path.open(), delimiter="\t"))
    assert rows, f"no syllable-class metrics in {path}"
    return rows


def fit(rows):
    x = np.log2([float(row["median_duration_ms"]) for row in rows])
    y = np.array([
        float(row["songmae_32x1_entropy_completeness"])
        - float(row["songmae_32x4_entropy_completeness"])
        for row in rows
    ])
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["dataset"], row["bird"]].append(index)
    centered_x, centered_y = x.copy(), y.copy()
    for indices in groups.values():
        centered_x[indices] -= centered_x[indices].mean()
        centered_y[indices] -= centered_y[indices].mean()
    slope = centered_x @ centered_y / (centered_x @ centered_x)
    residual = centered_y - slope * centered_x
    scores = [centered_x[indices] @ residual[indices] for indices in groups.values()]
    n, birds = len(rows), len(groups)
    variance = sum(score**2 for score in scores) / (centered_x @ centered_x) ** 2
    variance *= birds / (birds - 1) * (n - 1) / (n - birds - 1)
    p_value = 2 * t.sf(abs(slope / np.sqrt(variance)), birds - 1)
    r_squared = 1 - (residual @ residual) / (centered_y @ centered_y)
    return x, y, slope, r_squared, p_value


def main():
    parser = argparse.ArgumentParser(description="Plot syllable duration against SongMAE entropy advantage.")
    parser.add_argument(
        "--class_metrics",
        type=Path,
        default=Path(
            "results/syllable_kmeans_50birds_4models_raster_pca128_250k/"
            "duration_analysis/class_metrics.tsv"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("imgs/syllable_kmeans_entropy_duration.png"))
    args = parser.parse_args()

    rows = load_rows(args.class_metrics)
    x, y, slope, r_squared, p_value = fit(rows)
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=300)
    for species, (label, color) in SPECIES.items():
        selected = np.array([row["dataset"] == species for row in rows])
        ax.scatter(2**x[selected], y[selected], s=15, alpha=0.5, color=color, label=label)
    line_x = np.geomspace(2**x.min(), 2**x.max(), 200)
    line_y = y.mean() + slope * (np.log2(line_x) - x.mean())
    ax.plot(line_x, line_y, color="black", linewidth=2, label="Within-bird fit")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(75, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks([10, 20, 40, 75, 150, 300], ["10", "20", "40", "75", "150", "300"])
    ax.set_xlabel("Median syllable duration (ms)")
    ax.set_ylabel("32×1 entropy advantage ($H_{32×4} - H_{32×1}$)")
    ax.text(
        0.97,
        0.97,
        f"$\\beta$ = {slope:.3f} per doubling\nWithin-bird $R^2$ = {r_squared:.3f}\n$p$ = {p_value:.1e}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.output)
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
