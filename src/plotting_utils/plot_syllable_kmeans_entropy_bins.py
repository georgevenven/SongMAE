#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BINS = (
    (0, 20, "<20"),
    (20, 40, "20–40"),
    (40, 60, "40–60"),
    (60, 80, "60–80"),
    (80, 100, "80–100"),
)
MODELS = (
    ("SongMAE 32×1", "songmae_32x1", "#0072B2", "o"),
    ("SongMAE 32×4", "songmae_32x4", "#56B4E9", "s"),
    ("BirdAVES", "birdaves", "#009E73", "^"),
    ("HuBERT", "hubert", "#D55E00", "D"),
)


def main():
    parser = argparse.ArgumentParser(description="Plot normalized syllable entropy by duration bin.")
    parser.add_argument(
        "--class_metrics",
        type=Path,
        default=Path(
            "results/syllable_kmeans_50birds_4models_raster_pca128_250k/"
            "duration_analysis/class_metrics.tsv"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("imgs/syllable_kmeans_entropy_bins.png"))
    args = parser.parse_args()
    rows = list(csv.DictReader(args.class_metrics.open(), delimiter="\t"))
    assert rows

    summaries = []
    for low, high, label in BINS:
        selected = [
            row
            for row in rows
            if low <= float(row["median_duration_ms"]) < high
            or high == 100 and float(row["median_duration_ms"]) == high
        ]
        birds = {(row["dataset"], row["bird"]) for row in selected}
        for model, key, _, _ in MODELS:
            by_bird = defaultdict(list)
            for row in selected:
                by_bird[row["dataset"], row["bird"]].append(
                    1 - float(row[f"{key}_entropy_completeness"])
                )
            values = np.array([np.mean(value) for value in by_bird.values()])
            summaries.append({
                "duration_bin": label,
                "model": model,
                "classes": len(selected),
                "birds": len(birds),
                "mean_entropy": values.mean(),
                "sem": values.std(ddof=1) / np.sqrt(values.size),
            })

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    x = np.arange(len(BINS))
    for model, _, color, marker in MODELS:
        selected = [row for row in summaries if row["model"] == model]
        ax.errorbar(
            x,
            [row["mean_entropy"] for row in selected],
            yerr=[row["sem"] for row in selected],
            color=color,
            marker=marker,
            linewidth=2,
            markersize=5,
            capsize=3,
            label=model,
        )
    counts = [row for row in summaries if row["model"] == MODELS[0][0]]
    ax.set_xticks(
        x,
        [f'{row["duration_bin"]}\nn={row["classes"]}; {row["birds"]} birds' for row in counts],
        fontsize=8,
    )
    ax.set_xlabel("Median syllable duration (ms)")
    ax.set_ylabel("Normalized per-syllable cluster entropy")
    ax.set_ylim(0, 0.65)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(title="Mean ± SEM across birds", frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    table = args.output.with_suffix(".tsv")
    with table.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(summaries)
    print(args.output)
    print(args.output.with_suffix(".pdf"))
    print(table)


if __name__ == "__main__":
    main()
