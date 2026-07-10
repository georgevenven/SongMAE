#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("event_map", "Per-song macro event mAP"),
    ("r_precision", "Per-song R-precision"),
]
MODEL_LABELS = {
    "spectrogram_pca_euclidean": "Spectrogram PCA",
    "xcl_base_100k_p32x1_c010": "SongMAE 32x1",
    "xcl_base_100k_p16x4_c010": "SongMAE 16x4",
    "birdaves_biox_base": "BirdAVES",
    "hubert_base_ls960": "HuBERT base",
}
SPECIES_LABELS = {"zf": "Zebra Finch", "bf": "Bengalese Finch", "canary": "Canary"}


def load_rows(root):
    rows = []
    for path in sorted(Path(root).glob("**/summary.json")):
        row = json.loads(path.read_text())
        if all(key in row for key, _ in METRICS):
            rows.append(row)
    assert rows, f"no retrieval summaries under {root}"
    assert {row.get("ranking") for row in rows} == {"per_song_peaks"}, "results must rank peaks per song"
    keys = [(row["species"], row["bird"], row["model"]) for row in rows]
    assert len(keys) == len(set(keys)), "duplicate retrieval summaries"
    for species, bird in {(row["species"], row["bird"]) for row in rows}:
        group = [row for row in rows if (row["species"], row["bird"]) == (species, bird)]
        hashes = {
            (row["query_hash"], row["gallery_hash"])
            for row in group
        }
        assert len(hashes) == 1, f"retrieval inputs differ for {species}/{bird}"
    return rows


def ordered(values, preferred):
    values = set(values)
    return [value for value in preferred if value in values] + sorted(values - set(preferred))


def stats(rows, species, model, metric):
    values = np.asarray([
        float(row[metric]) for row in rows
        if row["species"] == species and row["model"] == model
    ])
    if not values.size:
        return np.nan, (0.0, 0.0)
    center = float(values.mean())
    if values.size == 1:
        return center, (0.0, 0.0)
    samples = np.random.default_rng(0).choice(values, (10_000, len(values))).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return center, (center - low, high - center)


def main():
    parser = argparse.ArgumentParser(description="Plot query-by-example syllable retrieval summaries.")
    parser.add_argument("--results_root", default="results/syllable_retrieval")
    parser.add_argument("--output", default="imgs/syllable_retrieval.png")
    args = parser.parse_args()

    rows = load_rows(args.results_root)
    species = ordered([row["species"] for row in rows], ["zf", "bf", "canary"])
    models = ordered(
        [row["model"] for row in rows],
        [
            "spectrogram_pca_euclidean",
            "xcl_base_100k_p16x4_c010",
            "xcl_base_100k_p32x1_c010",
            "birdaves_biox_base",
            "hubert_base_ls960",
        ],
    )
    x = np.arange(len(species))
    width = 0.8 / len(models)
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, len(METRICS), figsize=(9.5, 3.8), dpi=300, sharey=True)
    for ax, (metric, title) in zip(axes, METRICS):
        for i, model in enumerate(models):
            values = [stats(rows, item, model, metric) for item in species]
            offset = (i - (len(models) - 1) / 2) * width
            ax.bar(
                x + offset,
                [value[0] for value in values],
                width,
                yerr=np.asarray([value[1] for value in values]).T,
                color=colors(i),
                label=MODEL_LABELS.get(model, model.replace("_", " ")),
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SPECIES_LABELS.get(item, item.replace("_", " ").title()) for item in species])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Score")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
