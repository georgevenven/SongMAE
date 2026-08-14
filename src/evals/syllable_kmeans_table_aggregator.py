#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


SPECIES = (("canary", "canary"), ("zf", "zebra finch"), ("bf", "Bengalese finch"))
MODELS = (
    ("SongMAE-Large 32×1", "songmae_32x1"),
    ("SongMAE-Large 32×4", "songmae_32x4"),
    ("BirdAVES", "birdaves"),
    ("HuBERT", "hubert"),
    ("SongMAE-Micro 32×1", "micro_32x1"),
    ("SongMAE-Micro 32×4", "micro_32x4"),
    ("SongMAE-Base 32×1", "base_32x1"),
    ("SongMAE-Base 32×4", "base_32x4"),
)


def load_runs(root):
    runs = defaultdict(list)
    for path in sorted(root.glob("*/*/metrics.csv")):
        species, bird = path.parts[-3:-1]
        for row in csv.DictReader(path.open()):
            runs[species, row["model"]].append((bird, float(row["v_measure"])))
    assert runs, f"no K-means metrics under {root}"
    return runs


def values(runs, model):
    species_values = [[value for _, value in runs[species, model]] for species, _ in SPECIES]
    species_means = [sum(group) / len(group) for group in species_values]
    return species_means + [sum(species_means) / len(species_means)]


def print_table(runs, markdown):
    present = {model for _, model in runs}
    models = [(label, model) for label, model in MODELS if model in present]
    assert {model for _, model in models} == present, f"unknown models: {sorted(present)}"
    counts = [len(runs[species, models[0][1]]) for species, _ in SPECIES]
    headers = ["Model"] + [
        f"{label} V-measure (n={count})"
        for (_, label), count in zip(SPECIES, counts)
    ] + ["Equal-species mean V-measure"]
    rows = [[label] + [f"{value:.3f}" for value in values(runs, model)] for label, model in models]
    if not markdown:
        for row in [headers, *rows]:
            print("\t".join(row))
        return
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main():
    parser = argparse.ArgumentParser(description="Aggregate syllable K-means V-measure with equal species weighting.")
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path(
            "results/syllable_kmeans_50birds_4models_raster_pca128_250k_including_silence"
        ),
    )
    parser.add_argument("--format", choices=("tsv", "markdown"), default="tsv")
    args = parser.parse_args()
    print_table(load_runs(args.results_root), args.format == "markdown")


if __name__ == "__main__":
    main()
