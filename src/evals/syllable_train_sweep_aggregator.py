#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


SPECIES = {
    "bf": "Bengalese Finch",
    "zf": "Zebra Finch",
    "canary": "Canary",
    "cassins_vireo": "Cassin's vireo",
    "american_robin": "Robin",
}

MODEL_LABELS = {
    "xcl_base_500k_p32x4_default": "SongMAE (32x4 base)",
    "xcl_base_500k_p16x1_default": "SongMAE (16x1 base)",
    "xcl_micro_500k_p128x1_default": "SongMAE (128x1 micro)",
    "xcl_micro_500k_p16x1_default": "SongMAE (16x1 micro)",
    "xcl_micro_500k_p32x1_default": "SongMAE (32x1 micro)",
    "xcl_micro_500k_p32x4_default": "SongMAE (32x4 micro)",
    "songmae": "SongMAE",
    "songmae_random": "SongMAE (random init)",
    "aves": "BirdAVES",
    "hubert": "HuBERT",
}

FIELDNAMES = [
    "species",
    "species_label",
    "model",
    "model_label",
    "train_seconds",
    "n_birds",
    "f1",
    "fer",
]


def train_seconds(path):
    token = path.parent.name.removeprefix("train_").removesuffix("s")
    if token == "MAX":
        return token, (1, float("inf"))
    return f"{float(token):g}", (0, float(token))


def model_label(model):
    return MODEL_LABELS.get(model, model.replace("_", " "))


def mean(values):
    return float(np.asarray(values, dtype=np.float64).mean())


def aggregate_rows(root):
    by_key = {}
    sort_keys = {}
    for path in sorted(Path(root).glob("*/*/*/train_*s/metrics.json")):
        species, bird, model = path.parts[-5:-2]
        train_label, sort_key = train_seconds(path)
        data = json.loads(path.read_text())
        key = (species, model, train_label)
        by_key.setdefault(key, []).append((bird, float(data["macro_f1"]), float(data["fer"])))
        sort_keys[train_label] = sort_key

    rows = []
    for species, model, train_label in sorted(by_key, key=lambda x: (x[0], x[1], sort_keys[x[2]])):
        values = by_key[(species, model, train_label)]
        rows.append(
            {
                "species": species,
                "species_label": SPECIES.get(species, species.replace("_", " ").title()),
                "model": model,
                "model_label": model_label(model),
                "train_seconds": train_label,
                "n_birds": len({bird for bird, _, _ in values}),
                "f1": 100.0 * mean([value[1] for value in values]),
                "fer": 100.0 * mean([value[2] for value in values]),
            }
        )
    return rows


def write_csv(rows, out):
    writer = csv.DictWriter(out, fieldnames=FIELDNAMES)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Aggregate syllable train-sweep metrics by species/model/budget.")
    parser.add_argument("--results_root", default="results/syllable_classification_train_sweep")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = aggregate_rows(args.results_root)
    if args.output:
        with Path(args.output).open("w", encoding="utf-8", newline="") as f:
            write_csv(rows, f)
    else:
        write_csv(rows, sys.stdout)


if __name__ == "__main__":
    main()
