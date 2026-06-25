#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


SPECIES = [
    ("canary", "Canary"),
    ("zf", "Zebra"),
    ("bf", "Bengalese"),
    ("cassins_vireo", "Cassin's vireo"),
    ("american_robin", "Robin"),
]

ROWS = [
    ("SongMAE (16x1)", "xcl_base_500k_p16x1_default"),
    ("SongMAE (32x4)", "xcl_base_500k_p32x4_default"),
    ("SongMAE (16x1, random init)", "songmae_random"),
    ("BirdAVES", "aves"),
    ("HuBERT", "hubert"),
]


def load_runs(root):
    runs = {}
    for path in sorted(root.glob("*/*/*/metrics.json")):
        species, bird, model = path.parts[-4:-1]
        data = json.loads(path.read_text())
        runs.setdefault((species, model), []).append((bird, float(data["macro_f1"]), float(data["fer"])))
    return runs


def average(values):
    if not values:
        return None
    return float(np.asarray(values, dtype=np.float64).mean())


def species_cell(runs, species, model):
    rows = runs.get((species, model), [])
    if not rows:
        return None
    return average([row[1] for row in rows]), average([row[2] for row in rows])


def row_values(runs, model):
    values = [species_cell(runs, species, model) for species, _ in SPECIES]
    f1 = average([value[0] for value in values if value is not None])
    fer = average([value[1] for value in values if value is not None])
    return values + ([None] if f1 is None else [(f1, fer)])


def format_cell(value):
    if value is None:
        return "- / -"
    return f"{100.0 * value[0]:.2f} / {100.0 * value[1]:.2f}"


def print_rows(rows, tsv):
    headers = ["Model"] + [label for _, label in SPECIES] + ["Mean"]
    sep = "\t" if tsv else " | "
    if tsv:
        print(sep.join(headers))
    else:
        print("| " + sep.join(headers) + " |")
        print("| " + sep.join(["---"] * len(headers)) + " |")
    for label, values in rows:
        cells = [label] + [format_cell(value) for value in values]
        print(sep.join(cells) if tsv else "| " + sep.join(cells) + " |")


def main():
    parser = argparse.ArgumentParser(description="Aggregate SongMAE-vs-other syllable linear probe results.")
    parser.add_argument("--results_root", default="results/syllable_linear_probe")
    parser.add_argument("--format", choices=["markdown", "tsv"], default="markdown")
    args = parser.parse_args()

    runs = load_runs(Path(args.results_root))
    rows = [(label, row_values(runs, model)) for label, model in ROWS]
    print_rows(rows, args.format == "tsv")


if __name__ == "__main__":
    main()
