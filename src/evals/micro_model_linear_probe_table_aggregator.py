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

PATCH_ROWS = [
    ("128x1", "xcl_micro_500k_p128x1_default"),
    ("16x1", "xcl_micro_500k_p16x1_default"),
    ("32x1", "xcl_micro_500k_p32x1_default"),
    ("32x4", "xcl_micro_500k_p32x4_default"),
    ("4x4", "xcl_micro_500k_p4x4_default"),
]

EMPTY_ROWS = [
    ("Masking (32x1)", None),
    ("Random", None),
    ("Voronoi", None),
    ("Voronoi C parameter (best fine shape)", None),
    ("C=0.05", None),
    ("C=0.1", None),
    ("C=0.2", None),
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
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean())


def species_cell(runs, species, model):
    rows = runs.get((species, model), [])
    if not rows:
        return None
    return average([row[1] for row in rows]), average([row[2] for row in rows])


def format_cell(value):
    if value is None:
        return "- / -"
    return f"{100.0 * value[0]:.2f} / {100.0 * value[1]:.2f}"


def row_values(runs, model):
    values = [species_cell(runs, species, model) for species, _ in SPECIES]
    f1 = average([value[0] for value in values if value is not None])
    fer = average([value[1] for value in values if value is not None])
    mean = None if f1 is None else (f1, fer)
    return values + [mean]


def table_rows(runs):
    rows = [("Patch shape (Voronoi, C=0.1)", [None] * (len(SPECIES) + 1))]
    for label, model in PATCH_ROWS:
        rows.append((label, row_values(runs, model)))
    for label, _ in EMPTY_ROWS:
        rows.append((label, [None] * (len(SPECIES) + 1)))
    return rows


def print_markdown(rows):
    headers = ["Config"] + [label for _, label in SPECIES] + ["Mean"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for label, values in rows:
        print("| " + " | ".join([label] + [format_cell(value) for value in values]) + " |")


def print_tsv(rows):
    headers = ["Config"] + [label for _, label in SPECIES] + ["Mean"]
    print("\t".join(headers))
    for label, values in rows:
        print("\t".join([label] + [format_cell(value) for value in values]))


def main():
    parser = argparse.ArgumentParser(description="Aggregate micro-model syllable linear probe results.")
    parser.add_argument("--results_root", default="results/syllable_linear_probe")
    parser.add_argument("--format", choices=["markdown", "tsv"], default="markdown")
    args = parser.parse_args()

    rows = table_rows(load_runs(Path(args.results_root)))
    if args.format == "markdown":
        print_markdown(rows)
    else:
        print_tsv(rows)


if __name__ == "__main__":
    main()
