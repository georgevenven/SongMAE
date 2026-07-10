#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


SPECIES = [
    ("canary", "Canary"),
    ("zf", "Zebra"),
    ("bf", "Bengalese"),
]

ROWS = [
    ("Base 16x4", "xcl_base_100k_p16x4_c010"),
    ("Base 32x1", "xcl_base_100k_p32x1_c010"),
    ("BirdAVES", "birdaves_biox_base"),
    ("HuBERT base", "hubert_base_ls960"),
]


def load_runs(root):
    runs = {}
    for path in sorted(root.glob("*/*/*/metrics.json")):
        species, _, model = path.parts[-4:-1]
        data = json.loads(path.read_text())
        runs.setdefault((species, model), []).append(float(data["macro_fer"]))
    return runs


def average(values):
    if not values:
        return None
    return sum(values) / len(values)


def species_cell(runs, species, model):
    return average(runs.get((species, model), []))


def row_values(runs, model):
    values = [species_cell(runs, species, model) for species, _ in SPECIES]
    return values + [average([value for value in values if value is not None])]


def format_cell(value):
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}"


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
    parser.add_argument("--format", choices=["tsv", "markdown"], default="tsv")
    args = parser.parse_args()

    runs = load_runs(Path(args.results_root))
    rows = [(label, row_values(runs, model)) for label, model in ROWS]
    print_rows(rows, args.format == "tsv")


if __name__ == "__main__":
    main()
