#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.evals.syllable_metrics import macro_fer_breakdown


SPECIES = [
    ("canary", "Canary"),
    ("zf", "Zebra"),
    ("bf", "Bengalese"),
]

ROWS = [
    ("SongMAE-Large 32×1", "xcl_large_500k_p32x1_c005"),
    ("SongMAE-Large 32×4", "xcl_large_500k_p32x4_c010"),
    ("BirdAVES", "birdaves_biox_base"),
    ("HuBERT", "hubert_base_ls960"),
]


def load_runs(root):
    runs = {}
    for path in sorted(root.glob("*/*/*/metrics.json")):
        species, _, model = path.parts[-4:-1]
        data = json.loads(path.read_text())
        rates = macro_fer_breakdown(data["class_labels"], data["confusion_matrix"])
        assert abs(rates["macro_fer"] - data["macro_fer"]) < 1e-12, path
        runs.setdefault((species, model), []).append(rates)
    return runs


def average(values):
    if not values:
        return None
    return {key: sum(value[key] for value in values) / len(values) for key in values[0]}


def species_cell(runs, species, model):
    return average(runs.get((species, model), []))


def row_values(runs, model):
    values = [species_cell(runs, species, model) for species, _ in SPECIES]
    return values + [average([value for value in values if value is not None])]


def format_cell(value):
    if value is None:
        return "-"
    return f'{100.0 * value["macro_fer"]:.2f}'


def breakdown_cells(value):
    if value is None:
        return ["-", "-", "-"]
    return [
        f'{100.0 * value["macro_fer"]:.2f}',
        f'{100.0 * value["macro_parsing_error"]:.2f}',
        f'{100.0 * value["macro_identity_error"]:.2f}',
    ]


def print_rows(rows, tsv):
    headers = ["Model", "Canary", "Zebra", "Bengalese", "Mean", "Parsing", "Identity"]
    sep = "\t" if tsv else " | "
    if tsv:
        print(sep.join(headers))
    else:
        print("| " + sep.join(headers) + " |")
        print("| " + sep.join(["---"] * len(headers)) + " |")
    for label, values in rows:
        cells = [label] + [format_cell(value) for value in values[:-1]] + breakdown_cells(values[-1])
        print(sep.join(cells) if tsv else "| " + sep.join(cells) + " |")


def main():
    parser = argparse.ArgumentParser(description="Aggregate SongMAE-vs-other syllable linear probe results.")
    parser.add_argument(
        "--results_root", default="results/linear_probe_models_best_layers_kmax_pca128_logreg_c0001"
    )
    parser.add_argument("--format", choices=["tsv", "markdown"], default="tsv")
    args = parser.parse_args()

    runs = load_runs(Path(args.results_root))
    rows = [(label, row_values(runs, model)) for label, model in ROWS]
    print_rows(rows, args.format == "tsv")


if __name__ == "__main__":
    main()
