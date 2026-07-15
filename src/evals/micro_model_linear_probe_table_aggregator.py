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

TITLE = "Linear Probe Ablations"

SECTIONS = [
    (
        "Masking strategy (32x1; Voronoi C=0.1)",
        [
            ("Random", "xcl_micro_100k_p32x1_random"),
            ("Voronoi", "Xcl_micro_100k_p32x1_default"),
        ],
    ),
    (
        "Patch shape (Voronoi, C=0.1)",
        [
            ("128x1", "Xcl_micro_100k_p128x1_default"),
            ("32x1", "Xcl_micro_100k_p32x1_default"),
            ("16x1", "Xcl_micro_100k_p16x1_default"),
            ("32x4", "xcl_micro_100k_p32x4_qknorm_gelu_lr1e-4_bs128"),
            ("4x4", "Xcl_micro_100k_p4x4_default"),
        ],
    ),
    (
        "Voronoi C parameter (32x1)",
        [
            ("C=0.025", "Xcl_micro_100k_p32x1_c0025"),
            ("C=0.05", "Xcl_micro_100k_p32x1_c005"),
            ("C=0.1", "Xcl_micro_100k_p32x1_c010"),
            ("C=0.2", "Xcl_micro_100k_p32x1_c020"),
        ],
    ),
    (
        "Voronoi C parameter (32x4)",
        [
            ("C=0.025", "xcl_micro_100k_p32x4_c0025"),
            ("C=0.05", "xcl_micro_100k_p32x4_c005"),
            ("C=0.1", "xcl_micro_100k_p32x4_c010"),
            ("C=0.2", "xcl_micro_100k_p32x4_c020"),
        ],
    ),
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


def format_cell(value):
    if value is None:
        return "-"
    return (
        f'{100.0 * value["macro_fer"]:.2f} '
        f'({100.0 * value["macro_parsing_error"]:.2f}/{100.0 * value["macro_identity_error"]:.2f})'
    )


def row_values(runs, model):
    values = [species_cell(runs, species, model) for species, _ in SPECIES]
    return values + [average([value for value in values if value is not None])]


def print_markdown(runs):
    headers = ["Config"] + [f"{label} Macro FER (P/I)" for _, label in SPECIES] + ["Mean Macro FER (P/I)"]
    section = ["-"] * (len(headers) - 1)
    print(TITLE)
    print()
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for label, rows in SECTIONS:
        print("| " + " | ".join([label] + section) + " |")
        for row_label, model in rows:
            values = [format_cell(value) for value in row_values(runs, model)]
            print("| " + " | ".join([row_label] + values) + " |")


def print_tsv(runs):
    headers = ["Config"] + [f"{label} Macro FER (P/I)" for _, label in SPECIES] + ["Mean Macro FER (P/I)"]
    section = ["-"] * (len(headers) - 1)
    print("\t".join([TITLE] + section))
    print("\t".join(headers))
    for label, rows in SECTIONS:
        print("\t".join([label] + section))
        for row_label, model in rows:
            values = [format_cell(value) for value in row_values(runs, model)]
            print("\t".join([row_label] + values))


def main():
    parser = argparse.ArgumentParser(description="Aggregate micro-model syllable linear probe results.")
    parser.add_argument("--results_root", default="results/syllable_linear_probe")
    parser.add_argument("--format", choices=["tsv", "markdown"], default="tsv")
    args = parser.parse_args()

    runs = load_runs(Path(args.results_root))
    if args.format == "tsv":
        print_tsv(runs)
    else:
        print_markdown(runs)


if __name__ == "__main__":
    main()
