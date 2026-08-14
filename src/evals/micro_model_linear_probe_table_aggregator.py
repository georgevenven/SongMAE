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
        "Masking strategy (32×1; Voronoi C=0.1)",
        [
            ("Random", "xcl_micro_100k_p32x1_random"),
            ("Voronoi", "Xcl_micro_100k_p32x1_c010"),
        ],
    ),
    (
        "Patch shape (Voronoi, C=0.1)",
        [
            ("128×1", "Xcl_micro_100k_p128x1_default"),
            ("32×1", "Xcl_micro_100k_p32x1_c010"),
            ("16×1", "Xcl_micro_100k_p16x1_default"),
            ("32×4", "xcl_micro_100k_p32x4_c010"),
            ("4×4", "Xcl_micro_100k_p4x4_default"),
        ],
    ),
    (
        "Voronoi C parameter (32×1)",
        [
            ("C=0.025", "Xcl_micro_100k_p32x1_c0025"),
            ("C=0.05", "Xcl_micro_100k_p32x1_c005"),
            ("C=0.1", "Xcl_micro_100k_p32x1_c010"),
        ],
    ),
    (
        "Voronoi C parameter (32×4)",
        [
            ("C=0.025", "xcl_micro_100k_p32x4_c0025"),
            ("C=0.05", "xcl_micro_100k_p32x4_c005"),
            ("C=0.1", "xcl_micro_100k_p32x4_c010"),
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


def format_score(value):
    if value is None:
        return "-"
    return f'{100.0 * value["macro_fer"]:.2f}'


def breakdown_values(value):
    if value is None:
        return ["-", "-", "-"]
    return [
        f'{100.0 * value["macro_fer"]:.2f}',
        f'{100.0 * value["macro_parsing_error"]:.2f}',
        f'{100.0 * value["macro_identity_error"]:.2f}',
    ]


def row_values(runs, model):
    values = [species_cell(runs, species, model) for species, _ in SPECIES]
    mean = average([value for value in values if value is not None])
    return [format_score(value) for value in values] + breakdown_values(mean)


def config_cells(section, row_label):
    if section.startswith("Masking strategy"):
        return [row_label, "32×1", "0.100"]
    if section.startswith("Patch shape"):
        return ["Voronoi", row_label, "0.100"]
    patch = "32×1" if "32×1" in section else "32×4"
    return ["Voronoi", patch, f'{float(row_label.removeprefix("C=")):.3f}']


def print_markdown(runs):
    headers = ["Masking", "Patch", "C"]
    headers += [label for _, label in SPECIES] + ["Mean", "Parsing", "Identity"]
    print(TITLE)
    print()
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for section, rows in SECTIONS:
        for row_label, model in rows:
            print("| " + " | ".join(config_cells(section, row_label) + row_values(runs, model)) + " |")
        if section != SECTIONS[-1][0]:
            print()


def print_tsv(runs):
    headers = ["Masking", "Patch", "C"]
    headers += [label for _, label in SPECIES] + ["Mean", "Parsing", "Identity"]
    print("\t".join([TITLE]))
    print("\t".join(headers))
    for section, rows in SECTIONS:
        for row_label, model in rows:
            print("\t".join(config_cells(section, row_label) + row_values(runs, model)))


def main():
    parser = argparse.ArgumentParser(description="Aggregate micro-model syllable linear probe results.")
    parser.add_argument(
        "--results_root",
        default="results/linear_probe_micro_ablations_kmax_pca128_logreg_c0001",
    )
    parser.add_argument("--format", choices=["tsv", "markdown"], default="tsv")
    args = parser.parse_args()

    runs = load_runs(Path(args.results_root))
    if args.format == "tsv":
        print_tsv(runs)
    else:
        print_markdown(runs)


if __name__ == "__main__":
    main()
