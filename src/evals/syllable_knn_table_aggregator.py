#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


SPECIES = [("canary", "Canary"), ("zf", "Zebra"), ("bf", "Bengalese")]
METRICS = ["macro_same_purity", "vocal_macro_same_purity", "silence_same_purity"]
SECTIONS = [
    ("Masking (32×1, C=0.1)", [
        ("Random", "xcl_micro_100k_p32x1_random"),
        ("Voronoi", "Xcl_micro_100k_p32x1_c010"),
    ]),
    ("Patch shape (Voronoi, C=0.1)", [
        ("128×1", "Xcl_micro_100k_p128x1_default"),
        ("32×1", "Xcl_micro_100k_p32x1_c010"),
        ("16×1", "Xcl_micro_100k_p16x1_default"),
        ("32×4", "xcl_micro_100k_p32x4_c010"),
        ("4×4", "Xcl_micro_100k_p4x4_default"),
    ]),
    ("C sweep (32×1)", [
        ("C=0.025", "Xcl_micro_100k_p32x1_c0025"),
        ("C=0.05", "Xcl_micro_100k_p32x1_c005"),
        ("C=0.1", "Xcl_micro_100k_p32x1_c010"),
    ]),
    ("C sweep (32×4)", [
        ("C=0.025", "xcl_micro_100k_p32x4_c0025"),
        ("C=0.05", "xcl_micro_100k_p32x4_c005"),
        ("C=0.1", "xcl_micro_100k_p32x4_c010"),
    ]),
]


def load_runs(root):
    runs = defaultdict(list)
    for path in root.glob("*/*/*/layer_*/end_of_block/summary.json"):
        species = path.relative_to(root).parts[0]
        data = json.loads(path.read_text())
        for row in data["rows"]:
            runs[species, data["name"], row["k"]].append(row)
    return runs


def average(rows):
    if not rows:
        return None
    return [sum(row[metric] for row in rows) / len(rows) for metric in METRICS]


def percent(value):
    return f"{100 * value:.1f}%"


def cell(values):
    return "-" if values is None else f"{percent(values[0])} ({percent(values[1])}/{percent(values[2])})"


def result_row(runs, label, name, k):
    values = [average(runs[species, name, k]) for species, _ in SPECIES]
    present = [value for value in values if value is not None]
    mean = [sum(x[i] for x in present) / len(present) for i in range(3)]
    overall = ["-" if value is None else percent(value[0]) for value in values]
    return [label, *overall, cell(mean)]


def rows(runs, k):
    names = {name for _, name, candidate in runs if candidate == k}
    assert names, f"no k={k} results"
    for section, configs in SECTIONS:
        configs = [(label, name) for label, name in configs if name in names]
        if not configs:
            continue
        yield [section, *[""] * (len(SPECIES) + 1)]
        for label, name in configs:
            yield result_row(runs, label, name, k)


def main():
    parser = argparse.ArgumentParser(description="Aggregate occurrence-level kNN purity.")
    parser.add_argument(
        "--results_root",
        default="results/knn/micro_ablations_k50_k100_last_layer",
    )
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--format", choices=["tsv", "markdown"], default="tsv")
    args = parser.parse_args()
    title = f"kNN Purity Ablations (k={args.k})"
    header = ["Config", *[label for _, label in SPECIES], "Mean (vocal/silence)"]
    output = list(rows(load_runs(Path(args.results_root)), args.k))
    if args.format == "tsv":
        print("\t".join(header))
        for row in output:
            print("\t".join(row))
        return
    print(title)
    print()
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for row in output:
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
