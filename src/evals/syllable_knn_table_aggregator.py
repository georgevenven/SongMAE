#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SPECIES = [("canary", "Canary"), ("zf", "Zebra"), ("bf", "Bengalese")]
METRICS = ["macro_same_purity", "vocal_macro_same_purity", "silence_same_purity"]
PATCH_LABELS = {
    "128×1": "128 mels × 5 ms",
    "32×1": "32 mels × 5 ms",
    "16×1": "16 mels × 5 ms",
    "32×4": "32 mels × 20 ms",
    "4×4": "4 mels × 20 ms",
}
SECTIONS = [
    ("Masking (32×1, 10% seed patches)", [
        ("Random", "xcl_micro_100k_p32x1_random"),
        ("Voronoi", "Xcl_micro_100k_p32x1_c010"),
    ]),
    ("Patch shape (Voronoi, 10% seed patches)", [
        ("128×1", "Xcl_micro_100k_p128x1_default"),
        ("32×1", "Xcl_micro_100k_p32x1_c010"),
        ("16×1", "Xcl_micro_100k_p16x1_default"),
        ("32×4", "xcl_micro_100k_p32x4_c010"),
        ("4×4", "Xcl_micro_100k_p4x4_default"),
    ]),
    ("Seed-patch sweep (32×1)", [
        ("2.5%", "Xcl_micro_100k_p32x1_c0025"),
        ("5%", "Xcl_micro_100k_p32x1_c005"),
        ("10%", "Xcl_micro_100k_p32x1_c010"),
    ]),
    ("Seed-patch sweep (32×4)", [
        ("2.5%", "xcl_micro_100k_p32x4_c0025"),
        ("5%", "xcl_micro_100k_p32x4_c005"),
        ("10%", "xcl_micro_100k_p32x4_c010"),
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


def macro_purity(runs, name, k):
    values = [average(runs[species, name, k]) for species, _ in SPECIES]
    values = [value[0] for value in values if value is not None]
    return 100 * sum(values) / len(values)


def plot_sweep(runs, output):
    ks = [1, 5, 10, 50, 100]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8), sharex=True, sharey=True)
    for axis, (title, configs) in zip(axes.flat, SECTIONS):
        for label, name in configs:
            values = [macro_purity(runs, name, k) for k in ks]
            axis.plot(ks, values, "o-", label=PATCH_LABELS.get(label, label))
        axis.set_title(title)
        axis.set_xticks(ks)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.supylabel("Macro same-label purity (%)")
    fig.supxlabel("k")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output)


def main():
    parser = argparse.ArgumentParser(description="Aggregate occurrence-level kNN purity.")
    parser.add_argument(
        "--results_root",
        default="results/knn/micro_ablations_all_k_manuscript/raw",
    )
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--format", choices=["tsv", "markdown"], default="tsv")
    parser.add_argument("--plot_output")
    args = parser.parse_args()
    runs = load_runs(Path(args.results_root))
    if args.plot_output:
        plot_sweep(runs, args.plot_output)
        return
    title = f"kNN Purity Ablations (k={args.k})"
    header = ["Config", *[label for _, label in SPECIES], "Mean (vocal/silence)"]
    output = list(rows(runs, args.k))
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
