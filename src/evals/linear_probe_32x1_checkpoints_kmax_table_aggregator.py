#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.evals.linear_probe_table_utils import full_value, load_full_runs, print_tables


STEPS = ["000000", "010000", "050000", "499999"]
SIZES = [
    ("SongMAE Large 32x1", "large"),
    ("SongMAE Base 32x1", "base"),
    ("SongMAE Micro 32x1", "micro"),
]


def main():
    parser = argparse.ArgumentParser(description="Aggregate K=max 32x1 checkpoint probes.")
    parser.add_argument(
        "--results_root",
        default="results/linear_probe_32x1_checkpoints_kmax_pca128_logreg_c0001",
    )
    parser.add_argument("--format", choices=["tsv", "markdown"], default="tsv")
    args = parser.parse_args()
    runs = load_full_runs(args.results_root)
    print_tables(
        "K=max 32x1 Checkpoint Linear Probes",
        [("500k" if step == "499999" else f"{int(step) // 1000}k", step) for step in STEPS],
        SIZES,
        lambda species, size, step: full_value(runs, species, f"{size}_step_{step}"),
        args.format == "markdown",
    )


if __name__ == "__main__":
    main()
