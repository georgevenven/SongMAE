#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.evals.linear_probe_table_utils import load_capped_runs, print_tables, value


STEPS = ["000000", "010000", "050000", "499999"]
SIZES = [
    ("SongMAE Large 32x4", "large"),
    ("SongMAE Base 32x4", "base"),
    ("SongMAE Micro 32x4", "micro"),
]


def main():
    parser = argparse.ArgumentParser(description="Aggregate K=5 32x4 checkpoint linear probes.")
    parser.add_argument("--results_root", default="results/linear_probe_32x4_checkpoints_k5")
    parser.add_argument("--format", choices=["tsv", "markdown"], default="tsv")
    args = parser.parse_args()
    runs = load_capped_runs(args.results_root)
    print_tables(
        "K=5 32x4 Checkpoint Linear Probes",
        [("500k" if step == "499999" else f"{int(step) // 1000}k", step) for step in STEPS],
        SIZES,
        lambda species, size, step: value(runs, species, f"{size}_step_{step}", 5),
        args.format == "markdown",
    )


if __name__ == "__main__":
    main()
