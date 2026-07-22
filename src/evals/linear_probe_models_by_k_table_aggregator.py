#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.evals.linear_probe_table_utils import load_capped_runs, print_tables, value


CAPS = [1, 5, 10, 20, 50, 100]
MODELS = [
    ("SongMAE Large 32x1 (500k)", "xcl_large_500k_p32x1_c005"),
    ("SongMAE Large 32x4 (500k)", "xcl_large_500k_p32x4_c010"),
    ("BirdAVES", "birdaves_biox_base"),
    ("HuBERT base", "hubert_base_ls960"),
]


def main():
    parser = argparse.ArgumentParser(description="Aggregate linear-probe performance by label cap.")
    parser.add_argument(
        "--results_root",
        default="results/linear_probe_models_by_k_pca128_logreg_c0001",
    )
    parser.add_argument("--format", choices=["tsv", "markdown"], default="tsv")
    args = parser.parse_args()
    runs = load_capped_runs(args.results_root)
    print_tables(
        "Capped-label Linear Probes",
        [(f"K={cap} Macro FER (P/I)", cap) for cap in CAPS],
        MODELS,
        lambda species, model, cap: value(runs, species, model, cap),
        args.format == "markdown",
    )


if __name__ == "__main__":
    main()
