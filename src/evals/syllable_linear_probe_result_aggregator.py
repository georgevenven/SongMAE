#!/usr/bin/env python3
"""Write per-bird and equal-species macro-FER tables for linear probes."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.evals.syllable_metrics import macro_fer_breakdown


ERRORS = ("macro_fer", "macro_parsing_error", "macro_identity_error")


def write(path, fieldnames, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.glob("*/*/*/metrics.json")):
        species, bird, model = path.relative_to(args.root).parts[:3]
        data = json.loads(path.read_text())
        rates = macro_fer_breakdown(data["class_labels"], data["confusion_matrix"])
        assert abs(rates["macro_fer"] - data["macro_fer"]) < 1e-12, path
        rows.append({"species": species, "bird": bird, "model": model, **{key: 100 * rates[key] for key in ERRORS}})
    assert rows
    write(args.root / "per_bird.csv", list(rows[0]), rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["species"], row["model"])].append(row)
    summary = [
        {
            "species": species,
            "model": model,
            "n_birds": len(values),
            **{f"mean_{key}": mean(row[key] for row in values) for key in ERRORS},
        }
        for (species, model), values in sorted(grouped.items())
    ]
    write(args.root / "summary.csv", list(summary[0]), summary)
    equal_species = []
    for model in sorted({row["model"] for row in rows}):
        species_rows = [values for (species, name), values in grouped.items() if name == model]
        assert len(species_rows) == 3
        equal_species.append(
            {
                "model": model,
                **{
                    f"mean_{key}": mean(mean(row[key] for row in values) for values in species_rows)
                    for key in ERRORS
                },
            }
        )
    write(args.root / "equal_species.csv", list(equal_species[0]), equal_species)
    print(
        *(
            f'{row["model"]}\t{row["mean_macro_fer"]:.2f} '
            f'({row["mean_macro_parsing_error"]:.2f}/{row["mean_macro_identity_error"]:.2f})'
            for row in equal_species
        ),
        sep="\n",
    )


if __name__ == "__main__":
    main()
