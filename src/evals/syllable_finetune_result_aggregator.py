#!/usr/bin/env python3
"""Write per-bird and aggregate macro-FER tables for fixed finetuning runs."""

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


LABELS = {
    "xcl_base_100k_p32x1_c005": "SongMAE base 32x1 C=.05",
    "xcl_base_100k_p32x4_c010": "SongMAE base 32x4 C=.1",
    "birdaves_biox_base": "BirdAVES",
    "hubert_base_ls960": "HuBERT",
}
BUDGETS = {budget: index for index, budget in enumerate(("32", "64", "128", "256", "512", "MAX"))}
ERRORS = ("macro_fer", "macro_parsing_error", "macro_identity_error")


def write(path, fieldnames, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--allow_partial_species", action="store_true")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.rglob("metrics.json")):
        species, bird, model, _, train_dir = path.relative_to(args.root).parts[:5]
        budget = train_dir.removeprefix("train_").removesuffix("s")
        data = json.loads(path.read_text())
        rates = macro_fer_breakdown(data["class_labels"], data["confusion_matrix"])
        assert abs(rates["macro_fer"] - data["macro_fer"]) < 1e-12, path
        label = LABELS.get(model, model)
        rows.append(
            {
                "species": species,
                "bird": bird,
                "model": model,
                "model_label": label,
                "budget": budget,
                **{key: 100 * rates[key] for key in ERRORS},
                "macro_f1": 100 * data["macro_f1"],
                "fer": 100 * data["fer"],
                "train_seconds": data["train_seconds"],
                "encoder_lr": data["selected_encoder_lr"],
                "epochs": data["selected_epoch"],
            }
        )

    fields = list(rows[0])
    write(args.root / "per_bird.csv", fields, rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["species"], row["model"], row["budget"])].append(row)
    summary = [
        {
            "species": species,
            "model": model,
            "model_label": LABELS.get(model, model),
            "budget": budget,
            "n_birds": len(values),
            **{f"mean_{key}": mean(row[key] for row in values) for key in ERRORS},
        }
        for (species, model, budget), values in sorted(
            grouped.items(), key=lambda item: (item[0][0], item[0][1], BUDGETS[item[0][2]])
        )
    ]
    write(args.root / "summary.csv", list(summary[0]), summary)
    equal_species = []
    for model in sorted({row["model"] for row in rows}):
        for budget in BUDGETS:
            species_rows = [
                values
                for (species, name, tier), values in grouped.items()
                if name == model and tier == budget
            ]
            assert species_rows and (args.allow_partial_species or len(species_rows) == 3)
            equal_species.append(
                {
                    "model": model,
                    "model_label": LABELS.get(model, model),
                    "budget": budget,
                    **{
                        f"mean_{key}": mean(mean(row[key] for row in values) for values in species_rows)
                        for key in ERRORS
                    },
                }
            )
    write(args.root / "equal_species.csv", list(equal_species[0]), equal_species)
    print(
        *(
            f'{row["species"]}\t{row["model_label"]}\t{row["budget"]}\t{row["n_birds"]}\t'
            f'{row["mean_macro_fer"]:.2f} '
            f'({row["mean_macro_parsing_error"]:.2f}/{row["mean_macro_identity_error"]:.2f})'
            for row in summary
        ),
        sep="\n",
    )


if __name__ == "__main__":
    main()
