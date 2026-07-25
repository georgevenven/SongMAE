#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


FIXED = [
    ("SSL", "SongMAE-Large 32×4", 0.7709, 0.389),
    ("SSL", "SongMAE-Large 32×1", 0.781, 0.381),
    ("SSL", "BEATs pretrained", 0.774, 0.381),
    ("SSL", "EAT-base pretrained", 0.679, 0.252),
    ("SSL", "EAT-all", 0.709, 0.315),
    ("SSL", "Bird-BirdAVES-biox-base", 0.705, 0.34),
    ("SSL", "Bird-MAE-Huge", 0.766, 0.354),
    ("SL reference", "BirdNet", 0.796, 0.392),
]
TRAINED = [
    ("SongMAE-Base 32×4", "base_p32x4"),
    ("SongMAE-Base 32×1", "base_p32x1"),
    ("SongMAE-Micro 32×4", "micro_p32x4"),
    ("SongMAE-Micro 32×1", "micro_p32x1"),
]
PARAMETERS = {
    # SongMAE counts are exact; external counts are reported model sizes.
    "SongMAE-Large 32×4": 98_167_553,
    "SongMAE-Large 32×1": 98_645_249,
    "SongMAE-Base 32×4": 14_656_705,
    "SongMAE-Base 32×1": 14_889_409,
    "SongMAE-Micro 32×4": 1_674_305,
    "SongMAE-Micro 32×1": 1_751_873,
    "BEATs pretrained": 90_000_000,
    "EAT-base pretrained": 90_000_000,
    "EAT-all": 90_000_000,
    "Bird-BirdAVES-biox-base": 94_370_944,
    "Bird-MAE-Huge": 630_342_400,
    "BirdNet": 12_625_000,
}


def mean(path, key, expected):
    with path.open() as file:
        values = [float(row[key]) for row in csv.DictReader(file)]
    assert len(values) == expected, path
    return sum(values) / len(values)


def number(value):
    return f"{value:.4f}".rstrip("0").rstrip(".")


def table_rows(root):
    trained = [
        (
            "SSL",
            label,
            mean(root / slug / "classification.csv", "test_accuracy", 6),
            mean(root / slug / "detection.csv", "test_mAP", 5),
        )
        for label, slug in TRAINED
    ]
    return FIXED[:2] + trained + FIXED[2:]


def main():
    parser = argparse.ArgumentParser(description="Aggregate the BEANS probe table.")
    parser.add_argument(
        "--results_root",
        default="/media/george-vengrovski/disk1/avex_runs/"
        "songmae_beans_micro_base_500k/results",
    )
    args = parser.parse_args()
    print(
        "Training\tModel\tParameters (M)\t"
        "Probe Accuracy Classification\tProbe Detection mAP"
    )
    for training, model, classification, detection in table_rows(Path(args.results_root)):
        print(
            "\t".join(
                [
                    training,
                    model,
                    number(PARAMETERS[model] / 1e6),
                    number(classification),
                    number(detection),
                ]
            )
        )


if __name__ == "__main__":
    main()
