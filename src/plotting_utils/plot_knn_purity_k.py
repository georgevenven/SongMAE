#!/usr/bin/env python3
"""Plot species-balanced kNN purity at each encoder's selected layer."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = {
    "SongMAE 32x1": ("SongMAE-Large 32 mels × 5 ms", "#0072B2"),
    "SongMAE 32x4": ("SongMAE-Large 32 mels × 20 ms", "#56B4E9"),
    "BirdAVES": ("BirdAVES", "#D55E00"),
    "HuBERT": ("HuBERT", "#009E73"),
}
KS = [1, 5, 10, 50, 100]


def load(path):
    rows = list(csv.DictReader(Path(path).open(), delimiter="\t"))
    values = {(row["model"], int(row["k"])): float(row["species_macro"]) for row in rows}
    assert set(values) == {(model, k) for model in MODELS for k in KS}
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tsv")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    values = load(args.input_tsv)
    fig, axis = plt.subplots(figsize=(8.4, 4.8), dpi=200)
    for model, (label, color) in MODELS.items():
        axis.plot(
            KS, [values[model, k] for k in KS], "o-", color=color,
            linewidth=2.25, markersize=6, label=label,
        )
    axis.set_xlabel("k", fontsize=13)
    axis.set_ylabel("Macro kNN purity (%) ↑", fontsize=13)
    axis.set_xticks(KS)
    axis.tick_params(labelsize=11)
    axis.grid(alpha=0.18)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    fig.tight_layout()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "knn_all_k.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "knn_all_k_hq.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "knn_all_k.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
