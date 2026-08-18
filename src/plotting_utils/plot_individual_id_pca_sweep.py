#!/usr/bin/env python3
"""Plot the SongMAE 32x4 pink-noise PCA sweep."""
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "Individual_Id_paper_materials" / "results"
OUTPUT = ROOT / "Individual_Id_paper_materials" / "figures"
MODEL = "xcl_large_500k_p32x4_c010"
DIMS = (64, 128, 256, 512, 1024)
SPECIES = {
    "zebra_finch": ("Zebra Finch", "#0072B2"),
    "canary": ("Canary", "#D55E00"),
    "bengalese_finch": ("Bengalese Finch", "#009E73"),
}


def result_path(species, dim):
    if dim == 128:
        return RESULTS / "logistic" / "token" / "pink_0db" / species / MODEL / "layer_11" / "metrics.json"
    if dim == 1024:
        root = RESULTS / "pca_comparison"
    else:
        root = RESULTS / "pca_sweep"
    return root / "pink_0db" / "token" / species / MODEL / "layer_11" / f"pca_{dim}" / "metrics.json"


def main():
    values = {}
    for species in SPECIES:
        for dim in DIMS:
            data = json.loads(result_path(species, dim).read_text())
            assert data["pca_components"] == dim
            values[species, dim] = 100 * data["methods"]["token"]["macro_f1"]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "songmae_32x4_pca_sweep_pink_0db.tsv").open("w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(("pca_components", *SPECIES, "mean", "sem"))
        for dim in DIMS:
            row = [values[species, dim] for species in SPECIES]
            writer.writerow((dim, *[f"{value:.6f}" for value in row], f"{np.mean(row):.6f}", f"{np.std(row, ddof=1) / np.sqrt(len(row)):.6f}"))

    fig, axis = plt.subplots(figsize=(6.3, 4.0), dpi=200)
    for species, (label, color) in SPECIES.items():
        axis.plot(DIMS, [values[species, dim] for dim in DIMS], "o-", color=color, linewidth=1.6, markersize=5, alpha=0.8, label=label)
    means = [np.mean([values[species, dim] for species in SPECIES]) for dim in DIMS]
    axis.plot(DIMS, means, "o-", color="#202020", linewidth=3, markersize=6, label="Species mean")
    axis.set_xscale("log", base=2)
    axis.set_xticks(DIMS, [str(dim) for dim in DIMS])
    axis.set_ylim(25, 100)
    axis.set_xlabel("PCA components", fontsize=12)
    axis.set_ylabel("Token linear probe Macro F1 (%) ↑", fontsize=12)
    axis.set_title("SongMAE-Large (32 mels × 20 ms) · L11\nPink noise (0 dB)", fontsize=13)
    axis.grid(alpha=0.18)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    out = OUTPUT / "figure_4_songmae_pca_sweep"
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_name(out.name + "_hq.png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
