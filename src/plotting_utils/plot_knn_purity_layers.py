#!/usr/bin/env python3
"""Plot equal-species mean kNN purity across encoder layers."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


MODELS = {
    "xcl_large_500k_p32x1_c005": ("SongMAE-Large\n(32 mels × 5 ms)", "#0072B2"),
    "xcl_large_500k_p32x4_c010": ("SongMAE-Large\n(32 mels × 20 ms)", "#56B4E9"),
    "birdaves_biox_base": ("BirdAVES", "#D55E00"),
    "hubert_base_ls960": ("HuBERT", "#009E73"),
}
LAYERS = range(12)
SPECIES = ("bf", "canary", "zf")


def load_purity(root, k):
    values = defaultdict(list)
    for path in root.rglob("summary.json"):
        species, _, model, layer_name = path.relative_to(root).parts[:4]
        row = next(row for row in json.loads(path.read_text())["rows"] if row["k"] == k)
        values[model, int(layer_name.removeprefix("layer_")), species].append(
            100 * row["macro_same_purity"]
        )
    assert all(
        sum(len(values[model, layer, species]) for species in SPECIES) == 50
        for model in MODELS
        for layer in LAYERS
    )
    means = {
        (model, layer): sum(
            sum(values[model, layer, species]) / len(values[model, layer, species])
            for species in SPECIES
        ) / len(SPECIES)
        for model in MODELS
        for layer in LAYERS
    }
    return means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default="results/knn/four_models_all_layers_raw")
    parser.add_argument("--output_dir", default="imgs/knn_purity")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--overlay", action="store_true")
    args = parser.parse_args()
    means = load_purity(Path(args.results_root), args.k)
    limits = max(0, min(means.values()) - 5), min(100, max(means.values()) + 5)

    columns = 1 if args.overlay else 4
    fig, axes = plt.subplots(
        1, columns, figsize=(4.8 if args.overlay else 9.6, 2.9), dpi=200, sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    plot_axes = [axes[0]] * len(MODELS) if args.overlay else axes
    for axis, (model, (label, color)) in zip(plot_axes, MODELS.items()):
        purity = [means[model, layer] for layer in LAYERS]
        axis.plot(
            LAYERS, purity, color=color, marker="o", markersize=4, linewidth=2.25,
            label=label,
        )
        if not args.overlay:
            axis.set_title(label, fontsize=13)
        axis.set_xticks(LAYERS)
        axis.set_xticklabels([str(layer) if layer % 2 == 0 else "" for layer in LAYERS])
        axis.set_ylim(*limits)
        if not args.overlay:
            axis.set_box_aspect(1)
        axis.grid(alpha=0.18)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=11)
        best_layer = max(LAYERS, key=lambda layer: means[model, layer])
        print(f"{label}\tlayer {best_layer}\t{means[model, best_layer]:.3f}%")

    axes[0].set_ylabel("Macro kNN purity (%) ↑", fontsize=13)
    if args.overlay:
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="upper center",
            ncol=len(MODELS),
            frameon=False,
        )
    fig.supxlabel("Encoder layer", y=0.035, fontsize=13)
    if args.overlay:
        fig.subplots_adjust(left=0.15, right=0.995, bottom=0.23, top=0.86)
    else:
        fig.subplots_adjust(left=0.075, right=0.995, bottom=0.23, top=0.86, wspace=0.02)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_overlay" if args.overlay else ""
    output = output_dir / f"layer_purity_k{args.k}{suffix}.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output)
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
