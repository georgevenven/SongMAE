#!/usr/bin/env python3
"""Plot the ten-species individual-ID sweep."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = {
    "xcl_large_500k_p32x1_c005": ("SongMAE-Large\n(32 mels × 5 ms)", "#0072B2"),
    "xcl_large_500k_p32x4_c010": ("SongMAE-Large\n(32 mels × 20 ms)", "#56B4E9"),
    "birdaves_biox_base": ("BirdAVES", "#D55E00"),
    "hubert_base_ls960": ("HuBERT", "#009E73"),
}
CONDITIONS = {
    "clean": ("Clean", "-", "o", None),
    "pink_0db": ("Pink noise (0 dB)", "--", "o", "white"),
}
METHODS = {"token": "Token linear probe", "centroid": "Centroid linear probe"}
LAYERS = range(12)
KS = (1, 5, 10, 50, 100)

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})


def load(results):
    species = tuple(sorted(path.stem for path in (results / "manifests").glob("*.json")))
    assert len(species) == 10
    logistic, knn = {}, {}
    for path in (results / "logistic").glob("*/*/*/*/layer_*/metrics.json"):
        method, condition, name, model, layer, _ = path.relative_to(results / "logistic").parts
        data = json.loads(path.read_text())
        logistic[model, condition, method, int(layer[-2:]), name] = 100 * data["methods"][method]["macro_f1"]
    for path in (results / "knn").glob("*/*/*/layer_*/summary.json"):
        condition, name, model, layer, _ = path.relative_to(results / "knn").parts
        for row in json.loads(path.read_text())["rows"]:
            knn[model, condition, int(layer[-2:]), row["k"], name] = 100 * row["dn4_macro_accuracy"]
    assert len(logistic) == len(MODELS) * len(CONDITIONS) * len(METHODS) * len(LAYERS) * len(species)
    assert len(knn) == len(MODELS) * len(CONDITIONS) * len(LAYERS) * len(KS) * len(species)
    return species, logistic, knn


def stats(values):
    values = np.asarray(values)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def save(fig, output):
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_name(output.name + "_hq.png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def layer_figure(output, species, logistic):
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.3), dpi=200, sharex=True, sharey=True)
    for column, (model, (label, color)) in enumerate(MODELS.items()):
        axes[0, column].set_title(label, fontsize=13)
        for row, (method, method_label) in enumerate(METHODS.items()):
            axis = axes[row, column]
            for condition, (condition_label, style, marker, face) in CONDITIONS.items():
                means = [
                    stats([logistic[model, condition, method, layer, name] for name in species])[0]
                    for layer in LAYERS
                ]
                axis.plot(
                    LAYERS, means, color=color, linestyle=style, marker=marker,
                    markerfacecolor=face or color, markeredgecolor=color,
                    markersize=4, linewidth=2.25, label=condition_label,
                )
            axis.set_xticks(LAYERS)
            axis.set_xticklabels([str(layer) if layer % 2 == 0 else "" for layer in LAYERS])
            axis.set_ylim(0, 80)
            axis.grid(alpha=0.18)
            axis.set_axisbelow(True)
            axis.tick_params(labelsize=10)
            if column == 0:
                axis.set_ylabel(f"{method_label}\nMacro F1 (%) ↑", fontsize=11)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    fig.supxlabel("Encoder layer", y=0.035, fontsize=13)
    fig.subplots_adjust(left=0.1, right=0.995, bottom=0.13, top=0.84, hspace=0.18, wspace=0.05)
    save(fig, output / "figure_1_layer_sweep")


def best_layers(species, logistic, knn):
    logistic_layers = {
        (model, method): max(
            LAYERS,
            key=lambda layer: stats([logistic[model, "clean", method, layer, name] for name in species])[0],
        )
        for model in MODELS
        for method in METHODS
    }
    knn_layers = {
        model: max(
            LAYERS,
            key=lambda layer: stats([knn[model, "clean", layer, 5, name] for name in species])[0],
        )
        for model in MODELS
    }
    return logistic_layers, knn_layers


def noise_figure(output, species, logistic, layers):
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.9), dpi=200, sharey=True)
    for axis, (model, (label, color)) in zip(axes, MODELS.items()):
        layer = layers[model, "token"]
        clean = [logistic[model, "clean", "token", layer, name] for name in species]
        pink = [logistic[model, "pink_0db", "token", layer, name] for name in species]
        for clean_value, pink_value in zip(clean, pink):
            axis.plot((0, 1), (clean_value, pink_value), color="#B5B5B5", marker="o", markersize=3, linewidth=0.8, alpha=0.65)
        means = (np.mean(clean), np.mean(pink))
        axis.plot((0, 1), means, color=color, marker="o", markersize=7, linewidth=2.75, label="Species mean")
        axis.set_title(f"{label}\nlayer {layer}", fontsize=11)
        axis.set_xticks((0, 1), ("Clean", "Pink noise\n(0 dB)"))
        axis.set_xlim(-0.25, 1.25)
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=10)
    axes[0].set_ylabel("Token linear probe Macro F1 (%) ↑", fontsize=11)
    axes[-1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.2, top=0.78, wspace=0.06)
    save(fig, output / "figure_2_noise_robustness")


def knn_figure(output, species, knn, layers):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.25), dpi=200, sharey=True)
    for axis, (condition, (condition_label, _, _, _)) in zip(axes, CONDITIONS.items()):
        for model, (label, color) in MODELS.items():
            layer = layers[model]
            means = [stats([knn[model, condition, layer, k, name] for name in species])[0] for k in KS]
            axis.plot(KS, means, "o-", color=color, linewidth=2.25, markersize=5, label=f"{label.replace(chr(10), ' ')} · L{layer}")
        axis.set_title(condition_label, fontsize=13)
        axis.set_xticks(KS)
        axis.set_ylim(35, 90)
        axis.grid(alpha=0.18)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=10)
    axes[0].set_ylabel("Macro DN4 accuracy (%) ↑", fontsize=12)
    fig.supxlabel("k", y=0.035, fontsize=13)
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.2, top=0.88, wspace=0.08)
    save(fig, output / "figure_3_dn4_k_sweep")


def write_tables(output, species, logistic, knn, logistic_layers, knn_layers):
    with (output / "layer_sweep.tsv").open("w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(("model", "condition", "method", "layer", "macro_f1_mean", "macro_f1_sem", "n_species"))
        for model in MODELS:
            for condition in CONDITIONS:
                for method in METHODS:
                    for layer in LAYERS:
                        mean, sem = stats([logistic[model, condition, method, layer, name] for name in species])
                        writer.writerow((model, condition, method, layer, f"{mean:.6f}", f"{sem:.6f}", len(species)))
    with (output / "noise_robustness.tsv").open("w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(("model", "clean_selected_layer", "species", "clean_macro_f1", "pink_0db_macro_f1", "drop"))
        for model in MODELS:
            layer = logistic_layers[model, "token"]
            for name in species:
                clean = logistic[model, "clean", "token", layer, name]
                pink = logistic[model, "pink_0db", "token", layer, name]
                writer.writerow((model, layer, name, f"{clean:.6f}", f"{pink:.6f}", f"{clean - pink:.6f}"))
    with (output / "dn4_k_sweep.tsv").open("w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(("model", "clean_k5_selected_layer", "condition", "k", "macro_accuracy_mean", "macro_accuracy_sem", "n_species"))
        for model in MODELS:
            layer = knn_layers[model]
            for condition in CONDITIONS:
                for k in KS:
                    mean, sem = stats([knn[model, condition, layer, k, name] for name in species])
                    writer.writerow((model, layer, condition, k, f"{mean:.6f}", f"{sem:.6f}", len(species)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("Individual_Id_paper_materials/results"))
    parser.add_argument("--output", type=Path, default=Path("Individual_Id_paper_materials/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    species, logistic, knn = load(args.results)
    logistic_layers, knn_layers = best_layers(species, logistic, knn)
    layer_figure(args.output, species, logistic)
    noise_figure(args.output, species, logistic, logistic_layers)
    knn_figure(args.output, species, knn, knn_layers)
    write_tables(args.output, species, logistic, knn, logistic_layers, knn_layers)
    for model in MODELS:
        print(model, "token layer", logistic_layers[model, "token"], "DN4 layer", knn_layers[model])


if __name__ == "__main__":
    main()
