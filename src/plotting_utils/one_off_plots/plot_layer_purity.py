import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
SPECIES = ("zf", "bf", "canary")
SPECIES_LABELS = {"zf": "Zebra Finch", "bf": "Bengalese Finch", "canary": "Canary"}
MODEL_STYLES = {
    "songmae_32x1": ("SongMAE", "#0072B2", "-"),
    "birdaves-base": ("BirdAVES", "#D55E00", "--"),
    "hubert-base": ("HuBERT", "#009E73", ":"),
}
RUNS = (
    (
        "songmae_32x1",
        "SongMAE 32x1 (PCA-256)",
        ROOT / "results/knn_archive_delete_when_new_results_arrive/syllable_knn_songmae32x1_layers_pca256",
        1,
    ),
    ("birdaves-base", "BirdAVES-base (raw)", ROOT / "results/syllable_knn_birdaves_biox_base_layers_rawdims", 1),
    ("hubert-base", "HuBERT-base (raw)", ROOT / "results/syllable_knn_hubert_base_ls960_layers_rawdims", 0),
)


def layer_purity(path, offset):
    values = defaultdict(list)
    for summary in path.glob("**/summary.json"):
        species = summary.relative_to(path).parts[0]
        layer = int(next(part[6:] for part in summary.parts if part.startswith("layer_")))
        if species not in SPECIES or layer < 0:
            continue
        data = json.loads(summary.read_text())
        for row in data["rows"]:
            values[species, layer + offset, row["k"]].append(100 * row["macro_same_purity"])

    means = {key: sum(rows) / len(rows) for key, rows in values.items()}
    layers = sorted({layer for _, layer, _ in means})
    k_values = sorted({k for _, _, k in means})
    curves = {}
    for layer in layers:
        curves[layer] = []
        for k in k_values:
            curves[layer].append(sum(means[species, layer, k] for species in SPECIES) / len(SPECIES))
    return k_values, curves, means


def plot_species(output_dir, k_values, models):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), dpi=200, sharex=True, sharey=True)
    for ax, species in zip(axes, SPECIES):
        for name, (layer, means) in models.items():
            label, color, style = MODEL_STYLES[name]
            purity = [means[species, layer, k] for k in k_values]
            ax.plot(range(len(k_values)), purity, marker="o", markersize=3.5, linewidth=2, color=color, linestyle=style, label=f"{label}\n(block {layer})")
        ax.set_title(SPECIES_LABELS[species], fontsize=11)
        ax.set_ylim(70, 100)
        ax.set_xticks(range(len(k_values)), [str(k) for k in k_values])
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Macro kNN purity (%)")
    fig.supxlabel("Neighbors (k)", y=0.04)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.24, top=0.74, wspace=0.08)
    output = output_dir / "best_layer_by_species.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output)


def plot_rows(output_dir, k_values, layers_by_model, models):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), dpi=200, sharex=True, sharey=True)
    colors = plt.colormaps["viridis"]
    for ax, (name, curves) in zip(axes[0], layers_by_model.items()):
        layers = sorted(curves)
        for i, layer in enumerate(layers):
            ax.plot(range(len(k_values)), curves[layer], marker="o", markersize=3, linewidth=1.6, color=colors(i / (len(layers) - 1)))
        ax.set_title(MODEL_STYLES[name][0], fontsize=11)

    for ax, species in zip(axes[1], SPECIES):
        for name, (layer, means) in models.items():
            label, color, style = MODEL_STYLES[name]
            purity = [means[species, layer, k] for k in k_values]
            ax.plot(range(len(k_values)), purity, marker="o", markersize=3.5, linewidth=2, color=color, linestyle=style, label=f"{label}\n(block {layer})")
        ax.set_title(SPECIES_LABELS[species], fontsize=11)

    for ax in axes.flat:
        ax.set_ylim(70, 100)
        ax.set_xticks(range(len(k_values)), [str(k) for k in k_values])
        ax.grid(alpha=0.18)
    axes[0, 0].set_ylabel("Macro kNN purity (%)")
    axes[1, 0].set_ylabel("Macro kNN purity (%)")
    fig.supxlabel("Neighbors (k)", y=0.035)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", ncol=1, frameon=False, bbox_to_anchor=(0.84, 0.3), fontsize=8.5)
    fig.subplots_adjust(left=0.08, right=0.82, bottom=0.13, top=0.94, wspace=0.08, hspace=0.48)
    colorbar_ax = fig.add_axes([0.855, 0.71, 0.012, 0.18])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=colors), cax=colorbar_ax)
    colorbar.set_ticks([0, 1], labels=["Early", "Late"])
    colorbar_ax.text(0.5, -0.14, "Encoder\nDepth", transform=colorbar_ax.transAxes, ha="center", va="top", fontsize=8.5)
    output = output_dir / "layer_and_species_rows.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output)


def main():
    output_dir = ROOT / "imgs/layer_purity_inspection"
    output_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    layers_by_model = {}
    for name, title, path, offset in RUNS:
        k_values, curves, means = layer_purity(path, offset)
        layers_by_model[name] = curves
        layers = sorted(curves)
        best_layer = max(layers, key=lambda layer: curves[layer][k_values.index(10)])
        models[name] = best_layer, means
        colors = plt.colormaps["viridis"]
        norm = plt.Normalize(layers[0], layers[-1])
        fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=200)
        for layer in layers:
            ax.plot(range(len(k_values)), curves[layer], marker="o", markersize=4, linewidth=2, color=colors(norm(layer)))
        ax.set(title=title, xlabel="Neighbors (k)", ylabel="Macro kNN purity (%)", ylim=(70, 100))
        ax.set_xticks(range(len(k_values)), [str(k) for k in k_values])
        ax.grid(alpha=0.18)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colors), ax=ax, ticks=layers, label="Encoder block")
        fig.tight_layout()
        output = output_dir / f"{name}.png"
        fig.savefig(output)
        plt.close(fig)
        print(output)
    plot_species(output_dir, k_values, models)
    plot_rows(output_dir, k_values, layers_by_model, models)


if __name__ == "__main__":
    main()
