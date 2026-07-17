import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
SPECIES = ("zf", "bf", "canary")
SPECIES_LABELS = {"zf": "Zebra Finch", "bf": "Bengalese Finch", "canary": "Canary"}
MODEL_STYLES = {
    "songmae_32x4": ("SongMAE", "#0072B2", "-"),
    "birdaves-base": ("BirdAVES", "#D55E00", "--"),
    "hubert-base": ("HuBERT", "#009E73", ":"),
}
RESULTS_ROOT = ROOT / "results/knn/all_birds_models_all_layers_zscore"
RUNS = (
    (
        "songmae_32x4",
        "SongMAE",
        RESULTS_ROOT,
        "xcl_large_500k_p32x4_c010",
    ),
    ("birdaves-base", "BirdAVES-base (raw)", RESULTS_ROOT, "birdaves_biox_base"),
    ("hubert-base", "HuBERT-base (raw)", RESULTS_ROOT, "hubert_base_ls960"),
)


def is_end_of_block_summary(summary):
    relative = summary.parts
    layer_idx = next((i for i, part in enumerate(relative) if part.startswith("layer_")), None)
    if layer_idx is None:
        return False
    return layer_idx + 1 >= len(relative) - 1 or relative[layer_idx + 1] == "end_of_block"


def layer_purity(path, model):
    values = defaultdict(list)
    for summary in path.glob("**/summary.json"):
        relative = summary.relative_to(path)
        if len(relative.parts) < 5 or relative.parts[2] != model:
            continue
        species = relative.parts[0]
        if not is_end_of_block_summary(summary):
            continue
        layer = int(next(part[6:] for part in summary.parts if part.startswith("layer_")))
        if species not in SPECIES or layer < 0:
            continue
        data = json.loads(summary.read_text())
        for row in data["rows"]:
            values[species, layer, row["k"]].append(100 * row["macro_same_purity"])

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


def plot_average_row(output_dir, k_values, layers_by_model, models):
    fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.7), sharex=True, sharey=True)
    colors = plt.colormaps["viridis"]
    norm = plt.Normalize(0, 11)
    for ax, (name, curves) in zip(axes[:3], layers_by_model.items()):
        layers = sorted(curves)
        for i, layer in enumerate(layers):
            ax.plot(
                range(len(k_values)),
                curves[layer],
                marker="o",
                markersize=3,
                linewidth=1.6,
                color=colors(i / (len(layers) - 1)),
            )
        ax.set_title(MODEL_STYLES[name][0], fontsize=11)

    colorbar_ax = axes[0].inset_axes([0.08, 0.08, 0.05, 0.44])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=colors),
        cax=colorbar_ax,
    )
    colorbar.set_ticks([0, 11], labels=["0", "11"])
    colorbar.ax.tick_params(labelsize=5, pad=1, length=2)
    colorbar.set_label("Block", fontsize=6, labelpad=-1)

    ax = axes[3]
    for name, (layer, means) in models.items():
        purity = [sum(means[species, layer, k] for species in SPECIES) / len(SPECIES) for k in k_values]
        label, color, style = MODEL_STYLES[name]
        ax.plot(
            range(len(k_values)),
            purity,
            marker="o",
            markersize=3.5,
            linewidth=2,
            color=color,
            linestyle=style,
            label=f"{label}\n(block {layer})",
        )
    ax.set_title("Best Layer Purity", fontsize=11)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85, borderpad=0.3, handlelength=1.5)

    for ax in axes:
        ax.set_ylim(70, 100)
        ax.set_xticks(range(len(k_values)), [str(k) for k in k_values])
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Macro kNN purity (%)")
    fig.supxlabel("Neighbors (k)", y=0.03)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.22, top=0.9, wspace=0.18)
    output = output_dir / "layer_and_average_row.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output)


def main():
    output_dir = ROOT / "imgs/layer_purity_inspection"
    output_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    layers_by_model = {}
    for name, title, path, model in RUNS:
        k_values, curves, means = layer_purity(path, model)
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
    plot_average_row(output_dir, k_values, layers_by_model, models)


if __name__ == "__main__":
    main()
