#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SPECIES_ORDER = ["zf", "bf", "canary"]
SPECIES_LABELS = {
    "zf": "Zebra Finch",
    "bf": "Bengalese Finch",
    "canary": "Canary",
}
MODEL_ORDER = ["xcl_base_100k_p16x4_c010", "xcl_base_100k_p32x1_c010", "birdaves_biox_base", "hubert_base_ls960"]
LAYER_ORDER = list(range(13))
SUBLAYER_ORDER = [
    "out_proj",
    "attn_residual",
    "linear1",
    "gelu",
    "ffn",
    "end_of_block",
]
BEST_MODEL = "xcl_base_100k_p32x1_c010"
MODEL_LABELS = {
    "xcl_base_100k_p16x4_c010": "SongMAE 16x4",
    "xcl_base_100k_p32x1_c010": "SongMAE 32x1",
    "birdaves_biox_base": "BirdAVES",
    "hubert_base_ls960": "HuBERT base",
}
LAYER_LABELS = {layer: f"Layer {layer}" for layer in LAYER_ORDER}


def parse_layer(text):
    assert text.startswith("layer_")
    layer = int(text.replace("layer_", ""))
    return 6 if layer == -1 else layer


def read_rows(root):
    rows = []
    for path in sorted(Path(root).glob("**/summary.json")):
        rel = path.relative_to(root).parts
        species, bird, model, layer = rel[:4]
        if species not in SPECIES_ORDER:
            continue
        data = json.loads(path.read_text())
        target = rel[4] if len(rel) == 6 else data["target_feature_type"]
        for item in data["rows"]:
            rows.append({
                "species": species,
                "bird": f"{species}/{bird}",
                "model": data.get("name") or model,
                "layer": parse_layer(layer),
                "target": target,
                "k": int(item["k"]),
                "purity": 100.0 * float(item["macro_same_purity"]),
            })
    assert rows, f"no summary rows under {root}"
    return rows


def ordered(items, order):
    present = set(items)
    out = [item for item in order if item in present]
    out.extend(sorted(present - set(out)))
    return out


def points(rows, key, values, k_values):
    out = []
    for value in values:
        by_k = defaultdict(list)
        for row in rows:
            if row[key] == value:
                by_k[row["k"]].append(row["purity"])
        ys = [mean(by_k[k]) for k in k_values if by_k[k]]
        xs = [k for k in k_values if by_k[k]]
        if ys:
            out.append((value, xs, ys))
    return out


def draw_curves(
    ax,
    rows,
    key,
    values,
    labels,
    title,
    *,
    legend_cols=3,
    alpha=0.95,
    show_ylabel=True,
    legend_size=12,
    legend_below=False,
):
    k_values = sorted({row["k"] for row in rows})
    x = {k: i for i, k in enumerate(k_values)}
    curves = points(rows, key, values, k_values)
    colors = plt.get_cmap("viridis")
    for i, (value, xs, ys) in enumerate(curves):
        ax.plot(
            [x[k] for k in xs],
            ys,
            marker="o",
            markersize=4.8,
            linewidth=2.2,
            alpha=alpha,
            color=colors(i / max(1, len(curves) - 1)),
            label=labels.get(value, str(value)),
        )
    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.set_xlim(-0.25, len(k_values) - 0.75)
    ax.set_ylim(50.0, 100.0)
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([str(k) for k in k_values])
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(k_values))))
    ax.set_xlabel("k", fontsize=17, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Purity (%)", fontsize=17, fontweight="bold")
        ax.yaxis.set_label_coords(-0.18, 0.5)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.grid(True, alpha=0.18)
    if legend_below:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=legend_cols, frameon=False, fontsize=legend_size)
    elif legend_cols > 3:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=legend_cols, frameon=False, fontsize=legend_size)
    else:
        ax.legend(loc="lower left", ncol=legend_cols, frameon=True, fontsize=legend_size)
    ax.tick_params(axis="both", labelsize=17.5, width=1.0)
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_color("#404040")
    ax.set_box_aspect(1)


def save_plot(output, fig):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_curves(rows, key, values, labels, title, output, *, legend_cols=3, alpha=0.95):
    fig, ax = plt.subplots(figsize=(3.2, 3.4), dpi=300)
    draw_curves(ax, rows, key, values, labels, title, legend_cols=legend_cols, alpha=alpha, legend_size=12)
    save_plot(output, fig)


def plot_combined(panels, output):
    fig, axes = plt.subplots(1, 4, figsize=(22.0, 5.6), dpi=300, sharey=True)
    for i, (ax, panel) in enumerate(zip(axes.flat, panels)):
        draw_curves(ax, *panel, legend_cols=2, show_ylabel=i == 0, legend_size=12)
    fig.subplots_adjust(wspace=0.08)
    save_plot(output, fig)


def output_path(output, name):
    return output.with_name(f"{output.stem}_{name}{output.suffix}")


def main():
    parser = argparse.ArgumentParser(description="Plot syllable kNN purity curves.")
    parser.add_argument("--models_root", default="results/syllable_knn_allbirds_models_rawdims")
    parser.add_argument("--layers_root", default="results/syllable_knn_songmae32x1_layers_rawdims")
    parser.add_argument("--sublayers_root", default="results/syllable_knn_songmae32x1_layer5_sublayers_rawdims")
    parser.add_argument("--best_model", default=BEST_MODEL)
    parser.add_argument("--output", default="imgs/syllable_knn_purity.png")
    args = parser.parse_args()

    model_rows = read_rows(Path(args.models_root))
    layer_rows = read_rows(Path(args.layers_root))
    sublayer_rows = read_rows(Path(args.sublayers_root))
    sublayer_rows = [row for row in sublayer_rows if row["target"] in SUBLAYER_ORDER]
    best_rows = [row for row in model_rows if row["model"] == args.best_model]
    assert best_rows, f"no rows for best model {args.best_model}"

    output = Path(args.output)
    outputs = [
        output_path(output, "layers"),
        output_path(output, "sublayers"),
        output_path(output, "models"),
        output_path(output, "birds"),
        output_path(output, "combined"),
    ]
    panels = [
        (
        [row for row in layer_rows if row["target"] == "end_of_block"],
        "layer",
        ordered([row["layer"] for row in layer_rows], LAYER_ORDER),
        LAYER_LABELS,
        "Purity v Layers",
        ),
        (
        sublayer_rows,
        "target",
        ordered([row["target"] for row in sublayer_rows], SUBLAYER_ORDER),
        {target: target for target in SUBLAYER_ORDER},
        "Purity v Sub-layers",
        ),
        (
        model_rows,
        "model",
        ordered([row["model"] for row in model_rows], MODEL_ORDER),
        MODEL_LABELS,
        "Purity v Models",
        ),
        (
        best_rows,
        "species",
        ordered([row["species"] for row in best_rows], SPECIES_ORDER),
        SPECIES_LABELS,
        "Purity v Birds",
        ),
    ]
    for panel, path in zip(panels, outputs[:4]):
        plot_curves(*panel, path)
    plot_combined(panels, outputs[4])
    for path in outputs:
        print(path)
        print(path.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
