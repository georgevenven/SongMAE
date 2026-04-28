#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "results" / "individual_id_knn_graph_metrics" / "all_species_safe_purity"
DEFAULT_OUT = ROOT / "results" / "individual_id_knn_graph_metrics" / "paper_knn_heatmaps"
DEFAULT_SPECIES = ["zf", "bf", "canary", "ovenbird", "chiffchaff", "european_starling", "tree_pipit", "little_owl"]
OVERLAP_CMAP = LinearSegmentedColormap.from_list("overlap", ["#fffdf7", "#ffe66d", "#d7301f"])


def _species_keys(text):
    if text == "paper":
        return DEFAULT_SPECIES
    return [x.strip() for x in text.split(",") if x.strip()]


def _load_matrix(base_dir, species_key):
    path = base_dir / species_key / "knn_purity.npz"
    data = np.load(path, allow_pickle=True)
    return data["individual_neighbor_fraction"].astype(np.float32, copy=False)


def _tick_positions(n):
    if n <= 24:
        return np.arange(n), [str(i) for i in range(1, n + 1)]
    ticks = np.unique(np.r_[0, np.arange(4, n, 5), n - 1])
    return ticks, [str(i + 1) for i in ticks]


def _plot_one(out_dir, species_key, matrix, args, norm):
    fig, ax = plt.subplots(figsize=(args.inches, args.inches), dpi=args.dpi)
    ax.imshow(matrix, cmap=OVERLAP_CMAP, norm=norm, interpolation="nearest")
    ticks, labels = _tick_positions(matrix.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, fontsize=args.tick_fontsize)
    ax.set_yticklabels(labels, fontsize=args.tick_fontsize)
    ax.tick_params(length=1.5, width=0.5, pad=1)
    ax.set_xlabel("Neighbor bird #", fontsize=args.label_fontsize)
    ax.set_ylabel("Query bird #", fontsize=args.label_fontsize)
    ax.set_title(species_key.replace("_", " "), fontsize=args.title_fontsize, pad=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / f"{species_key}_paper_knn_heatmap.png", dpi=args.dpi)
    fig.savefig(out_dir / f"{species_key}_paper_knn_heatmap.pdf", dpi=args.dpi, format="pdf")
    plt.close(fig)


def _plot_collage(out_dir, species_keys, matrices, norms, args):
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), dpi=args.dpi)
    for index, species_key in enumerate(species_keys):
        ax = axes.flat[index]
        matrix = matrices[species_key]
        ax.imshow(matrix, cmap=OVERLAP_CMAP, norm=norms[species_key], interpolation="nearest")
        ticks, labels = _tick_positions(matrix.shape[0])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, fontsize=3.4)
        ax.set_yticklabels(labels, fontsize=3.4)
        ax.tick_params(length=1.0, width=0.4, pad=0.5)
        ax.set_title(species_key.replace("_", " "), fontsize=7, pad=2)
        if index // 4 == 1:
            ax.set_xlabel("Neighbor bird #", fontsize=5)
        if index % 4 == 0:
            ax.set_ylabel("Query bird #", fontsize=5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    for ax in axes.flat[len(species_keys) :]:
        ax.axis("off")
    fig.tight_layout(pad=0.35, w_pad=0.5, h_pad=0.6)
    fig.savefig(out_dir / "paper_knn_heatmap_collage.png", dpi=args.dpi)
    fig.savefig(out_dir / "paper_knn_heatmap_collage.pdf", dpi=args.dpi, format="pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Make paper-ready individual kNN heatmaps.")
    parser.add_argument("--base_dir", default=str(DEFAULT_BASE))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--species", default="paper")
    parser.add_argument("--inches", type=float, default=3.2)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--vmax", type=float, default=0.0)
    parser.add_argument("--vmax_percentile", type=float, default=97.5)
    parser.add_argument("--gamma", type=float, default=0.45)
    parser.add_argument("--shared_scale", action="store_true")
    parser.add_argument("--tick_fontsize", type=float, default=4.5)
    parser.add_argument("--label_fontsize", type=float, default=6.5)
    parser.add_argument("--title_fontsize", type=float, default=7.5)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    species_keys = _species_keys(args.species)
    matrices = {species_key: _load_matrix(base_dir, species_key) for species_key in species_keys}
    if args.shared_scale:
        values = np.concatenate([m.reshape(-1) for m in matrices.values()])
        shared_vmax = args.vmax or float(np.percentile(values, args.vmax_percentile))
    else:
        shared_vmax = 0.0

    vmax_by_species = {}
    norms = {}
    for species_key, matrix in matrices.items():
        vmax = shared_vmax or args.vmax or float(np.percentile(matrix, args.vmax_percentile))
        vmax = max(vmax, 1e-6)
        vmax_by_species[species_key] = vmax
        norms[species_key] = PowerNorm(gamma=args.gamma, vmin=0.0, vmax=vmax)
        _plot_one(out_dir, species_key, matrix, args, norms[species_key])

    _plot_collage(out_dir, species_keys, matrices, norms, args)

    summary = {
        "base_dir": str(base_dir),
        "species": species_keys,
        "row_normalized_in_plot": False,
        "figure_inches": args.inches,
        "cmap": "white-yellow-red",
        "color_norm": "power",
        "shared_scale": args.shared_scale,
        "gamma": args.gamma,
        "vmax_percentile": args.vmax_percentile,
        "vmax_by_species": vmax_by_species,
        "colorbar": False,
        "collage": "paper_knn_heatmap_collage",
    }
    (out_dir / "paper_knn_heatmaps_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[paper-heatmaps] out_dir={out_dir} shared_scale={args.shared_scale}")


if __name__ == "__main__":
    main()
