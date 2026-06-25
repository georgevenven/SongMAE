#!/usr/bin/env python3
"""Aggregate the SV1 annotation-R^2 sweep into publication-style figures.

Run once at the end of ``shell/sv1_annotation_r2_across_models.sh``. Reads the
orchestrator output tree::

    <results_root>/<model>/metrics.json        (from src/evals/sv1_annotation_r2.py, by_dataset rows)
    <results_root>/<model>/overlays/<species>.npz

and writes:

  1. A species x model heatmap of rasterized R^2 (how much the top singular vector
     tracks unit-coverage song state), plus a model-independent pixel-intensity
     (loudness) baseline column; styled after the linear-probe heatmap in
     ``individual_id/plotting/plot_linear_probe_paired_comparison.py``.
  2. Per-species spectrogram panels: each example shows the mel spectrogram with a
     red/blue colorbar strip beneath encoding the SV1.latent dot-product, plus a
     song-state strip (unit=red, non-song=black).
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


INK = "#33373d"

# Optional pretty names; dataset keys not listed fall back to a title-cased key.
SPECIES_DISPLAY = {
    "american_robin": "American Robin",
    "bf": "Bengalese Finch",
    "canary": "Canary",
    "cassins_vireo": "Cassin's Vireo",
    "zf": "Zebra Finch",
    "european_starling": "European Starling",
    "ovenbird": "Ovenbird",
    "little_owl": "Little Owl",
    "chiffchaff": "Chiffchaff",
    "swamp_sparrow": "Swamp Sparrow",
    "great_tit": "Great Tit",
    "tree_pipit": "Tree Pipit",
}


def species_name(key):
    return SPECIES_DISPLAY.get(key, key.replace("_", " ").title())


def row_value(row):
    # Headline is the rasterized R^2; fall back to token-level R^2 if absent.
    return float(row["r2_raster"]) if "r2_raster" in row else float(row["r2"])


def pixel_value(row):
    # Loudness baseline R^2 (model-independent); present only where spectrograms were stored.
    px = row.get("pixel_intensity")
    return row_value(px) if px else None


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover(results_root):
    """Layout: <model>/metrics.json with by_dataset rows from one SV1 fit per model."""
    results_root = Path(results_root)
    values, overlays, pixels = {}, {}, {}
    for metrics_path in sorted(results_root.glob("*/metrics.json")):
        model = metrics_path.parent.name
        metrics = json.loads(metrics_path.read_text())
        for row in metrics.get("by_dataset", []):
            dataset = row["dataset"]
            values.setdefault(dataset, {})[model] = row_value(row)
            px = pixel_value(row)
            if px is not None:
                pixels[dataset] = px
            overlay = metrics_path.parent / "overlays" / f"{dataset}.npz"
            if overlay.exists():
                overlays.setdefault(dataset, {})[model] = overlay
    assert values, f"no metrics.json found under {results_root}"
    datasets = sorted(values)
    models = sorted({model for row in values.values() for model in row})
    return datasets, models, values, overlays, pixels


def build_matrix(datasets, models, values):
    data = np.full((len(datasets), len(models)), np.nan, dtype=np.float64)
    for r, dataset in enumerate(datasets):
        for c, model in enumerate(models):
            if model in values[dataset]:
                data[r, c] = values[dataset][model]
    return data


# --------------------------------------------------------------------------- #
# Heatmap (style borrowed from the linear-probe paired-comparison figure)
# --------------------------------------------------------------------------- #
def muted_cmap(name, amount=0.45, n=256):
    base = plt.get_cmap(name)
    colors = base(np.linspace(0.0, 1.0, n))
    colors[:, :3] = colors[:, :3] + (1.0 - colors[:, :3]) * amount
    return mcolors.ListedColormap(colors)


def draw_heatmap(ax, data, row_labels, col_labels):
    cmap = muted_cmap("RdYlGn", amount=0.45)
    cmap.set_bad("#e8e8e8")
    finite = data[np.isfinite(data)]
    vmin = max(0.0, float(finite.min()) - 0.05) if finite.size else 0.0
    # aspect="auto" keeps cells legibly wide regardless of how many models/columns there are.
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=12, fontweight="bold",
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12, fontweight="bold")
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            text = "--" if not np.isfinite(data[r, c]) else f"{data[r, c]:.2f}"
            ax.text(c, r, text, ha="center", va="center", color="#202020", fontsize=12)
    ax.tick_params(length=0)


def save_fig(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    print(out_dir / f"{name}.png")


def write_heatmap(datasets, models, data, out_dir, pixels=None):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "svg.fonttype": "none",
    })

    # Columns high-to-low by median R^2 so the strongest encoders sit left.
    order = np.argsort(np.nanmedian(np.where(np.isnan(data), np.nan, data), axis=0))[::-1]
    ordered = data[:, order]
    col_labels = [models[i].replace("_", " ") for i in order]
    row_labels = [species_name(d) for d in datasets]

    # Append the model-independent pixel-intensity (loudness) baseline as a fixed last column.
    if pixels:
        pixel_col = np.array([[pixels.get(d, np.nan)] for d in datasets], dtype=np.float64)
        ordered = np.concatenate([ordered, pixel_col], axis=1)
        col_labels = col_labels + ["Pixel\nintensity"]

    fig, ax = plt.subplots(figsize=(1.5 * ordered.shape[1] + 4.0, 0.7 * len(datasets) + 2.0))
    draw_heatmap(ax, ordered, row_labels, col_labels)
    ax.set_title("SV1 song-state R² (rasterized)", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    save_fig(fig, out_dir, "sv1_annotation_r2_heatmap")
    plt.close(fig)

    with (out_dir / "sv1_annotation_r2.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Species"] + col_labels)
        for label, row in zip(row_labels, ordered):
            writer.writerow([label] + ["" if not np.isfinite(v) else f"{v:.4f}" for v in row])
    print(out_dir / "sv1_annotation_r2.csv")


# --------------------------------------------------------------------------- #
# Spectrogram + SV1 colorbar strip panels
# --------------------------------------------------------------------------- #
def load_overlay(path):
    data = np.load(path, allow_pickle=True)
    count = int(data["count"])
    items = []
    for i in range(count):
        items.append({
            "spectrogram": data[f"spectrogram_{i}"],
            "score": data[f"score_{i}"],
            "song": data[f"song_{i}"],
            "name": str(data["names"][i]),
        })
    return items


def write_spec_panels(dataset, model, overlay_path, out_dir, max_examples=6):
    items = load_overlay(overlay_path)[:max_examples]
    if not items:
        return

    # One global min-max over all shown events (shared scale), matching the R^2; per-event
    # normalization would be circular since events are intensity-defined.
    all_scores = np.concatenate([np.asarray(item["score"], dtype=np.float64) for item in items])
    g_lo, g_hi = float(all_scores.min()), float(all_scores.max())
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    score_cmap = plt.get_cmap("bwr")  # blue (low SV1) -> white -> red (high SV1)
    song_cmap = mcolors.ListedColormap(["black", "red"])  # non-song black, unit/song red

    n = len(items)
    heights = []
    for _ in range(n):
        heights += [4.0, 0.5, 0.45, 0.6]  # spec, score strip, song strip, gap
    fig = plt.figure(figsize=(11, 1.9 * n + 0.6), constrained_layout=False)
    grid = fig.add_gridspec(4 * n, 1, height_ratios=heights, hspace=0.05)
    fig.suptitle(f"{species_name(dataset)} | {model.replace('_', ' ')} | SV1·latent intensity",
                 fontsize=11, y=0.995)

    score_im = None
    for i, item in enumerate(items):
        spec = item["spectrogram"]
        raw = np.asarray(item["score"], dtype=np.float64)
        score = ((raw - g_lo) / (g_hi - g_lo) if g_hi > g_lo else np.zeros_like(raw))[None, :]
        song = item["song"][None, :]
        width = spec.shape[1]

        ax_spec = fig.add_subplot(grid[4 * i, 0])
        s_lo, s_hi = np.percentile(spec, [1, 99.5])
        ax_spec.imshow(spec, aspect="auto", origin="lower", cmap="magma",
                       vmin=s_lo, vmax=s_hi, extent=[0, width, 0, spec.shape[0]])
        ax_spec.set_xticks([])
        ax_spec.set_yticks([])
        ax_spec.set_ylabel(item["name"], rotation=0, ha="right", va="center", fontsize=7, labelpad=22)

        ax_score = fig.add_subplot(grid[4 * i + 1, 0])
        score_im = ax_score.imshow(score, aspect="auto", cmap=score_cmap, norm=norm,
                                   interpolation="nearest", extent=[0, width, 0, 1])
        ax_score.set_xticks([])
        ax_score.set_yticks([])
        ax_score.set_ylabel("SV1", rotation=0, ha="right", va="center", fontsize=7, labelpad=22)

        ax_song = fig.add_subplot(grid[4 * i + 2, 0])
        ax_song.imshow(song, aspect="auto", cmap=song_cmap, vmin=0, vmax=1,
                       interpolation="nearest", extent=[0, width, 0, 1])
        ax_song.set_yticks([])
        ax_song.set_ylabel("song", rotation=0, ha="right", va="center", fontsize=7, labelpad=22)
        ax_song.set_xticks([0, width])
        ax_song.tick_params(labelsize=6, length=2)

    if score_im is not None:
        cbar = fig.colorbar(score_im, ax=fig.axes, fraction=0.025, pad=0.02)
        cbar.set_label("SV1·latent (global min-max)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_{model}_spec_sv1.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


def main():
    parser = argparse.ArgumentParser(description="Heatmap + spec colorbar panels for the SV1 annotation-R^2 sweep.")
    parser.add_argument("--results_root", required=True, help="Sweep output root (<model>/metrics.json).")
    parser.add_argument("--out_dir", default=None, help="Where to write figures (default: <results_root>/plots).")
    parser.add_argument("--max_examples", type=int, default=6, help="Spectrogram examples per cell panel.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results_root) / "plots"
    datasets, models, values, overlays, pixels = discover(args.results_root)
    write_heatmap(datasets, models, build_matrix(datasets, models, values), out_dir, pixels)

    spec_dir = out_dir / "spec_panels"
    for dataset in datasets:
        for model in models:
            overlay_path = overlays.get(dataset, {}).get(model)
            if overlay_path is not None:
                write_spec_panels(dataset, model, overlay_path, spec_dir, args.max_examples)


if __name__ == "__main__":
    main()
