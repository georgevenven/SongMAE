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
  2. A dot plot with one semi-transparent circle per dataset/model value.
  3. Standalone spectrogram plots: each example shows the mel spectrogram with a
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
PIXEL_LABEL = "Spectrogram\nintensity"
SPECIES_COLORS = {
    "american_robin": "#4e79a7",
    "bf": "#f28e2b",
    "canary": "#59a14f",
    "swamp_sparrow": "#e15759",
    "zf": "#b07aa1",
}

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

MODEL_DISPLAY = {
    "xcl_micro_500k_p32x4_default": "SongMAE 32 x 4",
    "xcl_micro_500k_p32x1_default": "SongMAE 32 x 1",
    "aves": "BirdAVES",
    "hubert": "HuBERT",
}

MODEL_ORDER = {
    "xcl_micro_500k_p32x4_default": 0,
    "xcl_micro_500k_p32x1_default": 1,
    "aves": 2,
    "hubert": 3,
}


def species_name(key):
    return SPECIES_DISPLAY.get(key, key.replace("_", " ").title())


def model_name(key):
    return MODEL_DISPLAY.get(key, key.replace("_", " "))


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
    values, overlays, pixels, recording_values, recording_pixels, metrics_by_model = {}, {}, {}, [], {}, {}
    for metrics_path in sorted(results_root.glob("*/metrics.json")):
        model = metrics_path.parent.name
        metrics = json.loads(metrics_path.read_text())
        metrics_by_model[model] = metrics
        for row in metrics.get("by_dataset", []):
            dataset = row["dataset"]
            values.setdefault(dataset, {})[model] = row_value(row)
            px = pixel_value(row)
            if px is not None:
                pixels[dataset] = px
            overlay = metrics_path.parent / "overlays" / f"{dataset}.npz"
            if overlay.exists():
                overlays.setdefault(dataset, {})[model] = overlay
        for row in metrics.get("by_recording", []):
            dataset = row["dataset"]
            recording = row["recording"]
            recording_values.append((dataset, model, row_value(row)))
            px = pixel_value(row)
            if px is not None:
                recording_pixels[(dataset, recording)] = px
    assert values, f"no metrics.json found under {results_root}"
    datasets = sorted(values)
    models = sorted({model for row in values.values() for model in row})
    return datasets, models, values, overlays, pixels, recording_values, recording_pixels, metrics_by_model


def build_matrix(datasets, models, values):
    data = np.full((len(datasets), len(models)), np.nan, dtype=np.float64)
    for r, dataset in enumerate(datasets):
        for c, model in enumerate(models):
            if model in values[dataset]:
                data[r, c] = values[dataset][model]
    return data


def display_table(datasets, models, data, pixels=None):
    order = sorted(range(len(models)), key=lambda i: MODEL_ORDER.get(models[i], 100 + i))
    table = data[:, order]
    labels = [model_name(models[i]) for i in order]
    if pixels:
        pixel_col = np.array([[pixels.get(d, np.nan)] for d in datasets], dtype=np.float64)
        table = np.concatenate([table, pixel_col], axis=1)
        labels.append(PIXEL_LABEL)
    return table, labels


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
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=1.0, aspect="equal")
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
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300)
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

    ordered, col_labels = display_table(datasets, models, data, pixels)
    row_labels = [species_name(d) for d in datasets]

    mean_row = np.nanmean(ordered, axis=0, keepdims=True)
    ordered = np.concatenate([ordered, mean_row], axis=0)
    row_labels = row_labels + ["Mean"]

    side = max(8.0, 1.05 * max(ordered.shape) + 2.5)
    fig, ax = plt.subplots(figsize=(side, side))
    draw_heatmap(ax, ordered, row_labels, col_labels)
    ax.set_title("Singular-subspace song-state R² (rasterized)", fontsize=13, fontweight="bold", pad=12)
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
# Dot plot
# --------------------------------------------------------------------------- #
def write_dotplot(datasets, models, data, out_dir, pixels=None, recording_values=None, recording_pixels=None):
    table, labels = display_table(datasets, models, data, pixels)
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    ordered_models = sorted(models, key=lambda model: MODEL_ORDER.get(model, 100 + models.index(model)))
    x = {model: i for i, model in enumerate(ordered_models)}
    if pixels:
        x["pixel_intensity"] = len(ordered_models)

    points = list(recording_values or [])
    points.extend((dataset, "pixel_intensity", value) for (dataset, _), value in (recording_pixels or {}).items())
    if points:
        rng = np.random.default_rng(7)
        offsets = np.linspace(-0.27, 0.27, len(datasets))
        dataset_offsets = dict(zip(datasets, offsets))
        for dataset in datasets:
            rows = [(model, value) for d, model, value in points if d == dataset and model in x]
            if not rows:
                continue
            offset = dataset_offsets[dataset]
            xs = np.array([x[model] for model, _ in rows], dtype=np.float64)
            ys = np.array([value for _, value in rows], dtype=np.float64)
            ax.scatter(xs + offset + rng.uniform(-0.035, 0.035, size=ys.size), ys, s=10, alpha=0.16,
                       color=SPECIES_COLORS.get(dataset, "#777777"), edgecolor="none", rasterized=True)
            ax.scatter([], [], s=34, alpha=0.85, color=SPECIES_COLORS.get(dataset, "#777777"),
                       edgecolor="none", label=species_name(dataset))
        ymax = max(0.72, float(np.nanmax([value for _, _, value in points])) + 0.08)
    else:
        offsets = np.linspace(-0.2, 0.2, len(datasets))
        for offset, dataset, row in zip(offsets, datasets, table):
            mask = np.isfinite(row)
            ax.scatter(np.arange(len(labels))[mask] + offset, row[mask], s=150, alpha=0.58,
                       color=SPECIES_COLORS.get(dataset, "#777777"), edgecolor="white",
                       linewidth=0.7, label=species_name(dataset))
        ymax = max(0.72, float(np.nanmax(table)) + 0.08)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", rotation_mode="anchor", fontweight="bold")
    pooled_table, _ = display_table(datasets, models, data, pixels)
    if points:
        for offset, dataset, row in zip(np.linspace(-0.27, 0.27, len(datasets)), datasets, pooled_table):
            mask = np.isfinite(row)
            ax.scatter(np.arange(len(labels))[mask] + offset, row[mask], s=54, marker="D",
                       color=SPECIES_COLORS.get(dataset, "#777777"), edgecolor="#202020",
                       linewidth=0.55, alpha=0.95, zorder=5)
        ax.scatter([], [], s=54, marker="D", color="white", edgecolor="#202020",
                   linewidth=0.8, label="Pooled dataset R²")
    ax.set_ylabel("Recording-level R²; diamonds = pooled dataset R²")
    ax.set_ylim(0.0, min(1.0, ymax))
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=1, fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    save_fig(fig, out_dir, "sv1_annotation_r2_dotplot")
    plt.close(fig)


def recording_metric(rows, dataset, key, weighted=False):
    values, weights = [], []
    for row in rows:
        if row["dataset"] != dataset:
            continue
        item = row.get(key) if key == "pixel_intensity" else row
        if item is None:
            continue
        values.append(float(item["r2_raster"]))
        weights.append(float(item["frames_raster"]))
    if not values:
        return ""
    if weighted:
        return f"{np.average(values, weights=weights):.6f}"
    return f"{np.mean(values):.6f}"


def fmt_metric(row, key):
    if key not in row:
        return ""
    return f"{row[key]:.6f}"


def write_songmae_pixel_diagnostics(datasets, metrics_by_model, out_dir):
    model = "xcl_micro_500k_p32x1_default"
    if model not in metrics_by_model:
        return
    metrics = metrics_by_model[model]
    by_dataset = {row["dataset"]: row for row in metrics.get("by_dataset", [])}
    by_recording = metrics.get("by_recording", [])
    out_path = out_dir / "songmae32x1_vs_spectrogram_intensity_diagnostics.csv"
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Species",
            "Recordings",
            "SongMAE pooled R2",
            "Spectrogram intensity pooled R2",
            "SongMAE within-recording centered R2",
            "Spectrogram intensity within-recording centered R2",
            "SongMAE mean recording R2",
            "Spectrogram intensity mean recording R2",
            "SongMAE frame-weighted mean recording R2",
            "Spectrogram intensity frame-weighted mean recording R2",
        ])
        for dataset in datasets:
            row = by_dataset.get(dataset)
            if not row or "pixel_intensity" not in row:
                continue
            pixel = row["pixel_intensity"]
            writer.writerow([
                species_name(dataset),
                sum(1 for item in by_recording if item["dataset"] == dataset),
                f"{row['r2_raster']:.6f}",
                f"{pixel['r2_raster']:.6f}",
                fmt_metric(row, "r2_raster_within_recording_centered"),
                fmt_metric(pixel, "r2_raster_within_recording_centered"),
                recording_metric(by_recording, dataset, "sv1"),
                recording_metric(by_recording, dataset, "pixel_intensity"),
                recording_metric(by_recording, dataset, "sv1", weighted=True),
                recording_metric(by_recording, dataset, "pixel_intensity", weighted=True),
            ])
    print(out_path)


# --------------------------------------------------------------------------- #
# Spectrogram + SV1 colorbar strip plots
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


def safe_name(value):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def write_spec_example(dataset, model, item, index, g_lo, g_hi, out_dir):
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    score_cmap = plt.get_cmap("bwr")
    song_cmap = mcolors.ListedColormap(["black", "red"])
    spec = item["spectrogram"]
    raw = np.asarray(item["score"], dtype=np.float64)
    score = ((raw - g_lo) / (g_hi - g_lo) if g_hi > g_lo else np.zeros_like(raw))[None, :]
    song = item["song"][None, :]
    width = spec.shape[1]

    fig = plt.figure(figsize=(11, 2.6), constrained_layout=False)
    grid = fig.add_gridspec(3, 1, height_ratios=[4.0, 0.5, 0.45], hspace=0.05)
    fig.suptitle(f"{species_name(dataset)} | {model_name(model)} | {item['name']}", fontsize=10, y=0.995)

    ax_spec = fig.add_subplot(grid[0, 0])
    s_lo, s_hi = np.percentile(spec, [1, 99.5])
    ax_spec.imshow(spec, aspect="auto", origin="lower", cmap="magma",
                   vmin=s_lo, vmax=s_hi, extent=[0, width, 0, spec.shape[0]])
    ax_spec.set_xticks([])
    ax_spec.set_yticks([])

    ax_score = fig.add_subplot(grid[1, 0])
    score_im = ax_score.imshow(score, aspect="auto", cmap=score_cmap, norm=norm,
                               interpolation="nearest", extent=[0, width, 0, 1])
    ax_score.set_xticks([])
    ax_score.set_yticks([])
    ax_score.set_ylabel("SV1", rotation=0, ha="right", va="center", fontsize=7, labelpad=22)

    ax_song = fig.add_subplot(grid[2, 0])
    ax_song.imshow(song, aspect="auto", cmap=song_cmap, vmin=0, vmax=1,
                   interpolation="nearest", extent=[0, width, 0, 1])
    ax_song.set_yticks([])
    ax_song.set_ylabel("song", rotation=0, ha="right", va="center", fontsize=7, labelpad=22)
    ax_song.set_xticks([0, width])
    ax_song.tick_params(labelsize=6, length=2)

    cbar = fig.colorbar(score_im, ax=fig.axes, fraction=0.025, pad=0.02)
    cbar.set_label("SV1·latent (global min-max)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_{model}_{index:02d}_{safe_name(item['name'])}_spec_sv1.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


def write_spec_examples(dataset, model, overlay_path, out_dir, max_examples=6):
    items = load_overlay(overlay_path)[:max_examples]
    if not items:
        return

    all_scores = np.concatenate([np.asarray(item["score"], dtype=np.float64) for item in items])
    g_lo, g_hi = float(all_scores.min()), float(all_scores.max())
    for i, item in enumerate(items):
        write_spec_example(dataset, model, item, i, g_lo, g_hi, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Heatmap + standalone spec colorbar plots for the SV1 annotation-R^2 sweep.")
    parser.add_argument("--results_root", required=True, help="Sweep output root (<model>/metrics.json).")
    parser.add_argument("--out_dir", default=None, help="Where to write figures (default: <results_root>/plots).")
    parser.add_argument("--max_examples", type=int, default=6, help="Standalone spectrogram examples per dataset/model.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results_root) / "plots"
    datasets, models, values, overlays, pixels, recording_values, recording_pixels, metrics_by_model = discover(args.results_root)
    data = build_matrix(datasets, models, values)
    write_heatmap(datasets, models, data, out_dir, pixels)
    write_dotplot(datasets, models, data, out_dir, pixels, recording_values, recording_pixels)
    write_songmae_pixel_diagnostics(datasets, metrics_by_model, out_dir)

    spec_dir = out_dir / "spec_panels"
    for path in spec_dir.glob("*_spec_sv1.png"):
        path.unlink()
    for dataset in datasets:
        for model in models:
            overlay_path = overlays.get(dataset, {}).get(model)
            if overlay_path is not None:
                write_spec_examples(dataset, model, overlay_path, spec_dir, args.max_examples)


if __name__ == "__main__":
    main()
