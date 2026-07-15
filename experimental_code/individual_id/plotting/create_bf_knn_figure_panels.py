#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MATRIX = (
    PROJECT_ROOT
    / "results/individual_id_knn_graph_metrics/"
    / "affinity_matrix_k_sweep_songmae_unbinned_usable_allrecordings_nocap_fullgpu/k8/bf/"
    / "knn_attribution_matrices.npz"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "results/individual_id_knn_graph_metrics/bf_unaggregated_figure_panels"

KNN_CMAP = LinearSegmentedColormap.from_list("knn_overlap", ["#fffdf7", "#ffe66d", "#d7301f"])
BLUE = "#2f6fbb"
ORANGE = "#d95f02"
INK = "#202020"


def save_figure(fig, out_dir, stem, formats, dpi):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=dpi)
        paths.append(path)
    plt.close(fig)
    return paths


def quiet_axes(ax, keep_axes=False):
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    if keep_axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(INK)
        ax.spines["bottom"].set_color(INK)
        ax.spines["left"].set_linewidth(1.4)
        ax.spines["bottom"].set_linewidth(1.4)
        ax.tick_params(length=0)
        return
    ax.axis("off")


def parse_birds(text, bird_ids):
    if not text:
        return None
    index_by_id = {str(bird_id): i for i, bird_id in enumerate(bird_ids)}
    birds = []
    for value in text.split(","):
        value = value.strip()
        if value in index_by_id:
            birds.append(index_by_id[value])
        else:
            birds.append(int(value))
    assert birds
    return np.asarray(birds, dtype=np.int64)


def choose_birds(matrix, recording_birds, count, seed, requested):
    if requested is not None:
        assert requested.size == count
        return requested

    rng = np.random.default_rng(seed)
    rows = []
    for bird in np.unique(recording_birds):
        idx = np.flatnonzero(recording_birds == bird)
        same_mass = matrix[np.ix_(idx, idx)].sum(axis=1).mean()
        total_mass = matrix[idx].sum(axis=1).mean()
        rows.append((same_mass - (total_mass - same_mass), rng.random(), int(bird)))

    rows = sorted(rows, reverse=True)
    return np.asarray([bird for _, _, bird in rows[:count]], dtype=np.int64)


def plot_real_heatmap(data, args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix = data["recording_matrix"].astype(np.float32, copy=False)
    recording_birds = data["recording_birds"].astype(np.int64, copy=False)
    bird_ids = data["bird_ids"]

    requested = parse_birds(args.birds, bird_ids)
    birds = choose_birds(matrix, recording_birds, args.bird_count, args.seed, requested)
    birds = birds[np.argsort([np.flatnonzero(recording_birds == bird).size for bird in birds])[::-1]]

    rng = np.random.default_rng(args.seed + 101)
    selected = []
    for bird in birds:
        bird_recordings = np.flatnonzero(recording_birds == bird)
        assert bird_recordings.size >= args.recordings_per_bird
        selected.append(np.sort(rng.choice(bird_recordings, size=args.recordings_per_bird, replace=False)))
    indices = np.concatenate(selected)
    subset = matrix[np.ix_(indices, indices)]
    vmax = max(float(np.percentile(subset, args.heatmap_percentile)), 1e-6)

    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=args.dpi)
    ax.imshow(
        subset,
        cmap=KNN_CMAP,
        norm=PowerNorm(gamma=args.heatmap_gamma, vmin=0.0, vmax=vmax),
        interpolation="nearest",
        aspect="equal",
    )
    quiet_axes(ax)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    metadata = []
    for bird in birds:
        metadata.append(
            {
                "bird_index": int(bird),
                "bird_id": str(bird_ids[bird]),
                "recordings": int(args.recordings_per_bird),
            }
        )
    (args.out_dir / "bf_unaggregated_selected_birds.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    stem = f"bf_unaggregated_heatmap_{args.bird_count}_individuals_{args.recordings_per_bird}recordings"
    return save_figure(fig, args.out_dir, stem, args.formats, args.dpi)


def plot_toy_affinity_row(args):
    rng = np.random.default_rng(args.seed + 17)
    recordings_per_group = 22
    groups = 5
    query_group = 2
    n = groups * recordings_per_group
    x = np.arange(n)
    group_ids = x // recordings_per_group

    same = np.flatnonzero(group_ids == query_group)
    affinity = rng.normal(0.25, 0.055, size=n)
    affinity[same] = rng.normal(0.75, 0.075, size=same.size)
    affinity = np.clip(affinity, 0.08, 0.94)

    colors = np.where(group_ids == query_group, ORANGE, "#ffe8a3")

    fig, ax = plt.subplots(figsize=(6.0, 2.0), dpi=args.dpi)
    ax.bar(x, affinity, width=0.82, color=colors, edgecolor="none")
    for boundary in range(recordings_per_group, n, recordings_per_group):
        ax.axvline(boundary - 0.5, color=(0.72, 0.72, 0.72, 0.55), linewidth=0.7)
    ax.set_xlim(-1, n)
    ax.set_ylim(0, 1.0)
    quiet_axes(ax, keep_axes=True)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.98, bottom=0.11)
    return save_figure(fig, args.out_dir, "toy_affinity_row_distribution", args.formats, args.dpi)


def plot_toy_stable_rank_scatter(args):
    rng = np.random.default_rng(args.seed + 41)
    true_counts = np.repeat(np.arange(1, 11), 5)
    stable_rank = true_counts * 1.12 + rng.normal(0.0, 0.38, size=true_counts.size)
    stable_rank += rng.uniform(-0.16, 0.16, size=true_counts.size)

    slope, intercept = np.polyfit(stable_rank, true_counts, deg=1)
    x_line = np.linspace(float(stable_rank.min()) - 0.2, float(stable_rank.max()) + 0.2, 200)

    fig, ax = plt.subplots(figsize=(3.6, 3.0), dpi=args.dpi)
    ax.scatter(stable_rank, true_counts, s=28, color=BLUE, alpha=0.70, edgecolor="white", linewidth=0.35)
    ax.plot(x_line, slope * x_line + intercept, color=INK, linewidth=1.1, linestyle="--")
    ax.set_xlim(float(x_line.min()), float(x_line.max()))
    ax.set_ylim(0.35, 10.65)
    quiet_axes(ax, keep_axes=True)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.985, bottom=0.09)
    return save_figure(fig, args.out_dir, "toy_stable_rank_scatter", args.formats, args.dpi)


def parse_args():
    parser = argparse.ArgumentParser(description="Create unlabeled Bengalese finch kNN heatmap and toy figure panels.")
    parser.add_argument("--matrix-npz", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bird-count", type=int, default=4)
    parser.add_argument("--recordings-per-bird", type=int, default=10)
    parser.add_argument("--birds", default="", help="Comma-separated bird IDs or integer indices.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--formats", default="png,pdf,svg")
    parser.add_argument("--heatmap-percentile", type=float, default=99.5)
    parser.add_argument("--heatmap-gamma", type=float, default=0.45)
    args = parser.parse_args()
    args.formats = [item.strip().lstrip(".") for item in args.formats.split(",") if item.strip()]
    assert args.bird_count > 0
    assert args.recordings_per_bird > 0
    assert args.formats
    return args


def main():
    args = parse_args()
    data = np.load(args.matrix_npz, allow_pickle=True)
    paths = []
    paths.extend(plot_real_heatmap(data, args))
    paths.extend(plot_toy_affinity_row(args))
    paths.extend(plot_toy_stable_rank_scatter(args))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
