#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import umap

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.core.extract_embedding import extract_recording_embeddings
from src.core.model import TARGET_FEATURE_TYPES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one detected event in SongMAE latent space.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--spec_dir", required=True)
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "imgs" / "latent_space_interp")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--recording_stem", default=None)
    parser.add_argument("--bird", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--encoder_layer_idx", type=int, default=None)
    parser.add_argument("--target_feature_type", default="attn_residual", choices=TARGET_FEATURE_TYPES)
    parser.add_argument("--random_init", action="store_true")
    return parser.parse_args()


def zscore(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = np.maximum(features.std(axis=0, keepdims=True), 1e-8)
    return ((features - mean) / std).astype(np.float32, copy=False)


def label_colors(labels: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("tab20")
    colors = np.zeros((len(labels), 4))
    for index, label in enumerate(labels):
        colors[index] = [0.0, 0.0, 0.0, 1.0] if int(label) == -1 else cmap(int(label) % 20)
    return colors


def position_colors(xy: np.ndarray) -> np.ndarray:
    span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 1e-12)
    norm = (xy - xy.min(axis=0)) / span
    return np.column_stack([norm[:, 0], norm[:, 1], np.full(len(norm), 0.5)])


def plot_panel(segment: dict, xy: np.ndarray, out_dir: Path) -> tuple[Path, Path]:
    labels = segment["labels_downsampled"]
    gt_colors = label_colors(labels)
    pos_colors = position_colors(xy)

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    fig = plt.figure(figsize=(11, 8.875))
    grid = gridspec.GridSpec(
        4,
        2,
        height_ratios=[3, 2, 0.2, 0.2],
        hspace=0.25,
        wspace=0.1,
        left=0.05,
        right=0.99,
        top=0.88,
        bottom=0.06,
    )

    for ax, colors, panel_title in zip(
        [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
        [gt_colors, pos_colors],
        ["Ground Truth Labels", "Embedding Position"],
    ):
        ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=15, alpha=0.4, edgecolors="none")
        ax.set_title(panel_title, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontweight="bold")
        ax.set_ylabel("UMAP 2", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    ax_spec = fig.add_subplot(grid[1, :])
    ax_spec.imshow(segment["spectrograms"].T, aspect="auto", origin="lower", cmap="viridis")
    ax_spec.set_ylabel("Mel Freq. Bin", fontweight="bold")
    ax_spec.set_xticks([])
    ax_spec.set_yticks([])

    for row, colors, label in [(2, gt_colors, "Ground Truth Label"), (3, pos_colors, "Embedding Position")]:
        ax = fig.add_subplot(grid[row, :])
        ax.imshow(colors[np.newaxis, :, :], aspect="auto")
        ax.set_xlabel(label, fontweight="bold")
        ax.set_yticks([])
        ax.set_xticks([])

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{segment['recording_stem']}_event_{segment['song_id']}"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    extracted = extract_recording_embeddings({
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "spec_dir": args.spec_dir,
        "json_path": args.json_path,
        "bird": args.bird,
        "recording_stem": args.recording_stem,
        "recording_mode": "events",
        "encoder_layer_idx": args.encoder_layer_idx,
        "target_feature_type": args.target_feature_type,
        "random_init": args.random_init,
        "max_segments": 1,
    })
    segment = extracted["segments"][0]
    embeddings = zscore(segment["encoded_embeddings"])
    assert len(embeddings) >= 3

    xy = umap.UMAP(
        random_state=args.seed,
        n_neighbors=min(args.neighbors, len(embeddings) - 1),
        min_dist=args.min_dist,
        metric=args.metric,
        low_memory=True,
        n_jobs=-1,
    ).fit_transform(embeddings)
    for path in plot_panel(segment, xy, args.out_dir):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
