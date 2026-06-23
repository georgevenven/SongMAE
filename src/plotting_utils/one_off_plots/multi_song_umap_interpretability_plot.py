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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot several detected events in one SongMAE UMAP.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--spec_dir", required=True)
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "imgs" / "latent_space_interp_multi")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--recording_stem", default=None)
    parser.add_argument("--bird", default=None)
    parser.add_argument("--max_segments", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neighbors", type=int, default=50)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--encoder_layer_idx", type=int, default=None)
    parser.add_argument("--random_init", action="store_true")
    return parser.parse_args()


def label_colors(labels: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("tab20")
    colors = np.zeros((len(labels), 4))
    for index, label in enumerate(labels):
        colors[index] = [0.0, 0.0, 0.0, 0.35] if int(label) == -1 else cmap(int(label) % 20)
    return colors


def position_colors(xy: np.ndarray) -> np.ndarray:
    span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 1e-12)
    norm = (xy - xy.min(axis=0)) / span
    return np.column_stack([norm[:, 0], norm[:, 1], np.full(len(norm), 0.5), np.ones(len(norm))])


def split(array: np.ndarray, counts: list[int]) -> list[np.ndarray]:
    cuts = np.cumsum(counts)[:-1]
    return np.split(array, cuts)


def plot(extracted: dict, xy: np.ndarray, args: argparse.Namespace) -> tuple[Path, Path]:
    segments = extracted["segments"]
    n = len(segments)
    counts = [segment["encoded_embeddings"].shape[0] for segment in segments]
    labels = np.concatenate([segment["labels_downsampled"] for segment in segments])
    label_colors_by_segment = split(label_colors(labels), counts)
    position_colors_by_segment = split(position_colors(xy), counts)
    event_colors = plt.get_cmap("tab20")(np.linspace(0, 1, n))

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "figure.facecolor": "white",
        }
    )

    # Give every panel room to breathe: square UMAP scatters on top, then a
    # tall spectrogram + two color strips per detected event.
    fig = plt.figure(figsize=(20, 10 + 2.6 * n))
    outer = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[10, 2.6 * n],
        hspace=0.12,
        left=0.06,
        right=0.98,
        top=0.97,
        bottom=0.02,
    )

    # Top row: two interpretation views of the same UMAP. Left is colored by
    # ground-truth label (matches the "label" strip); right is colored by 2-D
    # UMAP position (matches the "umap" strip) so each point's color tells you
    # where it sits in the embedding.
    top = outer[0].subgridspec(1, 2, wspace=0.12)
    umap_panels = [
        (top[0, 0], label_colors(labels), "Ground-truth labels\n(matches 'label' strips)"),
        (top[0, 1], position_colors(xy), "UMAP position\n(matches 'umap' strips)"),
    ]
    for spec, colors, title in umap_panels:
        ax = fig.add_subplot(spec)
        ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=16, alpha=0.65, edgecolors="none")
        ax.set_box_aspect(1)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontweight="bold")
        ax.set_ylabel("UMAP 2", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Bottom: one block per event, spaced apart so titles never collide.
    bottom = outer[1].subgridspec(n, 1, hspace=0.55)
    for index, segment in enumerate(segments):
        block = bottom[index].subgridspec(3, 1, height_ratios=[1.0, 0.12, 0.12], hspace=0.06)
        title = f"{index}: {segment['recording_stem']} — event {segment['song_id']}"

        ax = fig.add_subplot(block[0])
        ax.imshow(segment["spectrograms"].T, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(title, loc="left", fontweight="bold", color=event_colors[index])
        ax.set_xticks([])
        ax.set_yticks([])

        for offset, colors, label in [
            (1, label_colors_by_segment[index], "label"),
            (2, position_colors_by_segment[index], "umap"),
        ]:
            ax_strip = fig.add_subplot(block[offset])
            ax_strip.imshow(colors[np.newaxis, :, :], aspect="auto")
            ax_strip.set_ylabel(label, rotation=0, ha="right", va="center", fontweight="bold")
            ax_strip.set_xticks([])
            ax_strip.set_yticks([])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.bird or 'all'}_{len(segments)}events_umap_interp"
    png_path = args.out_dir / f"{name}.png"
    pdf_path = args.out_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    extracted = extract_recording_embeddings(
        {
            "run_dir": args.run_dir,
            "checkpoint": args.checkpoint,
            "spec_dir": args.spec_dir,
            "json_path": args.json_path,
            "bird": args.bird,
            "recording_stem": args.recording_stem,
            "recording_mode": "events",
            "encoder_layer_idx": args.encoder_layer_idx,
            "random_init": args.random_init,
            "max_segments": args.max_segments,
        }
    )
    embeddings = np.concatenate([segment["encoded_embeddings"] for segment in extracted["segments"]])
    assert len(embeddings) >= 3
    # Standardize every latent dimension across the full extraction before UMAP.
    mean = embeddings.mean(axis=0, keepdims=True)
    std = np.maximum(embeddings.std(axis=0, keepdims=True), 1e-8)
    embeddings = (embeddings - mean) / std
    xy = umap.UMAP(
        random_state=args.seed,
        n_neighbors=min(args.neighbors, len(embeddings) - 1),
        min_dist=args.min_dist,
        metric="euclidean",
    ).fit_transform(embeddings)
    for path in plot(extracted, xy, args):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
