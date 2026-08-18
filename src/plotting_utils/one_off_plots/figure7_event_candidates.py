#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.core.utils import load_spec_slice
from src.embeddings.syllable_umap import build_palette


WINDOW_TOKENS = 1000
TOKEN_MS = 5


def parse_args():
    parser = argparse.ArgumentParser(description="Make Figure 7 candidates from an event-only syllable UMAP.")
    parser.add_argument("--umap_dir", type=Path, required=True)
    parser.add_argument("--index_dir", type=Path, required=True)
    parser.add_argument("--spec_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--min_classes", type=int, choices=[2, 3], default=3)
    parser.add_argument("--min_median_ms", type=float, default=35)
    parser.add_argument("--dot_size", type=float, default=10)
    parser.add_argument("--umap_aspect", type=float, default=1)
    parser.add_argument("--left_scale", type=float, default=1)
    return parser.parse_args()


def run_stats(labels):
    classes = np.unique(labels[labels >= 0])
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    runs = {int(label): [] for label in classes}
    for start, end in zip(cuts[:-1], cuts[1:]):
        label = int(labels[start])
        if label >= 0:
            runs[label].append((end - start) * TOKEN_MS)
    medians = {label: float(np.median(lengths)) for label, lengths in runs.items()}
    totals = {label: int(np.sum(lengths)) for label, lengths in runs.items()}
    counts = {label: len(lengths) for label, lengths in runs.items()}
    return medians, totals, counts


def select_candidates(labels, stems, song_ids, starts_ms, count, min_classes, min_median_ms):
    boundaries = np.r_[
        0,
        np.flatnonzero((stems[1:] != stems[:-1]) | (song_ids[1:] != song_ids[:-1])) + 1,
        len(labels),
    ]
    choices = []
    for event_start, event_end in zip(boundaries[:-1], boundaries[1:]):
        for start in range(event_start, event_end - WINDOW_TOKENS + 1, 5):
            window_labels = labels[start : start + WINDOW_TOKENS]
            classes = np.unique(window_labels[window_labels >= 0])
            if not min_classes <= len(classes) <= 4:
                continue
            medians, totals, occurrences = run_stats(window_labels)
            if min(medians.values()) < min_median_ms or min(occurrences.values()) < 2:
                continue
            durations = np.asarray(list(totals.values()))
            vocal_fraction = float(np.mean(window_labels >= 0))
            balance = float(durations.min() / durations.max())
            score = vocal_fraction * (1 + balance) / 2
            choices.append({
                "score": score,
                "vocal_fraction": vocal_fraction,
                "balance": balance,
                "start": start,
                "stem": str(stems[start]),
                "song_id": int(song_ids[start]),
                "start_ms": float(starts_ms[start]),
                "medians": medians,
                "totals": totals,
                "occurrences": occurrences,
            })

    selected = []
    for choice in sorted(choices, key=lambda row: row["score"], reverse=True):
        overlaps = any(
            choice["stem"] == prior["stem"]
            and choice["song_id"] == prior["song_id"]
            and abs(choice["start_ms"] - prior["start_ms"]) < 2500
            for prior in selected
        )
        if not overlaps:
            selected.append(choice)
        if len(selected) == count:
            break
    assert len(selected) == count, f"found only {len(selected)} candidates"
    return selected


def label_colors(labels, palette):
    return np.asarray([
        [0.25, 0.25, 0.25, 1.0] if label < 0 else [*palette[int(label)], 1.0]
        for label in labels
    ])


def position_colors(xy):
    minimum = xy.min(axis=0)
    span = np.maximum(xy.max(axis=0) - minimum, 1e-12)
    normalized = (xy - minimum) / span
    return np.column_stack([normalized[:, 0], normalized[:, 1], np.full(len(xy), 0.5)])


def plot_candidate(choice, labels, xy, palette, spec_dir, out_dir, dot_size, umap_aspect, left_scale):
    start = choice["start"]
    end = start + WINDOW_TOKENS
    window_labels = labels[start:end]
    window_xy = xy[start:end]
    gt_colors = label_colors(window_labels, palette)
    pos_colors = position_colors(window_xy)

    start_timebin = round(choice["start_ms"] / TOKEN_MS)
    spec = load_spec_slice(spec_dir / f"{choice['stem']}.npy", start_timebin, start_timebin + WINDOW_TOKENS)
    assert spec.shape == (128, WINDOW_TOKENS)

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })
    fig = plt.figure(figsize=(13, 4.8))
    grid = gridspec.GridSpec(
        1,
        3,
        width_ratios=[2, 1, 1],
        wspace=0.18,
        left=0.05,
        right=0.99,
        top=0.92,
        bottom=0.06,
    )
    assert 0 < left_scale <= 1
    left_slot = grid[0, 0]
    if left_scale < 1:
        margin = (1 - left_scale) / 2
        left_slot = left_slot.subgridspec(
            3, 1, height_ratios=[margin, left_scale, margin], hspace=0
        )[1]
    left = left_slot.subgridspec(3, 1, height_ratios=[8, 1, 1], hspace=0.7)

    for ax, colors, title in zip(
        [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])],
        [gt_colors, pos_colors],
        ["Ground Truth Labels", "Embedding Position"],
    ):
        ax.scatter(window_xy[:, 0], window_xy[:, 1], c=colors, s=dot_size, alpha=0.4, edgecolors="none")
        ax.set_box_aspect(umap_aspect)
        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])

    ax_spec = fig.add_subplot(left[0])
    ax_spec.imshow(spec, aspect="auto", origin="lower", cmap="viridis", extent=(0, 5, 0, 128))
    ax_spec.set_xticks(np.arange(6))
    ax_spec.set_yticks([0, 64, 128])
    ax_spec.set_xlabel("Time (s)", labelpad=0)
    ax_spec.set_ylabel("Mels")

    for row, colors, label in [(1, gt_colors, "Ground Truth Label"), (2, pos_colors, "Embedding Position")]:
        ax = fig.add_subplot(left[row])
        ax.imshow(colors[np.newaxis], aspect="auto")
        ax.set_xlabel(label)
        ax.set_xticks([])
        ax.set_yticks([])

    name = f"{choice['stem']}_event_{choice['song_id']}_{int(choice['start_ms'])}ms"
    png = out_dir / f"{name}.png"
    hq = out_dir / f"{name}_hq.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(hq, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return png


def write_contact_sheet(paths, choices, out_dir):
    images = [Image.open(path).convert("RGB") for path in paths]
    width = 1800
    images = [image.resize((width, round(image.height * width / image.width))) for image in images]
    cell_height = max(image.height for image in images) + 70
    columns = min(2, len(images))
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (width * columns, cell_height * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, choice) in enumerate(zip(images, choices)):
        x = index % columns * width
        y = index // columns * cell_height
        sheet.paste(image, (x, y))
        draw.text(
            (x + width // 2, y + image.height + 8),
            f"{index + 1} | {choice['stem']} | min median {min(choice['medians'].values()):.0f} ms",
            fill="black",
            anchor="ma",
        )
    sheet.save(out_dir / "contact_sheet.png")


def compact(values):
    return ";".join(f"{label}:{value:.1f}" for label, value in values.items())


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)

    labels = np.load(args.umap_dir / "labels.npy", mmap_mode="r")
    xy = np.load(args.umap_dir / "umap_points.npy", mmap_mode="r")
    indexed_labels = np.load(args.index_dir / "labels.npy", mmap_mode="r")
    index = args.index_dir / "embeddings"
    stems = np.load(index / "recording_stem.npy", mmap_mode="r")
    song_ids = np.load(index / "song_id.npy", mmap_mode="r")
    starts_ms = np.load(index / "token_start_ms.npy", mmap_mode="r")
    assert np.array_equal(labels, indexed_labels)
    assert len(labels) == len(xy) == len(stems) == len(song_ids) == len(starts_ms)

    choices = select_candidates(
        labels, stems, song_ids, starts_ms, args.count, args.min_classes, args.min_median_ms
    )
    palette = build_palette(labels)
    paths = [
        plot_candidate(
            choice,
            labels,
            xy,
            palette,
            args.spec_dir,
            args.out_dir,
            args.dot_size,
            args.umap_aspect,
            args.left_scale,
        )
        for choice in choices
    ]
    write_contact_sheet(paths, choices, args.out_dir)

    header = (
        "rank\trecording_stem\tsong_event_index\twindow_start_ms\twindow_end_ms\tclass_count\t"
        "vocal_fraction\tminimum_class_median_ms\tclass_median_ms\tclass_occurrences\n"
    )
    rows = []
    for rank, choice in enumerate(choices, 1):
        rows.append("\t".join([
            str(rank),
            choice["stem"],
            str(choice["song_id"]),
            f"{choice['start_ms']:.1f}",
            f"{choice['start_ms'] + WINDOW_TOKENS * TOKEN_MS:.1f}",
            str(len(choice["medians"])),
            f"{choice['vocal_fraction']:.3f}",
            f"{min(choice['medians'].values()):.1f}",
            compact(choice["medians"]),
            compact(choice["occurrences"]),
        ]))
    (args.out_dir / "candidates.tsv").write_text(header + "\n".join(rows) + "\n")
    class_word = "two" if args.min_classes == 2 else "three"
    (args.out_dir / "README.md").write_text(
        "# Figure 7 event-only candidates\n\n"
        "Each candidate is a five-second window wholly inside a detected canary song event. Windows contain "
        f"{class_word} to four vocal syllable classes, each appearing at least twice with a median duration of at least "
        f"{args.min_median_ms:g} ms. Candidates are ranked by vocal coverage and class-duration balance.\n\n"
        "The coordinates are reused directly from the bird-level raw-dimension UMAP generated by "
        "`src/embeddings/syllable_umap.py` from 250,000 detected-event tokens. No full-recording or background "
        "tokens are added, and no candidate-specific UMAP is refit. The displayed points are the chosen window's "
        "tokens in that shared event-only UMAP space, zoomed to the region occupied by the window.\n\n"
        f"UMAP marker area is {args.dot_size:g} points squared and panel height-to-width ratio is "
        f"{args.umap_aspect:g}. The left panel stack uses {args.left_scale:g} of the available height.\n\n"
        "The manuscript figure has not been replaced with any candidate in this folder.\n"
    )


if __name__ == "__main__":
    main()
