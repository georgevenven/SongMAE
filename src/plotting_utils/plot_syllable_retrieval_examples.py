#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.utils import resolve_single_spec_path
from src.plotting_utils.spectrogram_prediction_vs_groundtruth import load_spec_crop


COLORS = {"tp": "#2ca02c", "duplicate": "#e6a700", "fp": "#d62728"}
QUERY = "#1f77b4"


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def crop(path, start, end):
    pad = max(50, (end - start) // 2)
    spec, lo, hi = load_spec_crop(path, start - pad, end + pad)
    return spec, lo, hi, start, end


def show(ax, item, color, vmin, vmax):
    spec, lo, hi, start, end = item
    ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[lo, hi, 0, spec.shape[0]],
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    ax.axvspan(start, end, color=color, alpha=0.22)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(3)


def main():
    parser = argparse.ArgumentParser(description="Plot one evaluated syllable query and its ranked detections.")
    for name in ("example_dir", "spec_dir", "query_id", "recording", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    assert args.top_k > 0

    rows = read_rows(Path(args.example_dir) / "queries.csv")
    query = [item for item in rows if item["query_id"] == args.query_id]
    assert len(query) == 1, f"query not found: {args.query_id}"
    query = query[0]
    detections = [
        item for item in read_rows(Path(args.example_dir) / "detections.csv")
        if item["query_id"] == args.query_id and item["recording"] == args.recording
    ][: args.top_k]
    assert detections, "no saved detections"

    recordings = {query["query_recording"], args.recording}
    paths = {recording: resolve_single_spec_path(args.spec_dir, recording) for recording in recordings}
    crops = [crop(paths[query["query_recording"]], int(query["query_onset_ms"]), int(query["query_offset_ms"]))]
    crops += [crop(paths[item["recording"]], int(item["start_ms"]), int(item["end_ms"])) for item in detections]
    values = np.concatenate([item[0].ravel() for item in crops])
    vmin, vmax = np.percentile(values, [2, 99])

    fig, axes = plt.subplots(1, len(crops), figsize=(2 * len(crops), 3.2), dpi=220)
    show(axes[0], crops[0], QUERY, vmin, vmax)
    axes[0].set_title(f"QUERY\nclass {query['label']}", fontweight="bold", fontsize=10)
    for rank, (item, item_crop, ax) in enumerate(zip(detections, crops[1:], axes[1:]), 1):
        show(ax, item_crop, COLORS[item["status"]], vmin, vmax)
        ax.set_title(f"#{rank} {float(item['score']):.2f}\n{item['status']}", color=COLORS[item["status"]], fontsize=9)
    fig.suptitle(f"Per-song trajectory retrieval | {args.recording}", y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(output)
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
