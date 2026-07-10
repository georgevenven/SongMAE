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

from src.core.utils import load_spec, resolve_single_spec_path
from src.evals.syllable_classification import load_units
from src.plotting_utils.spectrogram_prediction_vs_groundtruth import spec_ms_per_bin


COLORS = {"tp": "#2ca02c", "duplicate": "#e6a700", "fp": "#d62728"}
QUERY = "#1f77b4"
TRUTH = "#00bcd4"
WINDOW_SECONDS = 5


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_song(ax, path, title, truth, detections, focus):
    spec = load_spec(path)
    duration = spec.shape[1] * spec_ms_per_bin(path) / 1000
    center = sum(focus) / 2000
    left = min(max(0, center - WINDOW_SECONDS / 2), max(0, duration - WINDOW_SECONDS))
    right = min(duration, left + WINDOW_SECONDS)
    truth = [(start, end) for start, end in truth if end / 1000 > left and start / 1000 < right]
    detections = [
        row for row in detections
        if float(row["end_ms"]) / 1000 > left and float(row["start_ms"]) / 1000 < right
    ]
    ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0, duration, 0, spec.shape[0]],
        cmap="magma",
        vmin=np.percentile(spec, 2),
        vmax=np.percentile(spec, 99),
    )
    for start, end in truth:
        ax.axvspan(start / 1000, end / 1000, color=TRUTH, alpha=0.2)
        ax.axvspan(start / 1000, end / 1000, facecolor="none", edgecolor=TRUTH, linewidth=1.5)
    for row in detections:
        start, end = float(row["start_ms"]) / 1000, float(row["end_ms"]) / 1000
        color = COLORS[row["status"]]
        ax.axvspan(start, end, facecolor="none", edgecolor=color, linewidth=2)
        ax.text(
            (start + end) / 2,
            0.02,
            f"#{row['rank']}",
            color="white",
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="bottom",
            transform=ax.get_xaxis_transform(),
            clip_on=True,
            bbox={"facecolor": color, "edgecolor": "none", "pad": 0.5},
        )
    ax.set_title(title, loc="left", fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(left, right)


def main():
    parser = argparse.ArgumentParser(description="Show ranked trajectory peaks in five-second song contexts.")
    for name in ("example_dir", "spec_dir", "annotations", "query_id", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_songs", type=int, default=6)
    args = parser.parse_args()
    assert args.top_k > 0 and args.max_songs > 0

    query = [
        row for row in read_rows(Path(args.example_dir) / "queries.csv")
        if row["query_id"] == args.query_id
    ]
    assert len(query) == 1, f"query not found: {args.query_id}"
    query = query[0]
    detections = [
        row for row in read_rows(Path(args.example_dir) / "detections.csv")
        if row["query_id"] == args.query_id and int(row["rank"]) <= args.top_k
    ]
    songs = [
        row for row in read_rows(Path(args.example_dir) / "songs.csv")
        if row["query_id"] == args.query_id
    ]
    songs.sort(key=lambda row: float(row["event_ap"]))
    songs = songs[: args.max_songs]
    recordings = [row["recording"] for row in songs]
    units = load_units(args.annotations)
    label = int(query["label"])
    paths = {
        stem: resolve_single_spec_path(args.spec_dir, stem)
        for stem in [query["query_recording"], *recordings]
    }

    fig, axes = plt.subplots(len(recordings) + 1, 1, figsize=(12, 1.8 * (len(recordings) + 1)), dpi=180)
    query_focus = int(query["query_onset_ms"]), int(query["query_offset_ms"])
    plot_song(axes[0], paths[query["query_recording"]], f"QUERY | {query['query_recording']}", [], [], query_focus)
    start = int(query["query_onset_ms"]) / 1000
    end = int(query["query_offset_ms"]) / 1000
    axes[0].axvspan(start, end, facecolor=QUERY, edgecolor=QUERY, alpha=0.35, linewidth=2)
    axes[0].text(
        (start + end) / 2,
        1.01,
        "query",
        color=QUERY,
        fontsize=7,
        ha="center",
        transform=axes[0].get_xaxis_transform(),
    )
    for ax, song in zip(axes[1:], songs):
        stem = song["recording"]
        truth = [
            (max(start, int(song["start_ms"])), min(end, int(song["end_ms"])))
            for start, end, item_label in units.get(stem, [])
            if item_label == label and start < int(song["end_ms"]) and end > int(song["start_ms"])
        ]
        rows = [row for row in detections if row["recording"] == stem]
        assert truth
        focus = next(
            ((int(row["start_ms"]), int(row["end_ms"])) for row in rows if row["status"] == "tp"),
            truth[0],
        )
        metrics = (
            f"AP {float(song['event_ap']):.2f} | R-prec {float(song['r_precision']):.2f} | "
            f"{song['target_events']} targets"
        )
        plot_song(ax, paths[stem], f"{stem} | {metrics}", truth, rows, focus)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Query class {label} | {len(recordings)} hardest positive songs | "
        "5 s views | blue: query | cyan: truth | green: TP | amber: duplicate | red: FP",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
