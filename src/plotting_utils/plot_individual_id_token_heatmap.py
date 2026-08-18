#!/usr/bin/env python3
"""Plot bird-balanced acoustic context around informative SongMAE tokens."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "Individual_Id_paper_materials/token_analysis"
SPECS = Path("/media/george-vengrovski/disk2/specs/individual_id_multispecies_background_robustness_5ms/zebra_finch/clean")
MODELS = {
    "xcl_large_500k_p32x4_c010": ("SongMAE 32 × 4", 20),
    "xcl_large_500k_p32x1_c005": ("SongMAE 32 × 1", 5),
}
PER_BIRD = 20
SEPARATION_MS = 250
WINDOW_FRAMES = 100
HOP_MS = 5


def load_rows(directory, model, filename, score):
    path = directory / model / filename
    with path.open() as file:
        rows = [
            (row["bird"], row["recording_stem"], (float(row["start_ms"]) + float(row["end_ms"])) / 2, float(row[score]))
            for row in csv.DictReader(file, delimiter="\t")
            if row["kind"] == "song"
        ]
    return sorted(rows, key=lambda row: -row[3])


def select(rows):
    informative = []
    counts = defaultdict(int)
    for row in rows:
        bird, stem, center, _ = row
        if counts[bird] == PER_BIRD:
            continue
        if any(old[0] == bird and old[1] == stem and abs(old[2] - center) < SEPARATION_MS for old in informative):
            continue
        informative.append(row)
        counts[bird] += 1
    assert len(set(counts.values())) == 1 and next(iter(counts.values())) == PER_BIRD

    by_recording = defaultdict(list)
    by_bird = defaultdict(list)
    for row in rows:
        by_recording[row[:2]].append(row)
        by_bird[row[0]].append(row)
    medians = {bird: np.median([row[3] for row in values]) for bird, values in by_bird.items()}
    controls, used = [], set()
    for bird, stem, center, _ in informative:
        candidates = [
            row for row in by_recording[bird, stem]
            if abs(row[2] - center) >= SEPARATION_MS and (row[1], row[2]) not in used
        ]
        if not candidates:
            candidates = [row for row in by_bird[bird] if (row[1], row[2]) not in used]
        control = min(candidates, key=lambda row: abs(row[3] - medians[bird]))
        controls.append(control)
        used.add((control[1], control[2]))
    return informative, controls


def patch(stem, center):
    spec = np.load(SPECS / f"{stem}.npy")
    middle = round(center / HOP_MS)
    first, last = middle - WINDOW_FRAMES, middle + WINDOW_FRAMES
    out = np.full((2 * WINDOW_FRAMES, spec.shape[1]), np.nan, dtype=np.float32)
    source_first, source_last = max(0, first), min(len(spec), last)
    out[source_first - first:source_last - first] = spec[source_first:source_last]
    return out


def mean_context(rows):
    return np.nanmean(np.stack([patch(stem, center) for _, stem, center, _ in rows]), axis=0).T


def render(directory, filename, score, output_name, title):
    values, selections = {}, {}
    for model in MODELS:
        informative, controls = select(load_rows(directory, model, filename, score))
        values[model] = mean_context(informative), mean_context(controls)
        selections[model] = informative, controls

    differences = [top - control for top, control in values.values()]
    limit = max(np.percentile(np.abs(row), 99) for row in differences)
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.2), dpi=200, sharex=True, sharey=True)
    extent = (-WINDOW_FRAMES * HOP_MS, WINDOW_FRAMES * HOP_MS, 0, 128)
    for row, (model, (label, duration)) in enumerate(MODELS.items()):
        informative, control = values[model]
        images = (
            axes[row, 0].imshow(informative, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=-65, vmax=-15),
            axes[row, 1].imshow(control, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=-65, vmax=-15),
            axes[row, 2].imshow(informative - control, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r", vmin=-limit, vmax=limit),
        )
        for axis in axes[row]:
            axis.axvspan(-duration / 2, duration / 2, color="cyan", alpha=0.3)
        axes[row, 0].set_ylabel(f"{label}\nMel bin")
    for axis, column_title in zip(axes[0], ("Most informative", "Matched typical tokens", "Informative − typical")):
        axis.set_title(column_title)
    for axis in axes[-1]:
        axis.set_xlabel("Time from token center (ms)")
    fig.colorbar(images[0], ax=axes[:, :2], label="Mean log-mel power (dB)", shrink=0.85)
    fig.colorbar(images[2], ax=axes[:, 2], label="Difference (dB)", shrink=0.85)
    fig.suptitle(f"{title} · {PER_BIRD} locations per individual", fontsize=13)
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.1, top=0.89, hspace=0.12, wspace=0.08)
    fig.savefig(directory / f"{output_name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{output_name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with (directory / f"{output_name}_selection.tsv").open("w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(("model", "role", "bird", "recording_stem", "center_ms", score))
        for model, pairs in selections.items():
            for role, rows in zip(("informative", "typical_control"), pairs):
                writer.writerows((model, role, bird, stem, f"{center:.3f}", f"{margin:.6f}") for bird, stem, center, margin in rows)


def main():
    render(ANALYSIS, "token_identity_margins.tsv", "identity_margin", "informative_token_acoustic_heatmap", "Bird-balanced DN4-margin contexts")
    render(
        ANALYSIS / "neighborhood_enrichment", "token_enrichment.tsv", "enrichment_k50",
        "token_enrichment_acoustic_heatmap", "Bird-balanced neighborhood-enrichment contexts",
    )


if __name__ == "__main__":
    main()
