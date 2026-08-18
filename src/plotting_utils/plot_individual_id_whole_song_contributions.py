#!/usr/bin/env python3
"""Plot token-probe contributions across one complete pink-noise song bout."""
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.plotting_utils.plot_individual_id_syllable_enrichment import syllables_by_clip


ANALYSIS = ROOT / "Individual_Id_paper_materials/token_analysis/probe_decomposition_songmae_32x4_pink_0db_token"
SPECS = Path("/media/george-vengrovski/disk2/individual_id_pink_noise_same_condition_0db/zebra_finch/spec")
EXAMPLE = "clean_00610"


def main():
    with (ANALYSIS / "token_contributions.tsv").open() as file:
        rows = [row for row in csv.DictReader(file, delimiter="\t") if row["recording_stem"] == EXAMPLE]
    rows.sort(key=lambda row: float(row["start_ms"]))
    starts = np.asarray([float(row["start_ms"]) for row in rows]) / 1000
    ends = np.asarray([float(row["end_ms"]) for row in rows]) / 1000
    centers = (starts + ends) / 2
    contributions = np.asarray([float(row["token_contribution"]) for row in rows])
    duration = ends[-1]
    spec = np.load(SPECS / f"{EXAMPLE}.npy").T
    units = syllables_by_clip()[EXAMPLE]
    limit = np.max(np.abs(contributions))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    fig = plt.figure(figsize=(11, 5.8), dpi=200)
    grid = fig.add_gridspec(3, 2, width_ratios=(1, 0.025), height_ratios=(3.2, 0.28, 1.35), hspace=0.12, wspace=0.12)
    axes = [fig.add_subplot(grid[0, 0])]
    axes.append(fig.add_subplot(grid[1, 0], sharex=axes[0]))
    axes.append(fig.add_subplot(grid[2, 0], sharex=axes[0]))
    colorbar_axis = fig.add_subplot(grid[:, 1])
    axes[0].imshow(spec, origin="lower", aspect="auto", extent=(0, spec.shape[1] * 0.005, 0, 128), cmap="magma", vmin=-65, vmax=-15)
    overlay = np.tile(np.interp(np.arange(spec.shape[1]) * 0.005, centers, contributions), (128, 1))
    axes[0].imshow(overlay, origin="lower", aspect="auto", extent=(0, spec.shape[1] * 0.005, 0, 128), cmap="RdBu_r", norm=norm, alpha=0.14)
    axes[0].set_ylabel("Mel bin")
    axes[0].set_title(f"R425 versus R402 · complete held-out bout · 0 dB pink noise", fontsize=12)
    axes[0].tick_params(labelbottom=False)

    gradient = axes[1].imshow(contributions[None], aspect="auto", extent=(starts[0], ends[-1], 0, 1), cmap="RdBu_r", norm=norm, interpolation="nearest")
    axes[1].set_yticks([])
    axes[1].set_ylabel("m(t)", rotation=0, labelpad=22, va="center")
    axes[1].tick_params(labelbottom=False)

    axes[2].axhline(0, color="#202020", linewidth=1)
    axes[2].plot(centers, contributions, color="#0072B2", linewidth=1.6)
    axes[2].fill_between(centers, 0, contributions, where=contributions >= 0, color="#D55E00", alpha=0.32)
    axes[2].fill_between(centers, 0, contributions, where=contributions < 0, color="#0072B2", alpha=0.28)
    for onset, offset, _ in units:
        axes[2].axvspan(onset / 1000, offset / 1000, color="#E69F00", alpha=0.12)
    axes[2].set(xlabel="Time (s)", ylabel="Token contribution", xlim=(0, duration))
    axes[2].grid(axis="y", alpha=0.18)
    axes[2].text(0.995, 0.95, "orange shading = annotated syllable", transform=axes[2].transAxes, ha="right", va="top", fontsize=8)

    colorbar = fig.colorbar(gradient, cax=colorbar_axis)
    colorbar.set_label("True-bird minus competitor contribution")
    fig.suptitle("Identity evidence through an entire song", fontsize=15)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.1, top=0.88)
    fig.savefig(ANALYSIS / "whole_song_contribution_gradient.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(ANALYSIS / "whole_song_contribution_gradient.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "recording_stem": EXAMPLE,
        "bird": rows[0]["bird"],
        "strongest_competitor": rows[0]["strongest_competitor"],
        "duration_seconds": float(duration),
        "selection": "largest token-contribution range among bouts lasting 1.5 to 3 seconds",
        "minimum_contribution": float(contributions.min()),
        "maximum_contribution": float(contributions.max()),
    }
    (ANALYSIS / "whole_song_contribution_gradient.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
