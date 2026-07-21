#!/usr/bin/env python3
"""Plot kNN-style linear-probe sweep figures from aggregator TSVs."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SPECIES = ("Zebra", "Bengalese", "Canary")
PANEL_SPECIES = SPECIES + ("Mean",)
SPECIES_LABELS = {
    "Zebra": "Zebra Finch",
    "Bengalese": "Bengalese Finch",
    "Canary": "Canary",
    "Mean": "Mean across species",
}
K_MODELS = {
    "SongMAE Large 32x1 (500k)": ("SongMAE 32×1", "#0072B2", "-"),
    "SongMAE Large 32x4 (500k)": ("SongMAE 32×4", "#56B4E9", "-"),
    "BirdAVES": ("BirdAVES", "#D55E00", "--"),
    "HuBERT base": ("HuBERT", "#009E73", ":"),
}
STEP_MODELS = {
    "SongMAE Large 32x1": ("Large", "#440154", "-"),
    "SongMAE Base 32x1": ("Base", "#21918C", "--"),
    "SongMAE Micro 32x1": ("Micro", "#FDE725", ":"),
}


def read_tsv(path):
    with Path(path).open() as file:
        rows = list(csv.reader(file, delimiter="\t"))
    sections = {}
    index = 0
    while index < len(rows):
        if not rows[index]:
            index += 1
            continue
        species = rows[index][0].rsplit(" - ", 1)[-1]
        species = "Mean" if species == "Mean across species" else species
        columns = rows[index + 1][1:]
        index += 2
        data = {}
        while index < len(rows) and rows[index]:
            data[rows[index][0]] = [
                None if cell == "-" else float(cell.split()[0])
                for cell in rows[index][1:]
            ]
            index += 1
        sections[species] = columns, data
    assert set(sections) == set(PANEL_SPECIES), sections
    columns = sections["Zebra"][0]
    assert all(section[0] == columns for section in sections.values())
    return columns, {species: sections[species][1] for species in PANEL_SPECIES}


def draw_row(axes, columns, sections, models, panel_species):
    keep = [
        index
        for index in range(len(columns))
        if any(
            sections[species][model][index] is not None
            for species in panel_species
            for model in models
        )
    ]
    labels = [
        columns[index].removesuffix(" Macro FER (P/I)").removeprefix("K=")
        for index in keep
    ]
    handles = []
    for model, (label, color, style) in models.items():
        handles.append(
            axes[0].plot([], [], marker="o", markersize=5, linewidth=2, color=color, linestyle=style, label=label)[0]
        )

    all_values = [
        sections[species][model][index]
        for species in panel_species
        for model in models
        for index in keep
        if sections[species][model][index] is not None
    ]
    upper = max(5, 5 * ((max(all_values) + 4.999) // 5))
    for axis, species in zip(axes, panel_species):
        for model, (label, color, style) in models.items():
            values = [sections[species][model][index] for index in keep]
            x = [position for position, value in enumerate(values) if value is not None]
            y = [value for value in values if value is not None]
            axis.plot(x, y, marker="o", markersize=4.5, linewidth=2, color=color, linestyle=style)
        axis.set_title(SPECIES_LABELS[species], fontsize=11)
        axis.set_ylim(0, upper)
        axis.set_xticks(range(len(labels)), labels)
        axis.grid(alpha=0.18)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Macro FER (%) ↓")
    return labels, handles


def plot_lines(columns, sections, models, xlabel, output, panel_species=SPECIES):
    width = 7.2 if len(panel_species) == 3 else 9.5
    fig, axes = plt.subplots(1, len(panel_species), figsize=(width, 2.8), dpi=200, sharex=True, sharey=True)
    labels, handles = draw_row(axes, columns, sections, models, panel_species)
    fig.supxlabel(xlabel, y=0.035)
    legend_axis = axes[2] if len(panel_species) == 4 else axes[-1]
    legend_axis.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        frameon=True,
        framealpha=0.78,
        facecolor="white",
        edgecolor="none",
        borderpad=0.3,
        handlelength=1.5,
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.92, wspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output)
    print(output.with_suffix(".pdf"))


def plot_stacked(k_table, checkpoint_table, output, panel_species=SPECIES):
    k_columns, k_sections = k_table
    step_columns, step_sections = checkpoint_table
    width = 7.2 if len(panel_species) == 3 else 9.5
    fig, axes = plt.subplots(2, len(panel_species), figsize=(width, 5.5), dpi=200, sharey="row")
    _, k_handles = draw_row(axes[0], k_columns, k_sections, K_MODELS, panel_species)
    _, step_handles = draw_row(axes[1], step_columns, step_sections, STEP_MODELS, panel_species)
    legend_index = 2 if len(panel_species) == 4 else len(panel_species) - 1
    for axis, handles in ((axes[0, legend_index], k_handles), (axes[1, legend_index], step_handles)):
        axis.legend(
            handles=handles,
            loc="upper right",
            fontsize=8,
            frameon=True,
            framealpha=0.78,
            facecolor="white",
            edgecolor="none",
            borderpad=0.3,
            handlelength=1.5,
        )
    fig.text(0.5, 0.02, "Training step", ha="center", va="bottom")
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.1, top=0.95, wspace=0.08, hspace=0.62)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output)
    print(output.with_suffix(".pdf"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("k_table")
    parser.add_argument("checkpoint_table")
    parser.add_argument("--output_dir", default="imgs/linear_probe")
    args = parser.parse_args()
    output = Path(args.output_dir)

    k_table = read_tsv(args.k_table)
    checkpoint_table = read_tsv(args.checkpoint_table)
    plot_lines(*k_table, K_MODELS, "Labeled occurrences per class (K)", output / "label_budget_points.png")
    plot_lines(*checkpoint_table, STEP_MODELS, "Training step", output / "checkpoint_points.png")
    plot_stacked(k_table, checkpoint_table, output / "stacked_points.png")
    plot_lines(*k_table, K_MODELS, "Labeled occurrences per class (K)", output / "label_budget_points_mean.png", PANEL_SPECIES)
    plot_lines(*checkpoint_table, STEP_MODELS, "Training step", output / "checkpoint_points_mean.png", PANEL_SPECIES)
    plot_stacked(k_table, checkpoint_table, output / "stacked_points_mean.png", PANEL_SPECIES)


if __name__ == "__main__":
    main()
