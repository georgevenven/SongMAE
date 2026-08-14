#!/usr/bin/env python3
"""Plot kNN-style linear-probe sweep figures from aggregator TSVs."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SPECIES = ("Zebra", "Bengalese", "Canary")
SPECIES_LABELS = {
    "Zebra": "zebra finch",
    "Bengalese": "Bengalese finch",
    "Canary": "canary",
}
K_MODELS = {
    "SongMAE Large 32x1 (500k)": ("SongMAE-Large 32×1", "#0072B2", "-"),
    "SongMAE Large 32x4 (500k)": ("SongMAE-Large 32×4", "#56B4E9", "-"),
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
    assert set(sections) == set(SPECIES + ("Mean",)), sections
    columns = sections["Zebra"][0]
    assert all(section[0] == columns for section in sections.values())
    return columns, {species: sections[species][1] for species in SPECIES}


def draw_row(axes, columns, sections, models):
    keep = [
        index
        for index in range(len(columns))
        if any(
            sections[species][model][index] is not None
            for species in SPECIES
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
            axes[0].plot(
                [], [], marker="o", markersize=5, linewidth=2,
                color=color, linestyle=style, label=label,
            )[0]
        )

    for axis, species in zip(axes, SPECIES):
        species_values = [
            sections[species][model][index]
            for model in models
            for index in keep
            if sections[species][model][index] is not None
        ]
        upper = max(5, 5 * ((max(species_values) + 4.999) // 5))
        for model, (label, color, style) in models.items():
            values = [sections[species][model][index] for index in keep]
            x = [position for position, value in enumerate(values) if value is not None]
            y = [value for value in values if value is not None]
            axis.plot(
                x, y, marker="o", markersize=4.5, linewidth=2,
                color=color, linestyle=style,
            )
        axis.set_title(SPECIES_LABELS[species], fontsize=11)
        axis.set_ylim(0, upper)
        axis.set_xticks(range(len(labels)), labels)
        axis.grid(alpha=0.18)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Macro FER (%) ↓")
    return handles


def plot_lines(
    columns,
    sections,
    models,
    xlabel,
    output,
    legend_title=None,
):
    fig, axes = plt.subplots(
        1, 3,
        figsize=(8.5, 2.8),
        dpi=200,
        sharex=True,
        sharey=False,
    )
    handles = draw_row(axes, columns, sections, models)
    fig.supxlabel(xlabel, y=0.035)
    axes[-1].legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        frameon=True,
        framealpha=0.78,
        facecolor="white",
        edgecolor="none",
        borderpad=0.3,
        handlelength=1.5,
        title=legend_title,
        title_fontsize=8,
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.92, wspace=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output)
    print(output.with_suffix(".pdf"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("k_table")
    parser.add_argument("checkpoint_32x1_table")
    parser.add_argument("--output_dir", default="imgs/linear_probe")
    parser.add_argument("--checkpoint_budget", choices=["K=5", "K=max"], default="K=5")
    args = parser.parse_args()
    output = Path(args.output_dir)

    k_columns, k_sections = read_tsv(args.k_table)
    for models in k_sections.values():
        for model in models:
            models[model] = models[model][1:]
    k_table = k_columns[1:], k_sections
    checkpoint_table = read_tsv(args.checkpoint_32x1_table)
    suffix = "_kmax" if args.checkpoint_budget == "K=max" else ""
    legend_title = f"32×1 · {args.checkpoint_budget}"
    plot_lines(
        *k_table,
        K_MODELS,
        "Labeled occurrences per class (N)",
        output / "label_budget_points.png",
    )
    plot_lines(
        *checkpoint_table,
        STEP_MODELS,
        "Training step",
        output / f"checkpoint_points_32x1{suffix}.png",
        legend_title=legend_title,
    )


if __name__ == "__main__":
    main()
