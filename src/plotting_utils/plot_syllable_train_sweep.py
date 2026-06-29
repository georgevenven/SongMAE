#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from syllable_train_sweep_aggregator import aggregate_rows


SPECIES_ORDER = ["bf", "zf", "canary", "cassins_vireo", "american_robin"]
MODEL_ORDER = [
    "xcl_base_500k_p32x4_default",
    "xcl_base_500k_p16x1_default",
    "xcl_micro_500k_p128x1_default",
    "xcl_micro_500k_p16x1_default",
    "xcl_micro_500k_p32x1_default",
    "xcl_micro_500k_p32x4_default",
    "songmae",
    "songmae_random",
    "aves",
    "hubert",
]
COLORS = {
    "songmae": "#1f77b4",
    "aves": "#d62728",
    "hubert": "#2ca02c",
}
METRIC_LABELS = {"f1": "F1 Score (%)", "fer": "Frame Error Rate (%)"}
METRIC_TITLES = {"f1": "F1", "fer": "FER"}


def model_family(model):
    if model == "aves":
        return "aves"
    if model == "hubert":
        return "hubert"
    return "songmae"


def model_style(model):
    if model == "xcl_base_500k_p32x4_default":
        return "-"
    if model in {"aves", "hubert"}:
        return "-"
    if model == "songmae_random":
        return (0, (1, 1))
    return "--"


def sort_key(train_seconds):
    if train_seconds == "MAX":
        return (1, float("inf"))
    return (0, float(train_seconds))


def read_rows(csv_path, results_root):
    if csv_path is None:
        return aggregate_rows(results_root)
    stream = sys.stdin if csv_path == "-" else Path(csv_path).open("r", encoding="utf-8", newline="")
    with stream:
        return list(csv.DictReader(stream))


def number(row, key):
    return float(row[key])


def ordered_values(rows):
    levels = sorted({row["train_seconds"] for row in rows}, key=sort_key)
    return levels, {value: idx for idx, value in enumerate(levels)}


def ordered_species(rows):
    present = {row["species"] for row in rows}
    species = [item for item in SPECIES_ORDER if item in present]
    species.extend(sorted(present - set(species)))
    return species


def ordered_models(rows):
    present = {row["model"] for row in rows}
    models = [item for item in MODEL_ORDER if item in present]
    models.extend(sorted(present - set(models)))
    return models


def draw_line(ax, rows, model, metric, x_index):
    model_rows = sorted([row for row in rows if row["model"] == model], key=lambda row: sort_key(row["train_seconds"]))
    if not model_rows:
        return None
    xs = [x_index[row["train_seconds"]] for row in model_rows]
    ys = [number(row, metric) for row in model_rows]
    family = model_family(model)
    return ax.plot(
        xs,
        ys,
        marker="o",
        markersize=4.8,
        linewidth=2.2,
        alpha=0.95,
        color=COLORS.get(family, "#666666"),
        linestyle=model_style(model),
        label=model_rows[0]["model_label"],
    )[0]


def draw_panel(ax, rows, species_label, metric, models, x_levels, x_index, show_ylabel, show_xlabel):
    handles = []
    for model in models:
        handle = draw_line(ax, rows, model, metric, x_index)
        if handle is not None:
            handles.append(handle)

    ax.set_title(f"{species_label} - {METRIC_TITLES[metric]}", fontsize=11.5, fontweight="bold")
    ax.grid(True, alpha=0.22)
    ax.set_xlim(-0.25, max(0.25, len(x_levels) - 0.75))
    ax.set_xticks(list(range(len(x_levels))))
    ax.set_xticklabels(x_levels)
    ax.xaxis.set_major_locator(ticker.FixedLocator(list(range(len(x_levels)))))
    ax.tick_params(axis="both", labelsize=10.5, width=1.0)

    if show_xlabel:
        ax.set_xlabel("# Training Seconds", fontsize=10, fontweight="bold")
    else:
        ax.tick_params(axis="x", labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10, fontweight="bold")
        ax.yaxis.set_label_coords(-0.18, 0.5)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)

    for side in ("top", "bottom", "left", "right"):
        spine = ax.spines[side]
        spine.set_linewidth(1.0)
        spine.set_color("#404040")
    return handles


def metric_names(metric):
    if metric == "both":
        return ("f1", "fer")
    return (metric,)


def set_ylim(ax, rows, metric):
    if metric == "f1":
        f1_min = min(number(row, "f1") for row in rows)
        ax.set_ylim(min(40.0, np.floor((f1_min - 1.0) / 5.0) * 5.0), 100.0)
    else:
        fer_max = max(number(row, "fer") for row in rows)
        ax.set_ylim(0.0, max(25.0, np.ceil((fer_max + 1.0) / 5.0) * 5.0))


def save_plot(rows, output, metric):
    species = ordered_species(rows)
    models = ordered_models(rows)
    metrics = metric_names(metric)
    x_levels, x_index = ordered_values(rows)
    fig, axes = plt.subplots(
        len(metrics),
        len(species),
        figsize=(3.2 * len(species), 3.4 * len(metrics)),
        dpi=300,
        squeeze=False,
        sharey="row",
    )

    all_handles = []
    for row_idx, metric_name in enumerate(metrics):
        for col_idx, species_key in enumerate(species):
            ax = axes[row_idx][col_idx]
            panel_rows = [row for row in rows if row["species"] == species_key]
            handles = draw_panel(
                ax,
                panel_rows,
                panel_rows[0]["species_label"],
                metric_name,
                models,
                x_levels,
                x_index,
                show_ylabel=col_idx == 0,
                show_xlabel=row_idx == len(metrics) - 1,
            )
            all_handles.extend(handles)
        set_ylim(axes[row_idx][0], rows, metric_name)

    seen = set()
    legend_handles = []
    for handle in all_handles:
        label = handle.get_label()
        if label in seen:
            continue
        seen.add(label)
        legend_handles.append(handle)
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=min(4, len(legend_handles)),
        frameon=False,
        fontsize=10,
    )

    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.16, top=0.78, wspace=0.12, hspace=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    pdf = output.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return output, pdf


def main():
    parser = argparse.ArgumentParser(description="Plot syllable train-sweep F1/FER curves by species.")
    parser.add_argument("--results_root", default="results/syllable_classification_train_sweep")
    parser.add_argument("--csv", default=None, help="Aggregate CSV path, or '-' for stdin.")
    parser.add_argument("--metric", choices=["both", "f1", "fer"], default="both")
    parser.add_argument("--output", default="imgs/syllable_train_sweep_f1_fer_models.png")
    args = parser.parse_args()

    rows = read_rows(args.csv, args.results_root)
    if not rows:
        raise SystemExit("No rows found.")
    png, pdf = save_plot(rows, Path(args.output), args.metric)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
