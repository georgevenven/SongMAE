#!/usr/bin/env python3
"""Plot BEANS probe scores against model parameter count."""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "DejaVu Sans"

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evals.beans_table_aggregator import PARAMETERS, table_rows


LABELS = {
    "BEATs pretrained": "BEATs",
    "EAT-base pretrained": "EAT-base pretrained",
    "EAT-all": "EAT-all",
    "Bird-BirdAVES-biox-base": "BirdAVES",
    "Bird-MAE-Huge": "Bird-MAE-Huge",
    "BirdNet": "BirdNET",
}
COLORS = {
    "BEATs pretrained": "#009E73",
    "EAT-base pretrained": "#009E73",
    "EAT-all": "#009E73",
    "Bird-BirdAVES-biox-base": "#009E73",
    "Bird-MAE-Huge": "#009E73",
    "BirdNet": "#222222",
}
SONGMAE = {
    "32×1": ("#0072B2", 2.4, 1.0),
    "32×4": ("#56B4E9", 1.8, 0.72),
}
ANNOTATIONS = {
    "classification": {
        "BEATs pretrained": (20.9, -17.1, "left", "top"),
        "EAT-base pretrained": (-11.3, -29.1, "left", "top"),
        "EAT-all": (16.6, 3.3, "left", "bottom"),
        "Bird-BirdAVES-biox-base": (22.8, -0.7, "left", "center"),
        "Bird-MAE-Huge": (-7, 33.1, "right", "center"),
        "BirdNet": (-26, 0, "right", "center"),
    },
    "detection": {
        "BEATs pretrained": (23.8, 3, "left", "top"),
        "EAT-base pretrained": (-103.2, 0, "left", "center"),
        "EAT-all": (26, 0, "left", "center"),
        "Bird-BirdAVES-biox-base": (26, 0, "left", "center"),
        "Bird-MAE-Huge": (-26, 0, "right", "center"),
        "BirdNet": (-26, 0, "right", "center"),
    },
}


def millions(value, _):
    return f"{value:g}"


def panel(axis, rows, score_index, name, title, ylabel):
    values = []
    for shape, (color, width, alpha) in SONGMAE.items():
        points = sorted(
            (
                PARAMETERS[model] / 1e6,
                (classification, detection)[score_index],
            )
            for _, model, classification, detection in rows
            if model.startswith("SongMAE") and model.endswith(shape)
        )
        x, y = zip(*points)
        values.extend(y)
        axis.plot(
            x, y,
            color=color,
            marker="o",
            markersize=6,
            linewidth=width,
            alpha=alpha,
            label=f"SongMAE {shape}",
            zorder=4,
        )

    for _, model, classification, detection in rows:
        if model.startswith("SongMAE"):
            continue
        score = (classification, detection)[score_index]
        parameters = PARAMETERS[model] / 1e6
        dx, dy, ha, va = ANNOTATIONS[name][model]
        values.append(score)
        axis.scatter(
            parameters,
            score,
            s=52,
            marker="s",
            facecolors=COLORS[model],
            edgecolors=COLORS[model],
            linewidths=0.8,
            zorder=5,
        )
        axis.annotate(
            LABELS[model],
            (parameters, score),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            ha=ha,
            va=va,
            arrowprops={
                "arrowstyle": "-",
                "color": "#777777",
                "linewidth": 0.65,
                "shrinkA": 2,
                "shrinkB": 4,
            },
        )

    margin = (max(values) - min(values)) * 0.12
    axis.set_xscale("log")
    axis.set_xlim(1.3, 700)
    axis.set_ylim(min(values) - margin, max(values) + margin)
    axis.set_title(title, fontsize=13)
    axis.set_xlabel("Parameters (millions)")
    axis.set_ylabel(ylabel)
    axis.set_box_aspect(1)
    axis.set_xticks([3, 10, 30, 100, 300])
    axis.xaxis.set_major_formatter(FuncFormatter(millions))
    axis.grid(alpha=0.18)
    axis.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_root",
        default="/media/george-vengrovski/disk1/avex_runs/"
        "songmae_beans_micro_base_500k/results",
    )
    parser.add_argument("--output_dir", default="imgs/beans")
    args = parser.parse_args()
    rows = [
        row for row in table_rows(Path(args.results_root))
        if row[1] in PARAMETERS
    ]
    assert {row[1] for row in rows} == PARAMETERS.keys()

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.1), dpi=200)
    panel(
        axes[0], rows, 0, "classification", "Classification",
        "Accuracy ↑",
    )
    panel(
        axes[1], rows, 1, "detection", "Detection",
        "mAP ↑",
    )
    axes[1].legend(
        frameon=False,
        loc="lower right",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.15, top=0.91, wspace=0.22)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "beans_score_vs_parameters.png"
    high_resolution = output.with_name(f"{output.stem}_hq.png")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(high_resolution, dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(output)
    print(high_resolution)
    print(output.with_suffix(".pdf"))
    print(output.with_suffix(".svg"))


if __name__ == "__main__":
    main()
