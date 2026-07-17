#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.evals.micro_model_linear_probe_table_aggregator import SECTIONS
from src.evals.songmae_vs_other_linear_probe_table_aggregator import ROWS
from src.plotting_utils.plot_linear_probe_tables import SECTION_TITLES, tick_label


SPECIES = {
    "canary": ("Canary", "#D45087"),
    "zf": ("Zebra Finch", "#7A5195"),
    "bf": ("Bengalese Finch", "#2A9D8F"),
}
OUTPUTS = {
    "micro": Path("imgs/linear_probe/micro_ablations_individual_birds.png"),
    "models": Path("imgs/linear_probe/model_comparison_individual_birds.png"),
}


def load_results(root):
    values = {}
    for path in sorted(root.glob("*/*/*/metrics.json")):
        species, bird, model = path.parts[-4:-1]
        values[model, species, bird] = 100 * json.loads(path.read_text())["macro_fer"]
    assert values
    return values


def configs(table, values):
    available = {key[0] for key in values}
    if table == "micro":
        return [
            (SECTION_TITLES[title], [(label, model) for label, model in rows if model in available])
            for title, rows in SECTIONS
        ]
    rows = [(label, model) for label, model in ROWS if model in available]
    return [("Model Comparison", rows)]


def mean_sem(values):
    assert len(values) > 1
    return np.mean(values), np.std(values, ddof=1) / np.sqrt(len(values))


def plot(table, values):
    panels = configs(table, values)
    width = 2.7 * len(panels) if table == "micro" else 4
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 4), squeeze=False, sharey=True)
    upper = 0

    for ax, (title, rows) in zip(axes[0], panels):
        for species, (label, color) in SPECIES.items():
            summaries = []
            for _, model in rows:
                bird_values = [value for (name, key, _), value in values.items() if name == model and key == species]
                summaries.append(mean_sem(bird_values) if bird_values else (np.nan, np.nan))
            means, errors = np.array(summaries).T
            upper = max(upper, np.nanmax(means + errors))
            ax.errorbar(
                range(len(rows)),
                means,
                yerr=errors,
                color=color,
                marker="o",
                markersize=4,
                linewidth=2,
                capsize=3,
                label=label,
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(rows)), [tick_label(label) for label, _ in rows], fontsize=8)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.set_box_aspect(1)

    axes[0, 0].set_ylim(0, 5 * math.ceil(upper / 5))
    axes[0, 0].set_ylabel("Macro FER (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Mean ± SEM", loc="upper center", ncol=3, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.07 if table == "micro" else 0.18, right=0.995, bottom=0.14, top=0.78, wspace=0.12)

    output = OUTPUTS[table]
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output)
    print(output.with_suffix(".pdf"))


def main():
    parser = argparse.ArgumentParser(description="Plot individual-bird linear-probe results.")
    parser.add_argument("table", choices=OUTPUTS)
    parser.add_argument("results_root", type=Path)
    args = parser.parse_args()
    plot(args.table, load_results(args.results_root))


if __name__ == "__main__":
    main()
