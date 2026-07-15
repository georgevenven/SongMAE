import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/syllable_finetune_all_birds_32s_fixed"
OUTPUT = ROOT / "imgs/finetune_curves"
BUDGETS = ("32", "64", "128", "256", "512", "MAX")
SPECIES = (("zf", "Zebra Finch"), ("bf", "Bengalese Finch"), ("canary", "Canary"))
SONGMAE = (
    ("xcl_base_100k_p32x1_c005", "32x1 C=0.05", "#56B4E9", "--"),
    ("xcl_base_100k_p32x4_c010", "32x4 C=0.1", "#0072B2", "-"),
)
COMPARISON = (
    SONGMAE[1],
    ("birdaves_biox_base", "BirdAVES Base", "#D55E00", "--"),
    ("hubert_base_ls960", "HuBERT Base", "#009E73", ":"),
)


def load(path, keys):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {tuple(row[key] for key in keys): float(row["mean_macro_fer"]) for row in rows}


def main():
    species_scores = load(RESULTS / "summary.csv", ("model", "species", "budget"))
    equal_scores = load(RESULTS / "equal_species.csv", ("model", "budget"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    x = range(len(BUDGETS))

    fig, axes = plt.subplots(2, 3, figsize=(8.2, 4.4), dpi=200, sharex=True, sharey=True)
    for row, models in enumerate((SONGMAE, COMPARISON)):
        for ax, (species, title) in zip(axes[row], SPECIES):
            for model, label, color, style in models:
                scores = [species_scores[model, species, budget] for budget in BUDGETS]
                ax.plot(x, scores, marker="o", markersize=3.5, linewidth=2, color=color, linestyle=style, label=label)
            ax.grid(alpha=0.18)
            if row == 0:
                ax.set_title(title, fontsize=11)
        axes[row, 0].set_ylabel("Macro FER (%)")
    for ax in axes[-1]:
        ax.set_xticks(x, BUDGETS)
        ax.tick_params(axis="x", pad=5)
    fig.supxlabel("Training budget (s)", y=0.035)
    for row, y in ((0, 0.73), (1, 0.27)):
        handles, labels = axes[row, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", frameon=False, bbox_to_anchor=(0.81, y), fontsize=8.5)
    fig.subplots_adjust(left=0.08, right=0.81, bottom=0.16, top=0.94, wspace=0.14, hspace=0.18)
    output = OUTPUT / "finetune_by_species.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
    for model, label, color, style in SONGMAE + COMPARISON[1:]:
        scores = [equal_scores[model, budget] for budget in BUDGETS]
        ax.plot(x, scores, marker="o", markersize=4, linewidth=2, color=color, linestyle=style, label=label)
    ax.set_xticks(x, BUDGETS)
    ax.set(xlabel="Training budget (s)", ylabel="Equal-species macro FER (%)")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    output = OUTPUT / "finetune_equal_species.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
