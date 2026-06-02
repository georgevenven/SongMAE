#!/usr/bin/env python3
"""Linear-probe individual-ID summaries.

  1. Species x encoder heatmap of mean AP (both orientations).
  2. mAP vs pitch-shift augmentation strength, one line per encoder.

Every species is evaluated on every encoder, so the heatmap is the cleanest
per-cell view; the pitch-shift panel shows how mean AP moves as the train-time
speed/pitch augmentation grows (none -> 0.25 -> 0.5).
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results/individual_id_linear_probe"
OUT_DIR = RESULTS_DIR / "speed0p25_paired_comparison"

SPECIES = [
    ("Bengalese finch", "bf"),
    ("Zebra finch", "zf"),
    ("Little owl", "little_owl"),
    ("Chiffchaff", "chiffchaff"),
    ("Canary", "canary"),
    ("European starling", "european_starling"),
    ("Ovenbird", "ovenbird"),
    ("Tree pipit", "tree_pipit"),
]

# Directory templates with {aug} (augmentation token) and {wh} (window/hop tag).
MODELS = [
    ("SongMAE", "songmae_before_data2vec_step499999_{aug}_windowtoken_pca_whiten_l21024_{wh}"),
    ("AVES", "aves_birdaves_base_{aug}_windowtoken_pca_whiten_l21024_{wh}"),
    ("BirdMAE", "birdmae_bird_mae_base_{aug}_windowtoken_pca_whiten_l21024_{wh}"),
    ("SongMAE\nRandom", "songmae_before_random_init_songmae_seed0_ctx5s_{aug}_windowtoken_pca_whiten_l21024_{wh}"),
    ("HuBERT", "hubert_hubert_base_ls960_{aug}_windowtoken_pca_whiten_l21024_{wh}"),
    ("Perch\n2.0", "perch_perch_v2_{aug}_pca_whiten_l2256_{wh}"),
]

# Pitch-shift augmentation levels: (display label, x value, dir token).
SHIFTS = [
    ("None", 0.0, "clean"),
    ("0.25", 0.25, "speed0p25_trainonly"),
    ("0.5", 0.5, "speed0p5_trainonly"),
]
DEFAULT_AUG = "speed0p25_trainonly"  # used for the heatmaps

# Colour by network family (green = bioacoustic, red = Perch, blue = general/untrained).
MODEL_COLORS = {
    "SongMAE": "#1b7a40",
    "BirdMAE": "#2f9e57",
    "AVES": "#7cc88f",
    "Perch\n2.0": "#d1463c",
    "SongMAE\nRandom": "#8fb4e0",
    "HuBERT": "#3a6fb5",
}
MODEL_MARKERS = {
    "SongMAE": "o", "BirdMAE": "s", "AVES": "^",
    "Perch\n2.0": "D", "SongMAE\nRandom": "v", "HuBERT": "o",
}

INK = "#33373d"


def wh_tag(species_key, model_label):
    if species_key == "european_starling" and model_label == "SongMAE":
        return "w50_h50"
    return "w30_h30"


def run_dir(species_key, model_label, template, aug):
    return RESULTS_DIR / f"{species_key}_{template.format(aug=aug, wh=wh_tag(species_key, model_label))}"


def individual_average_precision(path):
    data = np.load(path, allow_pickle=True)
    y_true = data["y_true"].astype(np.int64, copy=False)
    probs = data["probs"]
    aps = []
    for class_id in np.unique(y_true):
        positives = y_true == class_id
        aps.append(float(average_precision_score(positives.astype(np.int64), probs[:, class_id])))
    return aps, len(aps)


def short(label):
    return label.replace("\n", " ")


def muted_cmap(name, amount=0.45, n=256):
    """Blend a colormap toward white so overlaid black text stays legible."""
    base = plt.get_cmap(name)
    colors = base(np.linspace(0.0, 1.0, n))
    colors[:, :3] = colors[:, :3] + (1.0 - colors[:, :3]) * amount
    return mcolors.ListedColormap(colors)


def load_means(aug):
    """Return (species x model) matrix of per-species mean AP for one aug level."""
    means = np.zeros((len(SPECIES), len(MODELS)), dtype=np.float64)
    n_ind = np.zeros(len(SPECIES), dtype=np.int64)
    for s_idx, (_, species_key) in enumerate(SPECIES):
        for m_idx, (model_label, template) in enumerate(MODELS):
            path = run_dir(species_key, model_label, template, aug) / "val_predictions.npz"
            assert path.exists(), path
            aps, n = individual_average_precision(path)
            means[s_idx, m_idx] = float(np.mean(aps))
            n_ind[s_idx] = n
    return means, n_ind


# --------------------------------------------------------------------------- #
# Load default (0.25) means for the heatmaps, plus every shift level.
# --------------------------------------------------------------------------- #
species_model_means, species_n = load_means(DEFAULT_AUG)
shift_means = {token: load_means(token)[0] for _, _, token in SHIFTS}

model_labels = [label for label, _ in MODELS]
model_order = np.argsort(np.median(species_model_means, axis=0))[::-1]
ordered_labels = [model_labels[i] for i in model_order]
ordered_model_names = [short(l) for l in ordered_labels]
ordered_means = species_model_means[:, model_order]
species_labels = [f"{label} (N={species_n[i]})" for i, (label, _) in enumerate(SPECIES)]

OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "svg.fonttype": "none",
    }
)


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    print(OUT_DIR / f"{name}.png")


# --------------------------------------------------------------------------- #
# Heatmap of mean AP (orientation-agnostic).
# --------------------------------------------------------------------------- #
def draw_heatmap(ax, data, row_labels, col_labels):
    cmap = muted_cmap("RdYlGn", amount=0.45)
    ax.imshow(data, cmap=cmap, vmin=max(0.0, data.min() - 0.05), vmax=1.0, aspect="equal")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=12, fontweight="bold",
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12, fontweight="bold")
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            ax.text(c, r, f"{data[r, c]:.2f}", ha="center", va="center",
                    color="#202020", fontsize=12)
    ax.tick_params(length=0)


# Species as rows, encoders as columns.
fig, ax = plt.subplots(figsize=(7.5, 9.5))
draw_heatmap(ax, ordered_means, species_labels, ordered_model_names)
fig.tight_layout()
save(fig, "panelB_species_model_heatmap")
plt.close(fig)

# Transposed: encoders as rows, species as columns (species names only, no N).
species_names = [label for label, _ in SPECIES]
fig, ax = plt.subplots(figsize=(9.5, 7.5))
draw_heatmap(ax, ordered_means.T, ordered_model_names, species_names)
fig.tight_layout()
save(fig, "panelB_model_species_heatmap")
plt.close(fig)


# --------------------------------------------------------------------------- #
# mAP vs pitch-shift augmentation strength (one line per encoder).
# --------------------------------------------------------------------------- #
def draw_pitch_panel(ax):
    xs = [x for _, x, _ in SHIFTS]
    tokens = [t for _, _, t in SHIFTS]
    for label in ordered_labels:  # high-to-low for legend ordering
        m_idx = model_labels.index(label)
        ys = [shift_means[t][:, m_idx].mean() for t in tokens]
        ax.plot(xs, ys, color=MODEL_COLORS[label], marker=MODEL_MARKERS[label],
                markersize=9, markeredgecolor="white", markeredgewidth=0.8,
                linewidth=2.4, label=short(label), zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([s for s, _, _ in SHIFTS], fontsize=18, fontweight="bold")
    ax.set_xlabel("Pitch-shift factor", fontsize=20, fontweight="bold")
    ax.set_ylabel("mAP", fontsize=20, fontweight="bold")
    ax.set_xlim(-0.05, 0.55)
    ax.set_ylim(0.25, 1.0)
    ax.grid(axis="y", color="#404040", alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_color("#404040")
    ax.tick_params(labelsize=17, length=3.5, width=1.0, colors=INK)
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    ax.legend(fontsize=15, frameon=False, ncol=2, loc="lower center", columnspacing=1.2)


fig, ax = plt.subplots(figsize=(6.0, 9.0))
draw_pitch_panel(ax)
fig.tight_layout()
save(fig, "panelC_map_vs_pitch_shift")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Dump the underlying matrices for reference.
# --------------------------------------------------------------------------- #
with (OUT_DIR / "species_model_mean_ap.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["Species", "N"] + ordered_model_names)
    for i, (label, _) in enumerate(SPECIES):
        writer.writerow([label, int(species_n[i])] + [f"{v:.4f}" for v in ordered_means[i]])
    writer.writerow(["Mean", ""] + [f"{v:.4f}" for v in ordered_means.mean(axis=0)])
print(OUT_DIR / "species_model_mean_ap.csv")

with (OUT_DIR / "map_vs_pitch_shift.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["Model"] + [s for s, _, _ in SHIFTS])
    for label in ordered_labels:
        m_idx = model_labels.index(label)
        writer.writerow([short(label)] + [f"{shift_means[t][:, m_idx].mean():.4f}" for _, _, t in SHIFTS])
print(OUT_DIR / "map_vs_pitch_shift.csv")
