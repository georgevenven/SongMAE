"""Draft styling for compact spectrogram + label-stripe figures."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


LABEL_COLORS = np.array(
    [
        [0.05, 0.05, 0.05],
        [0.90, 0.20, 0.20],
        [0.20, 0.65, 0.95],
        [0.30, 0.80, 0.35],
        [0.95, 0.75, 0.20],
        [0.70, 0.35, 0.90],
        [0.95, 0.45, 0.15],
    ],
    dtype=np.float32,
)


def label_rgb(labels):
    idx = np.where(labels < 0, 0, (labels % (len(LABEL_COLORS) - 1)) + 1)
    return LABEL_COLORS[idx][None, :, :]


def save_labeled_spectrogram(spec, labels, title, path):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 3.4),
        dpi=180,
        gridspec_kw={"height_ratios": [12, 1], "hspace": 0.08},
    )
    axes[0].imshow(spec, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title(title, fontsize=10)
    axes[0].set_axis_off()
    axes[1].imshow(label_rgb(labels), aspect="auto", origin="lower")
    axes[1].set_axis_off()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
