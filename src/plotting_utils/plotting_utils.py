"""Shared plotting style for TinyBird figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SPEC_FIGSIZE = (10.0, 6.0)
SPEC_DPI = 300
SPEC_IMSHOW_KW = {"origin": "lower", "aspect": "auto", "interpolation": "none"}
SPEC_TITLE_KW = {"fontsize": 18, "fontweight": "bold"}
SPEC_TITLE_Y = 1.15
SPEC_COLORBAR_KW = {"fraction": 0.025, "pad": 0.02}
MASK_CMAP = "viridis"


def masked_cmap(name=MASK_CMAP):
    cmap = plt.get_cmap(name, 256)
    if hasattr(cmap, "with_extremes"):
        return cmap.with_extremes(bad="black")
    cmap = cmap.copy()
    cmap.set_bad("black")
    return cmap


def imshow_spec(ax, image, spec_shape, cmap=None):
    extent = (0, spec_shape[1], 0, spec_shape[0])
    im = ax.imshow(image, extent=extent, cmap=cmap, **SPEC_IMSHOW_KW)
    ax.set_xlim(0, spec_shape[1])
    ax.set_ylim(0, spec_shape[0])
    return im


def style_spec_ax(ax, title):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.5,
        SPEC_TITLE_Y,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        **SPEC_TITLE_KW,
    )


def plot_spec(ax, image, spec_shape, title, cmap=None, colorbars=0):
    assert colorbars in (0, 1, 2)
    im = imshow_spec(ax, image, spec_shape, cmap=cmap)
    style_spec_ax(ax, title)
    for _ in range(colorbars):
        ax.figure.colorbar(im, ax=ax, **SPEC_COLORBAR_KW)
    return im


def save_fig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=SPEC_DPI, bbox_inches="tight")
    plt.close(fig)
    return path
