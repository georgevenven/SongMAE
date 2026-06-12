"""One-off: how token/label coarseness straddles fine song structure.

The TinyBird story in one figure: the same song labeled at progressively
coarser temporal resolution. At the finest resolution labels track the song
exactly; as cells get coarser they straddle boundaries and the labels degrade.

Two variants are produced:
  - detection:      binary song/silence ground truth (black=silence, green=song).
                    Coarse bars show the per-cell mean, so straddling cells go
                    an intermediate (muddy) shade.
  - classification: each syllable gets a *type* color. Coarse bars take the
                    majority type per cell; timebins where that disagrees with
                    the ground truth are painted red (error), with a per-bar
                    error rate.

There are no annotations for this spec folder, so the "ground truth" is
synthesized: a song/silence track from spectral energy, and syllable types from
KMeans over per-segment mean spectra. It illustrates the concept, it is not real
labeling.

Run:  python -m src.plotting_utils.one_off_plots.token_coarseness
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from sklearn.cluster import KMeans

from src.plotting_utils.plotting_utils import imshow_spec, save_fig

SPEC_DIR = Path("/media/george-vengrovski/disk2/specs/zf_64hop_32khz")
SPEC_FILE = "B236_42619.24528998_9_6_6_48_48.npy"
COARSE_MS = [200, 100, 20, 5]  # bottom bars, coarse -> fine
FINE_MS = 1.0                  # ground-truth grid (spec is 2 ms/timebin -> upsampled)
N_TYPES = 5
CROP_MELS = 6                  # drop the lowest mel bins (noisy low-freq band)
OUT_DIR = Path(__file__).resolve().parents[3] / "imgs"

SPEC_CMAP = "viridis"  # house style: blue -> green -> yellow

# detection variant: black=silence -> green=song
DETECT_CMAP = LinearSegmentedColormap.from_list("song", ["black", "#2ca02c"])

# classification variant: 0=silence(black), 1..N=types, N+1=error(red, reserved).
# Types use a green -> blue palette to match the house spectrogram scheme.
ERROR_COLOR = "#d62728"
TYPE_PALETTE = ["#006d2c", "#41ae76", "#66c2a4", "#3690c0", "#045a8d"]
TYPE_COLORS = ["black"] + TYPE_PALETTE[:N_TYPES] + [ERROR_COLOR]
TYPE_CMAP = ListedColormap(TYPE_COLORS)
ERROR_IDX = N_TYPES + 1

FIGSIZE = (13.0, 6.5)


# --- shared helpers ----------------------------------------------------------

def load_activity():
    """Return (z, ms_per_bin, active) where active is a per-timebin 0/1 track."""
    params = json.loads((SPEC_DIR / "audio_params.json").read_text())
    ms_per_bin = params["hop_size"] / params["sr"] * 1000.0  # 2.0 ms
    spec = np.load(SPEC_DIR / SPEC_FILE)
    z = (spec - params["mean"]) / params["std"]
    z = z[CROP_MELS:, :]  # crop the lowest mel bins

    energy = z.mean(axis=0)
    thr = energy.min() + 0.4 * (energy.max() - energy.min())
    active = (energy > thr).astype(int)
    k = 9  # smooth ~18 ms to drop single-bin flicker
    active = (np.convolve(active, np.ones(k) / k, mode="same") > 0.5).astype(int)
    return z, ms_per_bin, active


def upsample(track, ms_per_bin):
    """Resample a per-timebin track onto the FINE_MS ground-truth grid."""
    up = max(1, round(ms_per_bin / FINE_MS))
    return np.repeat(track, up)


def segments(active):
    """(start, stop) of contiguous active runs."""
    edges = np.diff(np.concatenate([[0], active, [0]]))
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0]
    return list(zip(starts, stops))


def coarsen_mean(track, cell_w):
    """Per-cell mean, broadcast back to full length (continuous)."""
    out = np.empty(len(track), dtype=float)
    for start in range(0, len(track), cell_w):
        stop = min(start + cell_w, len(track))
        out[start:stop] = track[start:stop].mean()
    return out


def coarsen_mode(labels, cell_w):
    """Per-cell majority label, broadcast back to full length (categorical)."""
    out = np.empty_like(labels)
    for start in range(0, len(labels), cell_w):
        stop = min(start + cell_w, len(labels))
        vals, counts = np.unique(labels[start:stop], return_counts=True)
        out[start:stop] = vals[counts.argmax()]
    return out


def strip(ax, row, label, cmap, vmax):
    ax.imshow(row[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
              interpolation="none", extent=(0, len(row), 0, 1))
    ax.set_xlim(0, len(row))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_figure(z, n_coarse):
    n_rows = 2 + n_coarse
    fig = plt.figure(figsize=FIGSIZE)
    grid = fig.add_gridspec(
        n_rows, 1, height_ratios=[6, 0.5] + [0.5] * n_coarse, hspace=0.12,
    )
    axes = [fig.add_subplot(grid[i, 0]) for i in range(n_rows)]
    imshow_spec(axes[0], z, z.shape, cmap=SPEC_CMAP)
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    axes[0].set_ylabel("mels", rotation=0, ha="right", va="center", fontsize=11)
    return fig, axes


def time_axis(ax, n_fine):
    total_s = int(n_fine * FINE_MS / 1000)
    secs = np.arange(0, total_s + 1)
    ax.set_xticks(secs * 1000 / FINE_MS)
    ax.set_xticklabels([str(s) for s in secs])
    ax.set_xlabel("time (s)")


# --- variants ----------------------------------------------------------------

def plot_detection(z, ms_per_bin, active):
    truth = upsample(active.astype(float), ms_per_bin)
    fig, axes = make_figure(z, len(COARSE_MS))
    strip(axes[1], truth, "Human Labels", DETECT_CMAP, vmax=1)
    for ax, ms in zip(axes[2:], COARSE_MS):
        cell_w = max(1, round(ms / FINE_MS))
        strip(ax, coarsen_mean(truth, cell_w), f"{ms} ms", DETECT_CMAP, vmax=1)
    time_axis(axes[-1], truth.size)
    return save_fig(fig, OUT_DIR / "token_coarseness_detection.png")


def plot_classification(z, ms_per_bin, active):
    segs = segments(active)
    feats = np.stack([z[:, s:e].mean(axis=1) for s, e in segs])
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    km = KMeans(n_clusters=min(N_TYPES, len(segs)), n_init=10, random_state=0)
    seg_types = km.fit_predict(feats) + 1  # 1..N (0 reserved for silence)

    labels = np.zeros(len(active), dtype=int)
    for (s, e), t in zip(segs, seg_types):
        labels[s:e] = t
    truth = upsample(labels, ms_per_bin)

    fig, axes = make_figure(z, len(COARSE_MS))
    strip(axes[1], truth, "Human Labels", TYPE_CMAP, vmax=ERROR_IDX)
    for ax, ms in zip(axes[2:], COARSE_MS):
        cell_w = max(1, round(ms / FINE_MS))
        coarse = coarsen_mode(truth, cell_w)
        disp = np.where(coarse != truth, ERROR_IDX, coarse)
        err_pct = 100.0 * (coarse != truth).mean()
        strip(ax, disp, f"{ms} ms\n({err_pct:.0f}% err)", TYPE_CMAP, vmax=ERROR_IDX)
    time_axis(axes[-1], truth.size)
    return save_fig(fig, OUT_DIR / "token_coarseness_classification.png")


def main():
    z, ms_per_bin, active = load_activity()
    d = plot_detection(z, ms_per_bin, active)
    c = plot_classification(z, ms_per_bin, active)
    print(f"wrote {d}")
    print(f"wrote {c}")


if __name__ == "__main__":
    main()
