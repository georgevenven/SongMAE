"""One-off: how patch shape and the Voronoi seed percentage carve a spectrogram.

The SongMAE ablation story in one landscape figure per patch shape. A single
5 s canary song is turned into a 128 mel x 1000 timebin spectrogram
(5 ms/timebin, the default), then each seed percentage is shown as a row with
masked patches blacked out. Lower percentages produce fewer seeds and larger
contiguous holes; higher percentages produce many small local masks.

The same seed percentage does not produce the same masking difficulty across
tokenizations because physical hole size depends on patch geometry.

The masks come straight from SongMAE's masking methods so the figures match
what training actually does.

Run:  python -m src.plotting_utils.one_off_plots.patch_and_mask_configs
"""

from pathlib import Path

import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt

from src.core.audio2spec import compute_spectrogram
from src.core.model import SongMAE
from src.plotting_utils.plotting_utils import (
    MASK_CMAP,
    PAPER_SPEC_FONT_SIZE,
    PAPER_SPEC_TICK_SIZE,
    masked_cmap,
    save_fig,
)

AUDIO_FILE = Path(
    "/media/george-vengrovski/disk2/raw_data/canary/yarden_data/llb3_songs/"
    "llb3_0002_2018_04_23_14_18_03.wav"
)
START_S = 0.0          # window start within the file
DUR_S = 5.0
SR = 32_000
HOP = 160              # 160 / 32000 = 5 ms per timebin (default)
N_FFT = 1024
N_MELS = 128

# Patch shapes (mel_height, time_width) — the main tokenization ablation.
PATCH_SHAPES = [(128, 1), (32, 1), (16, 1), (32, 4), (4, 4)]
SINGLE_PATCH_SHAPE = (32, 1)
# Voronoi seed percentages used in the interaction ablation.
SEED_PERCENTAGES = [2.5, 5, 10, 20]
MASK_P = 0.75          # SongMAE default mask fraction
SEED = 0               # deterministic masks
FIGSIZE = (8.8, 11.0)

OUT_DIR = Path(__file__).resolve().parents[3] / "imgs" / "patch_and_mask_configs"
SPEC_CMAP = MASK_CMAP


def load_spec():
    """Load the audio window and return a (N_MELS, n_time) dB spectrogram."""
    wav, _ = librosa.load(AUDIO_FILE, sr=SR, mono=True, offset=START_S, duration=DUR_S)
    n = round(DUR_S * SR)
    wav = wav[:n]
    spec = compute_spectrogram(wav, sr=SR, n_fft=N_FFT, hop_size=HOP, n_mels=N_MELS)
    n_time = round(DUR_S * SR / HOP)
    return spec[:, :n_time]


def build_model(patch_shape, n_time):
    """Minimal SongMAE just so we can call its real masking methods."""
    ph, pw = patch_shape
    config = {
        "patch_size": (ph, pw),
        "patch_height": ph,
        "patch_width": pw,
        "mels": N_MELS,
        "num_timebins": n_time,
        "mask_p": MASK_P,
        "mask_c": SEED_PERCENTAGES[1] / 100,
        "enc_hidden_d": 8, "enc_n_head": 1, "enc_dim_ff": 8, "enc_n_layer": 1,
        "dec_hidden_d": 8, "dec_n_head": 1, "dec_dim_ff": 8, "dec_n_layer": 1,
        "dropout": 0.0,
    }
    return SongMAE(config).eval()


def expand_mask(mask_grid, patch_shape, shape):
    """Token-grid bool mask -> pixel-resolution bool mask matching `shape`."""
    ph, pw = patch_shape
    pix = np.repeat(np.repeat(mask_grid, ph, axis=0), pw, axis=1)
    return pix[: shape[0], : shape[1]]


def show(ax, image, vmin, vmax, cmap=SPEC_CMAP):
    ax.imshow(image, origin="lower", aspect="auto", interpolation="none",
              cmap=cmap, vmin=vmin, vmax=vmax, extent=(0, DUR_S, 0, N_MELS))
    ax.set_yticks([0, 64, 128])
    ax.set_ylabel("Mels", fontsize=PAPER_SPEC_FONT_SIZE, labelpad=6)
    ax.tick_params(axis="y", labelsize=PAPER_SPEC_TICK_SIZE, length=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["left"].set_visible(True)


def plot_patch_shape(spec, patch_shape, vmin, vmax, mcmap):
    ph, pw = patch_shape
    H, W = spec.shape[0] // ph, spec.shape[1] // pw
    model = build_model(patch_shape, spec.shape[1])
    fig, axes = plt.subplots(len(SEED_PERCENTAGES), 1, figsize=FIGSIZE, sharex=True)
    for ax, seed_percentage in zip(axes, SEED_PERCENTAGES):
        ax.set_title(
            f"Voronoi Masking, Seed Patches = {seed_percentage:g}%",
            fontsize=PAPER_SPEC_FONT_SIZE, fontweight="normal", pad=5,
        )
        torch.manual_seed(SEED)
        model.mask_c = seed_percentage / 100
        mask_grid = model.voronoi_mask((H, W), "cpu").numpy()
        pix_mask = expand_mask(mask_grid, patch_shape, spec.shape)
        show(ax, np.ma.array(spec, mask=pix_mask), vmin, vmax, cmap=mcmap)

    axes[-1].set_xticks(np.linspace(0, DUR_S, 6))
    axes[-1].set_xlabel("Time (s)", fontsize=PAPER_SPEC_FONT_SIZE, labelpad=5)
    axes[-1].tick_params(axis="x", labelsize=PAPER_SPEC_TICK_SIZE, length=4)
    axes[-1].spines["bottom"].set_visible(True)
    fig.suptitle(
        f"Patch Shape: {ph} mels × {5 * pw} ms",
        fontsize=PAPER_SPEC_FONT_SIZE,
        fontweight="normal",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.8)
    return save_fig(fig, OUT_DIR / f"p{ph}x{pw}.png")


def plot_vertical(spec, random_mask, voronoi_mask, vmin, vmax, mcmap, filename):
    fig, axes = plt.subplots(2, 1, figsize=(17.6, 6.4), sharex=True)
    for ax, mask, title in zip(
        axes,
        (random_mask, voronoi_mask),
        ("Random Masking", "Voronoi Masking"),
    ):
        ax.set_title(title, fontsize=PAPER_SPEC_FONT_SIZE, pad=5)
        image = np.ma.array(
            spec, mask=expand_mask(mask, SINGLE_PATCH_SHAPE, spec.shape)
        )
        show(ax, image, vmin, vmax, cmap=mcmap)
    axes[-1].set_xticks(np.linspace(0, DUR_S, 6))
    axes[-1].set_xlabel("Time (s)", fontsize=PAPER_SPEC_FONT_SIZE, labelpad=5)
    axes[-1].tick_params(axis="x", labelsize=PAPER_SPEC_TICK_SIZE, length=4)
    axes[-1].spines["bottom"].set_visible(True)
    fig.tight_layout(h_pad=1.2)
    return save_fig(fig, OUT_DIR / filename)


def plot_masking(spec, vmin, vmax, mcmap):
    ph, pw = SINGLE_PATCH_SHAPE
    H, W = spec.shape[0] // ph, spec.shape[1] // pw
    model = build_model(SINGLE_PATCH_SHAPE, spec.shape[1])
    torch.manual_seed(SEED)
    random_mask = model.random_mask((H, W), "cpu").numpy()
    torch.manual_seed(SEED)
    model.mask_c = 10 / 100
    voronoi_mask = model.voronoi_mask((H, W), "cpu").numpy()
    return plot_vertical(
        spec, random_mask, voronoi_mask, vmin, vmax, mcmap,
        "p32x1_masking.png",
    )


def main():
    spec = load_spec()
    vmin = float(spec.min())
    vmax = float(spec.max())
    mcmap = masked_cmap(SPEC_CMAP)
    for patch_shape in PATCH_SHAPES:
        print(f"wrote {plot_patch_shape(spec, patch_shape, vmin, vmax, mcmap)}")
    print(f"wrote {plot_masking(spec, vmin, vmax, mcmap)}")


if __name__ == "__main__":
    main()
