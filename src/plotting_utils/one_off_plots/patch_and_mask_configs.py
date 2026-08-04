"""One-off: how patch shape and the Voronoi mask scale C carve a spectrogram.

The SongMAE ablation story in one landscape figure per patch shape. A single
5 s canary song is turned into a 128 mel x 1000 timebin spectrogram
(5 ms/timebin, the default), then each C value is shown as a row with masked
patches blacked out. Lower C -> fewer seeds -> larger contiguous holes; higher
C -> many tiny local masks.

The point of crossing patch shape with C: the *same* C is not the *same* masking
difficulty across tokenizations. Approx masked tokens per Voronoi seed is
0.75 (1 - C) / C, independent of patch shape, but the physical hole size depends
on patch geometry. So 4x4 at C=0.1 is small 2D blobs (easy to interpolate) while
128x1 at C=0.1 is basically temporal dropout.

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
from src.plotting_utils.plotting_utils import MASK_CMAP, masked_cmap, save_fig

AUDIO_FILE = Path(
    "/media/george-vengrovski/disk2/raw_data/canary/yarden_data/llb3_songs/"
    "llb3_0002_2018_04_23_14_18_03.wav"
)
START_S = 0.0          # window start within the file
DUR_S = 5.0            # 5 s window
SR = 32_000
HOP = 160              # 160 / 32000 = 5 ms per timebin (default)
N_FFT = 1024
N_MELS = 128

# Patch shapes (mel_height, time_width) — the main tokenization ablation.
PATCH_SHAPES = [(128, 1), (32, 1), (16, 1), (32, 4), (4, 4)]
SINGLE_PATCH_SHAPE = (32, 1)
# Voronoi mask scales — the C interaction ablation.
C_VALUES = [0.025, 0.05, 0.1, 0.2]
MASK_P = 0.75          # SongMAE default mask fraction
SEED = 0               # deterministic masks
FIGSIZE = (8.8, 11.0)

OUT_DIR = Path(__file__).resolve().parents[3] / "imgs" / "patch_and_mask_configs"
SPEC_CMAP = MASK_CMAP


def load_spec():
    """Load the 5 s window and return a (N_MELS, n_time) dB spectrogram."""
    wav, _ = librosa.load(AUDIO_FILE, sr=SR, mono=True, offset=START_S, duration=DUR_S)
    n = round(DUR_S * SR)
    wav = wav[:n]
    spec = compute_spectrogram(wav, sr=SR, n_fft=N_FFT, hop_size=HOP, n_mels=N_MELS)
    n_time = round(DUR_S * SR / HOP)          # 1000 timebins for 5 s
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
        "mask_c": C_VALUES[1],
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
    ax.set_ylabel("Mels", fontsize=13, labelpad=6)
    ax.tick_params(axis="y", labelsize=11, length=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["left"].set_visible(True)


def plot_patch_shape(spec, patch_shape, vmin, vmax, mcmap):
    ph, pw = patch_shape
    H, W = spec.shape[0] // ph, spec.shape[1] // pw
    model = build_model(patch_shape, spec.shape[1])
    fig, axes = plt.subplots(len(C_VALUES), 1, figsize=FIGSIZE, sharex=True)
    for ax, c in zip(axes, C_VALUES):
        ax.set_title(
            f"Voronoi Masking, Percent Seed Patches (C) = {c:g}",
            fontsize=14, pad=5,
        )
        torch.manual_seed(SEED)
        model.mask_c = c
        mask_grid = model.voronoi_mask((H, W), "cpu").numpy()
        pix_mask = expand_mask(mask_grid, patch_shape, spec.shape)
        show(ax, np.ma.array(spec, mask=pix_mask), vmin, vmax, cmap=mcmap)

    axes[-1].set_xticks(np.arange(0, DUR_S + 1))
    axes[-1].set_xlabel("Time (s)", fontsize=14, labelpad=5)
    axes[-1].tick_params(axis="x", labelsize=12, length=4)
    axes[-1].spines["bottom"].set_visible(True)
    unit = "timebin" if pw == 1 else "timebins"
    fig.suptitle(f"Patch Shape: {ph} mels × {pw} {unit}", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.8)
    return save_fig(fig, OUT_DIR / f"p{ph}x{pw}.png")


def plot_single(spec, mask, title, filename, vmin, vmax, mcmap):
    fig, ax = plt.subplots(figsize=(8.8, 3.2))
    ax.set_title(title, fontsize=14, pad=5)
    show(ax, np.ma.array(spec, mask=expand_mask(mask, SINGLE_PATCH_SHAPE, spec.shape)),
         vmin, vmax, cmap=mcmap)
    ax.set_xticks(np.arange(0, DUR_S + 1))
    ax.set_xlabel("Time (s)", fontsize=14, labelpad=5)
    ax.tick_params(axis="x", labelsize=12, length=4)
    ax.spines["bottom"].set_visible(True)
    fig.tight_layout()
    return save_fig(fig, OUT_DIR / filename)


def plot_random(spec, vmin, vmax, mcmap):
    ph, pw = SINGLE_PATCH_SHAPE
    H, W = spec.shape[0] // ph, spec.shape[1] // pw
    model = build_model(SINGLE_PATCH_SHAPE, spec.shape[1])
    torch.manual_seed(SEED)
    mask = model.random_mask((H, W), "cpu").numpy()
    return plot_single(spec, mask, f"Random Masking, {MASK_P:.0%} Masked Patches",
                       "p32x1_random.png", vmin, vmax, mcmap)


def plot_c010(spec, vmin, vmax, mcmap):
    ph, pw = SINGLE_PATCH_SHAPE
    H, W = spec.shape[0] // ph, spec.shape[1] // pw
    model = build_model(SINGLE_PATCH_SHAPE, spec.shape[1])
    torch.manual_seed(SEED)
    model.mask_c = 0.1
    mask = model.voronoi_mask((H, W), "cpu").numpy()
    return plot_single(spec, mask, "Voronoi Masking, Percent Seed Patches (C) = 0.1",
                       "p32x1_c010.png", vmin, vmax, mcmap)


def main():
    spec = load_spec()
    vmin = float(spec.min())
    vmax = float(spec.max())
    mcmap = masked_cmap(SPEC_CMAP)
    for patch_shape in PATCH_SHAPES:
        print(f"wrote {plot_patch_shape(spec, patch_shape, vmin, vmax, mcmap)}")
    print(f"wrote {plot_random(spec, vmin, vmax, mcmap)}")
    print(f"wrote {plot_c010(spec, vmin, vmax, mcmap)}")


if __name__ == "__main__":
    main()
