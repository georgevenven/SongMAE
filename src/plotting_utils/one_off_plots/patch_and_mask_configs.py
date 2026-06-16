"""One-off: how patch shape and the Voronoi mask scale C carve a spectrogram.

The SongMAE ablation story in one collage. A single 5 s canary song is turned
into a 128 mel x 1000 timebin spectrogram (5 ms/timebin, the default), then for
each candidate tokenization (rows) we show the faithful SongMAE Voronoi mask at
each C value (columns), with masked patches blacked out. Lower C -> fewer seeds
-> larger contiguous holes; higher C -> many tiny local masks.

The point of crossing patch shape with C: the *same* C is not the *same* masking
difficulty across tokenizations. Approx masked tokens per Voronoi seed is
0.75 (1 - C) / C, independent of patch shape, but the physical hole size depends
on patch geometry. So 4x4 at C=0.1 is small 2D blobs (easy to interpolate) while
128x1 at C=0.1 is basically temporal dropout.

The masks come straight from src.core.model.SongMAE.voronoi_mask so the figure
matches what training actually does.

Run:  python -m src.plotting_utils.one_off_plots.patch_and_mask_configs
"""

from pathlib import Path

import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt

from src.core.audio2spec import compute_spectrogram
from src.core.model import SongMAE
from src.plotting_utils.plotting_utils import masked_cmap, save_fig, SPEC_DPI

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
PATCH_SHAPES = [(128, 1), (64, 1), (32, 1), (16, 1), (32, 4), (4, 4)]
# Voronoi mask scales — the C interaction ablation.
C_VALUES = [0.05, 0.1, 0.2]
MASK_P = 0.75          # SongMAE default mask fraction
SEED = 0               # deterministic masks

OUT_DIR = Path(__file__).resolve().parents[3] / "imgs"
SPEC_CMAP = "viridis"


def load_spec():
    """Load the 5 s window and return a (N_MELS, n_time) dB spectrogram."""
    wav, _ = librosa.load(AUDIO_FILE, sr=SR, mono=True, offset=START_S, duration=DUR_S)
    n = round(DUR_S * SR)
    wav = wav[:n]
    spec = compute_spectrogram(wav, sr=SR, n_fft=N_FFT, hop_size=HOP, n_mels=N_MELS)
    n_time = round(DUR_S * SR / HOP)          # 1000 timebins for 5 s
    return spec[:, :n_time]


def build_model(patch_shape, n_time):
    """Minimal SongMAE just so we can call the real voronoi_mask method."""
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
              cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def masked_per_seed(c):
    """Approx masked tokens per Voronoi seed: 0.75 (1 - C) / C."""
    return MASK_P * (1.0 - c) / c


def main():
    torch.manual_seed(SEED)
    spec = load_spec()
    vmax = float(spec.max())
    vmin = vmax - 60.0          # 60 dB window for contrast (floor is ~-98 dB)
    mcmap = masked_cmap(SPEC_CMAP)

    n_rows = len(PATCH_SHAPES)
    n_cols = len(C_VALUES)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(14.0 * n_cols, 2.6 * n_rows + 0.8),
        squeeze=False,
    )

    # column headers
    for j, c in enumerate(C_VALUES):
        axes[0][j].set_title(
            f"C = {c:g}\n(~{masked_per_seed(c):.1f} masked tok/seed)",
            fontsize=14, pad=10,
        )

    for i, patch_shape in enumerate(PATCH_SHAPES):
        ph, pw = patch_shape
        H, W = spec.shape[0] // ph, spec.shape[1] // pw
        model = build_model(patch_shape, spec.shape[1])

        # row label: patch shape + resulting token grid
        axes[i][0].set_ylabel(
            f"{ph}x{pw}\n{H}x{W} = {H * W} tok",
            rotation=0, ha="right", va="center", fontsize=14, labelpad=48,
        )

        # faithful Voronoi mask at each C
        for j, c in enumerate(C_VALUES):
            torch.manual_seed(SEED)
            mask_grid = model.voronoi_mask((H, W), p=MASK_P, c=c).cpu().numpy()
            pix_mask = expand_mask(mask_grid, patch_shape, spec.shape)
            shown = np.ma.array(spec, mask=pix_mask)
            show(axes[i][j], shown, vmin, vmax, cmap=mcmap)

    fig.suptitle(
        "SongMAE tokenization x Voronoi mask scale  "
        f"(canary, 5 s, {N_MELS} mel x {spec.shape[1]} timebin, 5 ms/bin, "
        f"masked patches in black, p={MASK_P:g})",
        fontsize=13, y=0.99,
    )
    fig.tight_layout(rect=(0.04, 0, 1, 0.97))
    out = save_fig(fig, OUT_DIR / "patch_and_mask_configs.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
