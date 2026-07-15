from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results" / "extract_embedding_zf_whitened"
NPZ_PATHS = [
    OUT_DIR / "B402_43365.65021275_9_22_18_3_41_xcl_5s_fp8_no_whiten.npz",
    OUT_DIR / "R402_43362.25834234_9_19_7_10_34_xcl_5s_fp8_no_whiten.npz",
]


def token_spectrogram(z):
    spec = z["spectrograms"]
    token_count = z["encoded_embeddings_before_pos_removal"].shape[0]
    patch_width = int(z["patch_width"].item())
    spec = spec[: token_count * patch_width]
    return spec.reshape(token_count, patch_width, spec.shape[1]).mean(axis=1).T


def svd_scores(z):
    x = z["encoded_embeddings_before_pos_removal"].astype(np.float32, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T, singular_values[:2]


def plot_one(npz_path):
    z = np.load(npz_path)
    scores, singular_values = svd_scores(z)
    bars = scores.T
    vmax = np.percentile(np.abs(bars), 99)

    fig = plt.figure(figsize=(13, 3.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[5, 0.8], width_ratios=[30, 1])

    ax_spec = fig.add_subplot(grid[0, 0])
    spec = token_spectrogram(z)
    spec_vmin, spec_vmax = np.percentile(spec, [1, 99.5])
    im_spec = ax_spec.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=spec_vmin,
        vmax=spec_vmax,
    )
    ax_spec.set_title(npz_path.name.replace("_xcl_5s_fp8_no_whiten.npz", ""))
    ax_spec.set_ylabel("mel")
    ax_spec.set_xticks([])

    ax_bar = fig.add_subplot(grid[1, 0], sharex=ax_spec)
    im_bar = ax_bar.imshow(
        bars,
        aspect="auto",
        origin="upper",
        cmap="coolwarm",
        norm=colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
    )
    ax_bar.set_yticks([0, 1], ["SV1 / x", "SV2 / y"])
    ax_bar.set_xticks([])

    ax_bar.set_xlabel(f"token, singular values: {singular_values[0]:.2f}, {singular_values[1]:.2f}")

    fig.colorbar(im_spec, cax=fig.add_subplot(grid[0, 1]))
    fig.colorbar(im_bar, cax=fig.add_subplot(grid[1, 1]), label="projection")

    out_path = npz_path.with_suffix(".svd_projection_bars.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(out_path)


for path in NPZ_PATHS:
    plot_one(path)
