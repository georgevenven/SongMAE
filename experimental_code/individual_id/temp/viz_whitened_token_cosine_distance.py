import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from src.core import extract_embedding


RUN_DIR = ROOT / "runs" / "xcl_voronoi_mask_no_normalize_32h_10w_5s_fp8"
SPEC_DIR = Path("/media/george-vengrovski/disk2/specs/zf_64hop_32khz")
OUT_DIR = ROOT / "results" / "extract_embedding_zf_whitened"
SAMPLES = [
    "B402_43365.65021275_9_22_18_3_41",
    "R402_43362.25834234_9_19_7_10_34",
]
TRANSFORM_PATH = OUT_DIR / "tmp_zf_samples_xcl_5s_fp8_pca_whiten_l2_d1024_transform.npz"


def args_for(recording_stem, transform_mode):
    args = {
        "run_dir": str(RUN_DIR),
        "checkpoint": "model_step_499999.pth",
        "spec_dir": str(SPEC_DIR),
        "json_path": None,
        "bird": None,
        "recording_stem": recording_stem,
        "recording_stems": None,
        "recording_mode": "full_recordings",
        "num_timebins": 0,
        "embedding_postprocess": "pca_whiten_l2",
        "embedding_postprocess_dim": 1024,
        "embedding_postprocess_key": "encoded_embeddings_before_pos_removal",
        "embedding_postprocess_load": None,
        "embedding_postprocess_save": None,
        "encoder_layer_idx": None,
    }
    if transform_mode == "fit":
        args["recording_stem"] = None
        args["recording_stems"] = SAMPLES
        args["embedding_postprocess_save"] = str(TRANSFORM_PATH)
    else:
        assert transform_mode == "load"
        args["npz_dir"] = str(OUT_DIR / f"{recording_stem}_xcl_5s_fp8_pca_whiten_l2_d1024.npz")
        args["embedding_postprocess_load"] = str(TRANSFORM_PATH)
    return args


def token_spectrogram(z):
    spec = z["spectrograms"]
    token_count = z["encoded_embeddings_before_pos_removal"].shape[0]
    patch_width = int(z["patch_width"].item())
    spec = spec[: token_count * patch_width]
    return spec.reshape(token_count, patch_width, spec.shape[1]).mean(axis=1).T


def plot_one(npz_path):
    z = np.load(npz_path)
    x = z["encoded_embeddings_before_pos_removal"].astype(np.float32, copy=False)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    distance = 1.0 - x @ x.T
    offdiag = distance[~np.eye(distance.shape[0], dtype=bool)]
    vmin, vcenter, vmax = np.percentile(offdiag, [1, 50, 99])

    fig = plt.figure(figsize=(11, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 7], width_ratios=[30, 1])

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
    ax_spec.set_title(npz_path.name.replace("_xcl_5s_fp8_pca_whiten_l2_d1024.npz", ""))
    ax_spec.set_ylabel("mel")
    ax_spec.set_xticks([])

    ax_dist = fig.add_subplot(grid[1, 0], sharex=ax_spec)
    im_dist = ax_dist.imshow(
        distance,
        aspect="auto",
        origin="upper",
        cmap="coolwarm",
        norm=colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax),
    )
    ax_dist.set_xlabel("token")
    ax_dist.set_ylabel("token")

    fig.colorbar(im_spec, cax=fig.add_subplot(grid[0, 1]))
    fig.colorbar(
        im_dist,
        cax=fig.add_subplot(grid[1, 1]),
        label=f"1 - cosine similarity, clipped p1-p99 ({vmin:.3f}-{vmax:.3f})",
    )

    out_path = npz_path.with_suffix(".cosine_distance.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(out_path)


OUT_DIR.mkdir(parents=True, exist_ok=True)
model_state = extract_embedding.load_model_state(args_for(None, "load"))
norm, stats_dir = extract_embedding.get_native_input_normalization(model_state)
fit_args = args_for(None, "fit")
fit_args["spec_normalization"] = norm
fit_args["normalization_stats_dir"] = stats_dir
extract_embedding.extract_recording_embeddings_with_state(fit_args, model_state)

for sample in SAMPLES:
    sample_args = args_for(sample, "load")
    sample_args["spec_normalization"] = norm
    sample_args["normalization_stats_dir"] = stats_dir
    extract_embedding.main(sample_args)
    plot_one(Path(sample_args["npz_dir"]))
