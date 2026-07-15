import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from src.core import extract_embedding


RUN_DIR = ROOT / "runs" / "xcl_voronoi_mask_no_normalize_32h_10w_5s_fp8"
OUT_DIR = ROOT / "results" / "extract_embedding_species_svd_grid"
SPECIES = [
    ("canary", Path("/media/george-vengrovski/disk2/specs/canary_64hop_32khz")),
    ("bf", Path("/media/george-vengrovski/disk2/specs/bf_64hop_32khz")),
    ("zf", Path("/media/george-vengrovski/disk2/specs/zf_64hop_32khz")),
]
TIMEBINS = 2500


def args_for(spec_dir, recording_stem):
    return {
        "run_dir": str(RUN_DIR),
        "checkpoint": "model_step_499999.pth",
        "spec_dir": str(spec_dir),
        "json_path": None,
        "bird": None,
        "recording_stem": recording_stem,
        "recording_stems": None,
        "recording_mode": "full_recordings",
        "num_timebins": TIMEBINS,
        "embedding_postprocess": "none",
        "embedding_postprocess_dim": 1024,
        "embedding_postprocess_key": "encoded_embeddings_before_pos_removal",
        "embedding_postprocess_load": None,
        "embedding_postprocess_save": None,
        "encoder_layer_idx": None,
    }


def choose_examples(spec_dir):
    stems = []
    for path in sorted(spec_dir.glob("*.npy")):
        spec = np.load(path, mmap_mode="r")
        if spec.shape[1] >= TIMEBINS:
            stems.append(path.stem)
        if len(stems) == 3:
            return stems
    assert len(stems) == 3


def token_spectrogram(z):
    segment = z["segments"][0]
    spec = segment["spectrograms"]
    token_count = segment["encoded_embeddings_before_pos_removal"].shape[0]
    patch_width = int(z["patch_width"])
    spec = spec[: token_count * patch_width]
    return spec.reshape(token_count, patch_width, spec.shape[1]).mean(axis=1).T


def svd_bars(z):
    x = z["segments"][0]["encoded_embeddings_before_pos_removal"].astype(np.float32, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    return (x @ vt[:2].T).T, singular_values[:2]


OUT_DIR.mkdir(parents=True, exist_ok=True)
model_state = extract_embedding.load_model_state(args_for(SPECIES[0][1], ""))
norm, stats_dir = extract_embedding.get_native_input_normalization(model_state)

items = []
for species, spec_dir in SPECIES:
    for stem in choose_examples(spec_dir):
        args = args_for(spec_dir, stem)
        args["spec_normalization"] = norm
        args["normalization_stats_dir"] = stats_dir
        extracted = extract_embedding.extract_recording_embeddings_with_state(args, model_state)
        items.append((species, stem, extracted))

fig = plt.figure(figsize=(17, 10), constrained_layout=True)
grid = fig.add_gridspec(6, 3, height_ratios=[4, 0.8, 4, 0.8, 4, 0.8])

for i, (species, stem, extracted) in enumerate(items):
    row = (i // 3) * 2
    col = i % 3
    spec = token_spectrogram(extracted)
    bars, singular_values = svd_bars(extracted)

    ax_spec = fig.add_subplot(grid[row, col])
    spec_vmin, spec_vmax = np.percentile(spec, [1, 99.5])
    ax_spec.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=spec_vmin,
        vmax=spec_vmax,
    )
    ax_spec.set_title(stem, fontsize=8)
    ax_spec.set_xticks([])
    if col == 0:
        ax_spec.set_ylabel(f"{species}\nmel")
    else:
        ax_spec.set_yticks([])

    ax_bar = fig.add_subplot(grid[row + 1, col], sharex=ax_spec)
    vmax = np.percentile(np.abs(bars), 99)
    ax_bar.imshow(
        bars,
        aspect="auto",
        origin="upper",
        cmap="coolwarm",
        norm=colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
    )
    ax_bar.set_xticks([])
    if col == 0:
        ax_bar.set_yticks([0, 1], ["SV1", "SV2"])
    else:
        ax_bar.set_yticks([])
    ax_bar.set_xlabel(f"s={singular_values[0]:.1f}, {singular_values[1]:.1f}", fontsize=8)

out_path = OUT_DIR / "canary_bf_zf_3x3_5s_svd_projection_bars.png"
fig.savefig(out_path, dpi=220)
plt.close(fig)
print(out_path)
