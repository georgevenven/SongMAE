import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from src.core import extract_embedding


RUN_DIR = ROOT / "runs" / "xcl_voronoi_mask_no_normalize_32h_10w_5s_fp8"
OUT_DIR = ROOT / "results" / "extract_embedding_species_svd_grid"
SLICE_ROOT = OUT_DIR / "temp_sv1_derivative_slices"
TIMEBINS = 2500
EXAMPLES_PER_DATASET = 30
PLOTS_PER_DATASET = 3
DATASETS = [
    ("xcm_detect_train", Path("/media/george-vengrovski/disk2/specs/xcm_detect_train"), ROOT / "files" / "XCM_train_annotations.json"),
    ("canary", Path("/media/george-vengrovski/disk2/specs/canary_64hop_32khz"), ROOT / "files" / "canary_annotations.json"),
    ("bf", Path("/media/george-vengrovski/disk2/specs/bf_64hop_32khz"), ROOT / "files" / "bf_annotations.json"),
    ("zf", Path("/media/george-vengrovski/disk2/specs/zf_64hop_32khz"), ROOT / "files" / "zf_annotations.json"),
]


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


def load_events(annotation_path):
    data = json.load(open(annotation_path))
    return {
        Path(rec["recording"]["filename"]).stem: rec.get("detected_events", [])
        for rec in data["recordings"]
    }


def ms_to_timebins(ms, audio_params):
    return int((ms / 1000.0) * audio_params[0] / audio_params[2])


def token_song_state(events, audio_params, patch_width, token_count, window_start):
    frames = np.zeros(token_count * patch_width, dtype=np.float32)
    for event in events:
        start = max(0, min(ms_to_timebins(event["onset_ms"], audio_params) - window_start, frames.size))
        end = max(start, min(ms_to_timebins(event["offset_ms"], audio_params) - window_start, frames.size))
        frames[start:end] = 1.0
    return frames.reshape(token_count, patch_width).max(axis=1)


def candidate_starts(events, audio_params, total_timebins):
    starts = {0}
    for event in events[:5]:
        onset = ms_to_timebins(event["onset_ms"], audio_params)
        offset = ms_to_timebins(event["offset_ms"], audio_params)
        midpoint = (onset + offset) // 2
        for start in [onset - 1250, onset - 500, onset - 100, midpoint - 1250, offset - 2400, offset - 1250]:
            starts.add(max(0, min(start, total_timebins - TIMEBINS)))
    return sorted(starts)


def choose_examples(spec_dir, events_by_stem, audio_params):
    examples = []
    for path in sorted(spec_dir.glob("*.npy")):
        if path.stem not in events_by_stem:
            continue
        spec = np.load(path, mmap_mode="r")
        if spec.shape[1] < TIMEBINS:
            continue
        for start in candidate_starts(events_by_stem[path.stem], audio_params, spec.shape[1]):
            state = token_song_state(events_by_stem[path.stem], audio_params, 10, TIMEBINS // 10, start)
            if state.min() == state.max():
                continue
            examples.append((path.stem, start, state))
            break
        if len(examples) == EXAMPLES_PER_DATASET:
            return examples
    assert len(examples) == EXAMPLES_PER_DATASET


def write_slice_spec(spec_dir, dataset, stem, start):
    slice_dir = SLICE_ROOT / dataset
    slice_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(spec_dir / "audio_params.json", slice_dir / "audio_params.json")
    slice_stem = f"{stem}__start{start}"
    out_path = slice_dir / f"{slice_stem}.npy"
    if not out_path.exists():
        spec = np.load(spec_dir / f"{stem}.npy", mmap_mode="r")
        np.save(out_path, np.asarray(spec[:, start : start + TIMEBINS], dtype=np.float32))
    return slice_dir, slice_stem


def token_spectrogram(extracted):
    segment = extracted["segments"][0]
    spec = segment["spectrograms"]
    token_count = segment["encoded_embeddings_before_pos_removal"].shape[0]
    patch_width = int(extracted["patch_width"])
    spec = spec[: token_count * patch_width]
    return spec.reshape(token_count, patch_width, spec.shape[1]).mean(axis=1).T


def top_right_singular_vector(features):
    centered = features - features.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    values, vectors = np.linalg.eigh(cov)
    idx = int(np.argmax(values))
    return vectors[:, idx].astype(np.float32, copy=False), features.mean(axis=0)


def robust_line(values, bottom, height):
    low, high = np.percentile(values, [2, 98])
    if high <= low:
        high = low + 1.0
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return bottom + scaled * height


def plot(items):
    fig, axes = plt.subplots(4, 3, figsize=(18, 11), constrained_layout=True)
    fig.suptitle("SV1 dot product, first derivative, second derivative over spectrograms", fontsize=13)
    colors = {
        "dot": "#22d3ee",
        "d1": "#f97316",
        "d2": "#a78bfa",
    }

    for i, (ax, item) in enumerate(zip(axes.flat, items)):
        dataset, stem, spec, dot, song = item
        d1 = np.gradient(dot)
        d2 = np.gradient(d1)
        spec_vmin, spec_vmax = np.percentile(spec, [1, 99.5])
        ax.imshow(spec, aspect="auto", origin="lower", cmap="magma", vmin=spec_vmin, vmax=spec_vmax)

        token_x = np.arange(dot.size)
        band = spec.shape[0] / 4.5
        ax.plot(token_x, robust_line(dot, band * 3.1, band * 0.9), color=colors["dot"], linewidth=1.2, label="dot")
        ax.plot(token_x, robust_line(d1, band * 1.85, band * 0.9), color=colors["d1"], linewidth=1.0, label="d1")
        ax.plot(token_x, robust_line(d2, band * 0.6, band * 0.9), color=colors["d2"], linewidth=1.0, label="d2")
        ax.fill_between(token_x, 0, 5, where=song > 0, color="#fef08a", alpha=0.85, step="mid")

        ax.set_title(stem, fontsize=8)
        ax.set_xticks([])
        row = i // 3
        col = i % 3
        if col == 0:
            ax.set_ylabel(f"{dataset}\nmel")
        else:
            ax.set_yticks([])
        if row == 0 and col == 2:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.75)

    out_path = OUT_DIR / "sv1_dot_derivative_second_derivative_overlay_30each_first3_per_group.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(out_path)


OUT_DIR.mkdir(parents=True, exist_ok=True)
model_state = extract_embedding.load_model_state(args_for(DATASETS[0][1], ""))
norm, stats_dir = extract_embedding.get_native_input_normalization(model_state)

items = []
plot_features = []
features = []
song_states = []
for dataset, spec_dir, annotation_path in DATASETS:
    events_by_stem = load_events(annotation_path)
    audio = extract_embedding.load_audio_params(spec_dir, require_stats=False)
    audio_params = (audio["sr"], audio["mels"], audio["hop_size"], audio["fft"])
    for i, (stem, start, state) in enumerate(choose_examples(spec_dir, events_by_stem, audio_params)):
        slice_dir, slice_stem = write_slice_spec(spec_dir, dataset, stem, start)
        args = args_for(slice_dir, slice_stem)
        args["spec_normalization"] = norm
        args["normalization_stats_dir"] = stats_dir
        extracted = extract_embedding.extract_recording_embeddings_with_state(args, model_state)
        x = extracted["segments"][0]["encoded_embeddings_before_pos_removal"].astype(np.float32, copy=False)
        features.append(x)
        song_states.append(state[: x.shape[0]])
        if i < PLOTS_PER_DATASET:
            items.append([dataset, stem, token_spectrogram(extracted), None, state[: x.shape[0]]])
            plot_features.append(x)

all_features = np.concatenate(features, axis=0).astype(np.float32, copy=False)
all_song = np.concatenate(song_states)
sv1, mean = top_right_singular_vector(all_features)
all_dot = (all_features - mean) @ sv1
if all_dot[all_song == 1].mean() < all_dot[all_song == 0].mean():
    sv1 *= -1.0

plot_items = []
for item, x in zip(items, plot_features):
    item[3] = (x - mean) @ sv1
    plot_items.append(item)

plot(plot_items)
