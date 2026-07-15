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
SLICE_ROOT = OUT_DIR / "temp_sv1_overlay_slices"
TIMEBINS = 2500
EXAMPLES_PER_DATASET = 30
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


def token_song_state(events, audio_params, patch_width, token_count, window_start=0):
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
            state = token_song_state(events_by_stem[path.stem], audio_params, 10, TIMEBINS // 10, window_start=start)
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


def pearson(x, y):
    x = x.astype(np.float64, copy=False) - x.mean()
    y = y.astype(np.float64, copy=False) - y.mean()
    return float((x @ y) / np.sqrt((x @ x) * (y @ y)))


def plot_grid(items, values_by_item, song_by_item, name, ylabel):
    all_values = np.concatenate(values_by_item)
    vmin, vmax = np.percentile(all_values, [1, 99])
    if vmax <= vmin:
        vmax = vmin + 1.0

    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    grid = fig.add_gridspec(4, 3)
    fig.suptitle(ylabel, fontsize=12)

    for i, ((dataset, stem, extracted), values, song) in enumerate(zip(items[:12], values_by_item[:12], song_by_item[:12])):
        row = i // 3
        col = i % 3
        spec = token_spectrogram(extracted)
        spec_vmin, spec_vmax = np.percentile(spec, [1, 99.5])
        ax = fig.add_subplot(grid[row, col])
        ax.imshow(spec, aspect="auto", origin="lower", cmap="magma", vmin=spec_vmin, vmax=spec_vmax)

        token_x = np.arange(values.size)
        scaled = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
        line_y = 8 + scaled * (spec.shape[0] - 16)
        ax.plot(token_x, line_y, color="#2dd4bf", linewidth=1.4)
        ax.fill_between(token_x, 0, 5, where=song > 0, color="#fef08a", alpha=0.85, step="mid")

        ax.set_title(stem, fontsize=8)
        ax.set_xticks([])
        if col == 0:
            ax.set_ylabel(f"{dataset}\nmel")
        else:
            ax.set_yticks([])

    out_path = OUT_DIR / name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(out_path)


OUT_DIR.mkdir(parents=True, exist_ok=True)
model_state = extract_embedding.load_model_state(args_for(DATASETS[0][1], ""))
norm, stats_dir = extract_embedding.get_native_input_normalization(model_state)

items = []
features = []
song_states = []
for dataset, spec_dir, annotation_path in DATASETS:
    events_by_stem = load_events(annotation_path)
    audio = extract_embedding.load_audio_params(spec_dir, require_stats=False)
    audio_params = (audio["sr"], audio["mels"], audio["hop_size"], audio["fft"])
    for stem, start, state in choose_examples(spec_dir, events_by_stem, audio_params):
        slice_dir, slice_stem = write_slice_spec(spec_dir, dataset, stem, start)
        args = args_for(slice_dir, slice_stem)
        args["spec_normalization"] = norm
        args["normalization_stats_dir"] = stats_dir
        extracted = extract_embedding.extract_recording_embeddings_with_state(args, model_state)
        x = extracted["segments"][0]["encoded_embeddings_before_pos_removal"].astype(np.float32, copy=False)
        items.append((dataset, stem, extracted))
        features.append(x)
        song_states.append(state[: x.shape[0]])

all_features = np.concatenate(features, axis=0).astype(np.float32, copy=False)
all_song = np.concatenate(song_states)
sv1, mean = top_right_singular_vector(all_features)
projection = [(x - mean) @ sv1 for x in features]
all_projection = np.concatenate(projection)
if all_projection[all_song == 1].mean() < all_projection[all_song == 0].mean():
    sv1 *= -1.0
    projection = [-p for p in projection]
    all_projection *= -1.0

cosine = []
for x in features:
    centered = x - mean
    cosine.append((centered @ sv1) / np.maximum(np.linalg.norm(centered, axis=1), 1e-12))

summary = {
    "projection_global_r": pearson(np.concatenate(projection), all_song),
    "cosine_global_r": pearson(np.concatenate(cosine), all_song),
}
summary_path = OUT_DIR / "sv1_overlay_projection_vs_cosine_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n")

plot_grid(
    items,
    projection,
    song_states,
    "sv1_projection_overlay_30each_first3_per_group.png",
    "SV1 centered dot-product projection overlay; yellow strip = song",
)
plot_grid(
    items,
    cosine,
    song_states,
    "sv1_cosine_overlay_30each_first3_per_group.png",
    "SV1 cosine overlay after centering; yellow strip = song",
)
print(summary)
print(summary_path)
