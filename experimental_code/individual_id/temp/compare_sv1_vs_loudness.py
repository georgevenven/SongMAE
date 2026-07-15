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
SLICE_ROOT = OUT_DIR / "temp_sv1_loudness_slices"
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
    return spec.reshape(token_count, patch_width, spec.shape[1])


def pearson(x, y):
    x = x.astype(np.float64, copy=False) - x.mean()
    y = y.astype(np.float64, copy=False) - y.mean()
    denom = np.sqrt((x @ x) * (y @ y))
    assert denom > 0.0
    return float((x @ y) / denom)


def auc_score(score, label):
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(score.size, dtype=np.float64) + 1.0
    pos = label == 1
    neg = label == 0
    auc = (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0) / (pos.sum() * neg.sum())
    return float(auc)


def orient(score, label):
    if score[label == 1].mean() < score[label == 0].mean():
        return -score
    return score


def top_right_singular_vector(features):
    centered = features - features.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    values, vectors = np.linalg.eigh(cov)
    idx = int(np.argmax(values))
    return vectors[:, idx].astype(np.float32, copy=False), features.mean(axis=0)


def metric_row(name, score, label):
    score = orient(score.astype(np.float32, copy=False), label)
    return {
        "measure": name,
        "pearson_r": pearson(score, label),
        "auc": auc_score(score, label),
        "song_mean": float(score[label == 1].mean()),
        "silence_mean": float(score[label == 0].mean()),
    }


OUT_DIR.mkdir(parents=True, exist_ok=True)
model_state = extract_embedding.load_model_state(args_for(DATASETS[0][1], ""))
norm, stats_dir = extract_embedding.get_native_input_normalization(model_state)

features = []
labels = []
loudness = {"pixel_mean": [], "pixel_max": [], "pixel_p95": []}
dataset_names = []

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
        spec = token_spectrogram(extracted)
        features.append(x)
        labels.append(state[: x.shape[0]])
        dataset_names.append(np.full((x.shape[0],), dataset))
        loudness["pixel_mean"].append(spec.mean(axis=(1, 2)))
        loudness["pixel_max"].append(spec.max(axis=(1, 2)))
        loudness["pixel_p95"].append(np.percentile(spec, 95, axis=(1, 2)))

all_features = np.concatenate(features, axis=0).astype(np.float32, copy=False)
all_labels = np.concatenate(labels, axis=0).astype(np.float32, copy=False)
all_dataset_names = np.concatenate(dataset_names, axis=0)
sv1, mean = top_right_singular_vector(all_features)
projection = orient((all_features - mean) @ sv1, all_labels)
centered = all_features - mean
cosine = orient((centered @ sv1) / np.maximum(np.linalg.norm(centered, axis=1), 1e-12), all_labels)

measures = {
    "sv1_projection": projection,
    "sv1_cosine": cosine,
}
for key, values in loudness.items():
    measures[key] = np.concatenate(values, axis=0).astype(np.float32, copy=False)

summary = {
    "global": [metric_row(name, score, all_labels) for name, score in measures.items()],
    "by_dataset": {},
}
for dataset, _, _ in DATASETS:
    mask = all_dataset_names == dataset
    summary["by_dataset"][dataset] = [
        metric_row(name, score[mask], all_labels[mask])
        for name, score in measures.items()
    ]

names = [row["measure"] for row in summary["global"]]
pearson_values = [row["pearson_r"] for row in summary["global"]]
auc_values = [row["auc"] for row in summary["global"]]
x = np.arange(len(names))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
axes[0].bar(x, pearson_values, color="#4c78a8")
axes[0].set_xticks(x, names, rotation=25, ha="right")
axes[0].set_ylabel("Pearson r")
axes[0].set_title("Correlation with song state")
axes[1].bar(x, auc_values, color="#f58518")
axes[1].set_xticks(x, names, rotation=25, ha="right")
axes[1].set_ylabel("AUC")
axes[1].set_title("Song vs silence separation")
for ax in axes:
    ax.set_ylim(0, max(pearson_values + auc_values) + 0.08)

plot_path = OUT_DIR / "sv1_vs_loudness_song_state_comparison.png"
summary_path = OUT_DIR / "sv1_vs_loudness_song_state_comparison.json"
fig.savefig(plot_path, dpi=220)
plt.close(fig)
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print(plot_path)
print(summary_path)
