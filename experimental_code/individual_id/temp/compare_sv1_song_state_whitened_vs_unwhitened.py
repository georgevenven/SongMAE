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
SLICE_ROOT = OUT_DIR / "temp_sv1_comparison_slices"
TIMEBINS = 2500
EXAMPLES_PER_DATASET = 30
DATASETS = [
    ("xcm_detect_train", Path("/media/george-vengrovski/disk2/specs/xcm_detect_train"), ROOT / "files" / "XCM_train_annotations.json"),
    ("canary", Path("/media/george-vengrovski/disk2/specs/canary_64hop_32khz"), ROOT / "files" / "canary_annotations.json"),
    ("bf", Path("/media/george-vengrovski/disk2/specs/bf_64hop_32khz"), ROOT / "files" / "bf_annotations.json"),
    ("zf", Path("/media/george-vengrovski/disk2/specs/zf_64hop_32khz"), ROOT / "files" / "zf_annotations.json"),
]
EXCLUDED = ["tree_pipit", "chiffchaff", "european_starling", "little_owl"]


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


def pearson(x, y):
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x @ x) * (y @ y))
    assert denom > 0.0
    return float((x @ y) / denom)


def top_right_singular_vector(features):
    centered = features - features.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    values, vectors = np.linalg.eigh(cov)
    idx = int(np.argmax(values))
    return vectors[:, idx].astype(np.float32, copy=False), float(np.sqrt(values[idx])), features.mean(axis=0)


def summarize(mode, features_by_item, labels_by_item, item_meta):
    all_features = np.concatenate(features_by_item, axis=0).astype(np.float32, copy=False)
    all_labels = np.concatenate(labels_by_item, axis=0).astype(np.float32, copy=False)
    sv1, singular_value, mean = top_right_singular_vector(all_features)
    scores = (all_features - mean) @ sv1
    if scores[all_labels == 1].mean() < scores[all_labels == 0].mean():
        scores *= -1.0

    by_dataset = []
    for dataset, _, _ in DATASETS:
        parts = []
        label_parts = []
        start = 0
        for meta, labels in zip(item_meta, labels_by_item):
            end = start + labels.size
            if meta["dataset"] == dataset:
                parts.append(scores[start:end])
                label_parts.append(labels)
            start = end
        dataset_scores = np.concatenate(parts)
        dataset_labels = np.concatenate(label_parts)
        by_dataset.append(
            {
                "dataset": dataset,
                "token_count": int(dataset_labels.size),
                "song_fraction": float(dataset_labels.mean()),
                "pearson_r": pearson(dataset_scores, dataset_labels),
                "song_score_mean": float(dataset_scores[dataset_labels == 1].mean()),
                "silence_score_mean": float(dataset_scores[dataset_labels == 0].mean()),
            }
        )

    return {
        "mode": mode,
        "sv1_singular_value": singular_value,
        "global": {
            "token_count": int(all_labels.size),
            "song_fraction": float(all_labels.mean()),
            "pearson_r": pearson(scores, all_labels),
            "song_score_mean": float(scores[all_labels == 1].mean()),
            "silence_score_mean": float(scores[all_labels == 0].mean()),
        },
        "by_dataset": by_dataset,
    }


OUT_DIR.mkdir(parents=True, exist_ok=True)
model_state = extract_embedding.load_model_state(args_for(DATASETS[0][1], ""))
norm, stats_dir = extract_embedding.get_native_input_normalization(model_state)

features = []
labels = []
meta = []
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
        item_features = extracted["segments"][0]["encoded_embeddings_before_pos_removal"].astype(np.float32, copy=False)
        features.append(item_features)
        labels.append(state[: item_features.shape[0]])
        meta.append({"dataset": dataset, "recording": stem, "window_start_timebins": int(start)})

raw_summary = summarize("no_whiten", features, labels, meta)
whiten_transform = extract_embedding.fit_feature_postprocess(
    np.concatenate(features, axis=0),
    mode="pca_whiten_l2",
    dim=1024,
)
whitened_features = [
    extract_embedding.apply_feature_postprocess_transform(item_features, whiten_transform)
    for item_features in features
]
whiten_summary = summarize("pca_whiten_l2_d1024", whitened_features, labels, meta)

summary = {
    "model": str(RUN_DIR),
    "timebins": TIMEBINS,
    "examples_per_dataset": EXAMPLES_PER_DATASET,
    "included_datasets": [dataset for dataset, _, _ in DATASETS],
    "excluded_datasets": EXCLUDED,
    "selection_note": "excluded datasets had no usable 5s mixed song/silence windows under detected_events labels",
    "modes": [raw_summary, whiten_summary],
}

names = ["global"] + [dataset for dataset, _, _ in DATASETS]
raw_values = [raw_summary["global"]["pearson_r"]] + [row["pearson_r"] for row in raw_summary["by_dataset"]]
white_values = [whiten_summary["global"]["pearson_r"]] + [row["pearson_r"] for row in whiten_summary["by_dataset"]]
x = np.arange(len(names))
width = 0.36
fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
ax.axhline(0, color="black", linewidth=0.8)
ax.bar(x - width / 2, raw_values, width, label="no whitening", color="#4c78a8")
ax.bar(x + width / 2, white_values, width, label="pca_whiten_l2_d1024", color="#f58518")
ax.set_xticks(x, names, rotation=20, ha="right")
ax.set_ylabel("Pearson r: SV1 dot product vs song state")
ax.set_title("Global SV1 learned from 30 mixed 5s clips per group")
ax.legend(frameon=False)
ax.set_ylim(min(raw_values + white_values) - 0.1, max(raw_values + white_values) + 0.1)

plot_path = OUT_DIR / "sv1_song_state_whiten_vs_no_whiten_30each_comparison.png"
summary_path = OUT_DIR / "sv1_song_state_whiten_vs_no_whiten_30each_comparison.json"
fig.savefig(plot_path, dpi=220)
plt.close(fig)
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print(plot_path)
print(summary_path)
