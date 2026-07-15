import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from src.core import extract_embedding


RUN_DIR = ROOT / "runs" / "xcl_voronoi_mask_no_normalize_32h_10w_5s_fp8"
OUT_DIR = ROOT / "results" / "extract_embedding_species_svd_grid"
TIMEBINS = 2500
DATASETS = [
    (
        "xcm_detect_train",
        Path("/media/george-vengrovski/disk2/specs/xcm_detect_train"),
        ROOT / "files" / "XCM_train_annotations.json",
    ),
    (
        "canary",
        Path("/media/george-vengrovski/disk2/specs/canary_64hop_32khz"),
        ROOT / "files" / "canary_annotations.json",
    ),
    (
        "bf",
        Path("/media/george-vengrovski/disk2/specs/bf_64hop_32khz"),
        ROOT / "files" / "bf_annotations.json",
    ),
    (
        "zf",
        Path("/media/george-vengrovski/disk2/specs/zf_64hop_32khz"),
        ROOT / "files" / "zf_annotations.json",
    ),
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


def choose_examples(spec_dir, events_by_stem, audio_params):
    stems = []
    for path in sorted(spec_dir.glob("*.npy")):
        if path.stem not in events_by_stem:
            continue
        spec = np.load(path, mmap_mode="r")
        if spec.shape[1] >= TIMEBINS:
            state = token_song_state(
                events_by_stem[path.stem],
                audio_params,
                patch_width=10,
                token_count=TIMEBINS // 10,
            )
            if state.min() == state.max():
                continue
            stems.append(path.stem)
        if len(stems) == 3:
            return stems
    assert len(stems) == 3


def ms_to_timebins(ms, audio_params):
    return int((ms / 1000.0) * audio_params[0] / audio_params[2])


def token_song_state(events, audio_params, patch_width, token_count):
    frames = np.zeros(token_count * patch_width, dtype=np.float32)
    for event in events:
        start = max(0, min(ms_to_timebins(event["onset_ms"], audio_params), frames.size))
        end = max(start, min(ms_to_timebins(event["offset_ms"], audio_params), frames.size))
        frames[start:end] = 1.0
    return frames.reshape(token_count, patch_width).max(axis=1)


def pearson(x, y):
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x @ x) * (y @ y))
    if denom == 0.0:
        return None
    return float((x @ y) / denom)


def mean_or_none(values):
    if values.size == 0:
        return None
    return float(values.mean())


OUT_DIR.mkdir(parents=True, exist_ok=True)
model_state = extract_embedding.load_model_state(args_for(DATASETS[0][1], ""))
norm, stats_dir = extract_embedding.get_native_input_normalization(model_state)

items = []
for dataset, spec_dir, annotation_path in DATASETS:
    events_by_stem = load_events(annotation_path)
    audio = extract_embedding.load_audio_params(spec_dir, require_stats=False)
    audio_params = (audio["sr"], audio["mels"], audio["hop_size"], audio["fft"])
    for stem in choose_examples(spec_dir, events_by_stem, audio_params):
        args = args_for(spec_dir, stem)
        args["spec_normalization"] = norm
        args["normalization_stats_dir"] = stats_dir
        extracted = extract_embedding.extract_recording_embeddings_with_state(args, model_state)
        segment = extracted["segments"][0]
        features = segment["encoded_embeddings_before_pos_removal"].astype(np.float32, copy=False)
        labels = token_song_state(
            events_by_stem[stem],
            extracted["audio_params"],
            int(extracted["patch_width"]),
            features.shape[0],
        )
        items.append((dataset, stem, features, labels))

all_features = np.concatenate([features for _, _, features, _ in items], axis=0)
all_labels = np.concatenate([labels for _, _, _, labels in items], axis=0)
mean = all_features.mean(axis=0, keepdims=True)
_, singular_values, vt = np.linalg.svd(all_features - mean, full_matrices=False)
scores = (all_features - mean) @ vt[0]
if scores[all_labels == 1].mean() < scores[all_labels == 0].mean():
    scores *= -1.0

summary = {
    "model": str(RUN_DIR),
    "timebins": TIMEBINS,
    "patch_width": int(model_state["patch_width"]),
    "sv1_singular_value": float(singular_values[0]),
    "global": {
        "token_count": int(all_labels.size),
        "song_fraction": float(all_labels.mean()),
        "pearson_r": pearson(scores, all_labels),
        "song_score_mean": mean_or_none(scores[all_labels == 1]),
        "silence_score_mean": mean_or_none(scores[all_labels == 0]),
    },
    "by_dataset": [],
    "by_recording": [],
}

start = 0
for dataset, stem, features, labels in items:
    end = start + labels.size
    item_scores = scores[start:end]
    row = {
        "dataset": dataset,
        "recording": stem,
        "token_count": int(labels.size),
        "song_fraction": float(labels.mean()),
        "pearson_r": pearson(item_scores, labels),
        "song_score_mean": mean_or_none(item_scores[labels == 1]),
        "silence_score_mean": mean_or_none(item_scores[labels == 0]),
    }
    summary["by_recording"].append(row)
    start = end

for dataset, _, _ in DATASETS:
    rows = [row for row in summary["by_recording"] if row["dataset"] == dataset]
    dataset_scores = []
    dataset_labels = []
    start = 0
    for item_dataset, _, _, labels in items:
        end = start + labels.size
        if item_dataset == dataset:
            dataset_scores.append(scores[start:end])
            dataset_labels.append(labels)
        start = end
    dataset_scores = np.concatenate(dataset_scores)
    dataset_labels = np.concatenate(dataset_labels)
    summary["by_dataset"].append(
        {
            "dataset": dataset,
            "token_count": int(dataset_labels.size),
            "song_fraction": float(dataset_labels.mean()),
            "pearson_r": pearson(dataset_scores, dataset_labels),
            "song_score_mean": mean_or_none(dataset_scores[dataset_labels == 1]),
            "silence_score_mean": mean_or_none(dataset_scores[dataset_labels == 0]),
        }
    )

summary_path = OUT_DIR / "xcm_canary_bf_zf_4x3_5s_global_sv1_song_state_correlation.json"
npz_path = OUT_DIR / "xcm_canary_bf_zf_4x3_5s_global_sv1_song_state_scores.npz"
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
np.savez(
    npz_path,
    scores=scores.astype(np.float32, copy=False),
    song_state=all_labels.astype(np.float32, copy=False),
    sv1=vt[0].astype(np.float32, copy=False),
    feature_mean=mean.squeeze(0).astype(np.float32, copy=False),
)
print(json.dumps(summary, indent=2))
print(summary_path)
print(npz_path)
