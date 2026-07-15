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


def token_song_state(events, audio_params, patch_width, token_count):
    frames = np.zeros(token_count * patch_width, dtype=np.float32)
    for event in events:
        start = max(0, min(ms_to_timebins(event["onset_ms"], audio_params), frames.size))
        end = max(start, min(ms_to_timebins(event["offset_ms"], audio_params), frames.size))
        frames[start:end] = 1.0
    return frames.reshape(token_count, patch_width).max(axis=1)


def choose_examples(spec_dir, events_by_stem, audio_params):
    stems = []
    for path in sorted(spec_dir.glob("*.npy")):
        if path.stem not in events_by_stem:
            continue
        spec = np.load(path, mmap_mode="r")
        if spec.shape[1] < TIMEBINS:
            continue
        state = token_song_state(events_by_stem[path.stem], audio_params, 10, TIMEBINS // 10)
        if state.min() == state.max():
            continue
        stems.append(path.stem)
        if len(stems) == EXAMPLES_PER_DATASET:
            return stems
    assert len(stems) == EXAMPLES_PER_DATASET


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
        features = extracted["segments"][0]["encoded_embeddings_before_pos_removal"].astype(np.float32, copy=False)
        labels = token_song_state(
            events_by_stem[stem],
            extracted["audio_params"],
            int(extracted["patch_width"]),
            features.shape[0],
        )
        items.append((dataset, stem, features, labels))

all_features = np.concatenate([features for _, _, features, _ in items], axis=0).astype(np.float32, copy=False)
all_labels = np.concatenate([labels for _, _, _, labels in items], axis=0).astype(np.float32, copy=False)
sv1, singular_value, mean = top_right_singular_vector(all_features)
scores = (all_features - mean) @ sv1
if scores[all_labels == 1].mean() < scores[all_labels == 0].mean():
    sv1 *= -1.0
    scores *= -1.0

summary = {
    "model": str(RUN_DIR),
    "timebins": TIMEBINS,
    "examples_per_dataset": EXAMPLES_PER_DATASET,
    "patch_width": int(model_state["patch_width"]),
    "sv1_singular_value": singular_value,
    "global": {
        "token_count": int(all_labels.size),
        "song_fraction": float(all_labels.mean()),
        "pearson_r": pearson(scores, all_labels),
        "song_score_mean": float(scores[all_labels == 1].mean()),
        "silence_score_mean": float(scores[all_labels == 0].mean()),
    },
    "by_dataset": [],
    "by_recording": [],
}

start = 0
for dataset, stem, _, labels in items:
    end = start + labels.size
    item_scores = scores[start:end]
    summary["by_recording"].append(
        {
            "dataset": dataset,
            "recording": stem,
            "token_count": int(labels.size),
            "song_fraction": float(labels.mean()),
            "pearson_r": pearson(item_scores, labels),
            "song_score_mean": float(item_scores[labels == 1].mean()),
            "silence_score_mean": float(item_scores[labels == 0].mean()),
        }
    )
    start = end

for dataset, _, _ in DATASETS:
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
            "song_score_mean": float(dataset_scores[dataset_labels == 1].mean()),
            "silence_score_mean": float(dataset_scores[dataset_labels == 0].mean()),
        }
    )

summary_path = OUT_DIR / "xcm_canary_bf_zf_30each_5s_global_sv1_song_state_correlation.json"
npz_path = OUT_DIR / "xcm_canary_bf_zf_30each_5s_global_sv1_song_state_scores.npz"
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
np.savez(
    npz_path,
    scores=scores.astype(np.float32, copy=False),
    song_state=all_labels.astype(np.float32, copy=False),
    sv1=sv1.astype(np.float32, copy=False),
    feature_mean=mean.astype(np.float32, copy=False),
)
print(json.dumps(summary, indent=2))
print(summary_path)
print(npz_path)
