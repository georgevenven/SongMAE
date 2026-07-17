#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import librosa
import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
import umap
from sklearn.cluster import HDBSCAN

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.audio2spec import compute_spectrogram  # noqa: E402
from src.core.extract_embedding import _extract_segment_arrays  # noqa: E402
from src.core.utils import load_audio_params, load_model_state, normalize_spectrogram  # noqa: E402


DATA = Path("/media/george-vengrovski/disk1/data/SNE_bambird_samples")
AUDIO = DATA / "XC574895.mp3"
METADATA = DATA / "XC574895.json"
RUN_DIR = ROOT / "runs/xcl_large_500k_p32x1_c005"
OUT_ROOT = ROOT / "results/sne_bambird_umap/linspa/XC574895/bambird_event_3_full_hdbscan_120_20"
EVENT_ID = 3
SEED = 42
MIN_CLUSTER_SIZE = 120
MIN_SAMPLES = 20


def extract_event(checkpoint):
    metadata = json.loads(METADATA.read_text())
    events = metadata["detected_events_seconds"]
    assert len(events) == len(metadata["event_cluster"])

    params = load_audio_params(RUN_DIR)
    wav, sr = librosa.load(AUDIO, sr=params["sr"], mono=True)
    spec = compute_spectrogram(wav, sr, params["fft"], params["hop_size"], params["mels"])
    state = load_model_state(RUN_DIR, checkpoint.name)
    start, end = events[EVENT_ID]
    first = round(start * sr / params["hop_size"])
    last = round(end * sr / params["hop_size"])
    region = spec[:, first:last]
    assert region.shape[1] > 0
    features = []
    for offset in range(0, region.shape[1], state["model_num_timebins"]):
        chunk = region[:, offset:offset + state["model_num_timebins"]]
        encoded = _extract_segment_arrays(
            normalize_spectrogram(chunk, params["mean"], params["std"]),
            np.full(chunk.shape[1], -1, dtype=np.int64),
            state["model"],
            state["device"],
            state["model_num_timebins"],
            state["patch_width"],
            state["num_patches_height"],
            state["num_patches_time"],
            None,
            "end_of_block",
            False,
        )["encoded_embeddings"]
        features.append(encoded)
    features = np.concatenate(features)
    assert len(features) == region.shape[1]
    features -= features.mean(axis=0, keepdims=True)
    features /= np.maximum(features.std(axis=0, keepdims=True), 1e-6)
    print(f"event {EVENT_ID}: {start:.3f}-{last * params['hop_size'] / sr:.3f}s, {len(features)} points")
    return features, region[:, :len(features)], np.arange(first, first + len(features)), metadata


def position_colors(xy):
    span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 1e-12)
    scaled = (xy - xy.min(axis=0)) / span
    return np.column_stack([scaled[:, 0], scaled[:, 1], np.full(len(xy), 0.5)])


def cluster_colors(labels):
    colors = np.full((len(labels), 4), (0.2, 0.2, 0.2, 0.25))
    cmap = plt.get_cmap("tab20")
    for index, label in enumerate(sorted(set(labels) - {-1})):
        colors[labels == label] = cmap(index % 20)
    return colors


def plot(xy, clusters, spectrogram, checkpoint, out_dir):
    position = position_colors(xy)
    hdbscan_colors = cluster_colors(clusters)
    fig = plt.figure(figsize=(12, 9), dpi=300)
    grid = gridspec.GridSpec(4, 2, height_ratios=[3, 2, 0.2, 0.2], hspace=0.25, wspace=0.12)

    for ax, colors, title in zip(
        [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
        [position, hdbscan_colors],
        ["UMAP position", "HDBSCAN labels"],
    ):
        ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=8, edgecolors="none")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontweight="bold")
        ax.set_ylabel("UMAP 2", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    ax = fig.add_subplot(grid[1, :])
    ax.imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis")
    ax.set_ylabel("Mel bin", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for row, colors, label in [(2, position, "UMAP position"), (3, hdbscan_colors, "HDBSCAN label")]:
        ax = fig.add_subplot(grid[row, :])
        ax.imshow(colors[np.newaxis, :, :], aspect="auto")
        ax.set_xlabel(label, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Lincoln's Sparrow | XC574895 | full Bambird event {EVENT_ID} | {checkpoint.stem}", fontweight="bold")
    fig.savefig(out_dir / "umap_position_hdbscan.png", bbox_inches="tight")
    fig.savefig(out_dir / "umap_position_hdbscan.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    checkpoint = sorted((RUN_DIR / "weights").glob("model_step_*.pth"))[-1]
    out_dir = OUT_ROOT / checkpoint.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    features, spectrogram, source_timebins, metadata = extract_event(checkpoint)
    xy = umap.UMAP(
        n_neighbors=50,
        min_dist=0.0,
        metric="cosine",
        random_state=SEED,
        low_memory=True,
    ).fit_transform(features)
    clusters = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES).fit_predict(xy)
    plot(xy, clusters, spectrogram, checkpoint, out_dir)

    np.savez_compressed(
        out_dir / "umap_points.npz",
        xy=xy,
        hdbscan_labels=clusters,
        source_timebins=source_timebins,
        bambird_event=np.full(len(xy), EVENT_ID),
    )
    start, end = metadata["detected_events_seconds"][EVENT_ID]
    summary = {
        "recording": AUDIO.stem,
        "ebird_code": metadata["ebird_code"],
        "common_name": metadata["common_name"],
        "bambird_events_total": len(metadata["detected_events_seconds"]),
        "bambird_events_used": 1,
        "bambird_event": EVENT_ID,
        "bambird_event_start_seconds": start,
        "bambird_event_end_seconds": end,
        "bambird_event_duration_seconds": end - start,
        "bambird_event_cluster": metadata["event_cluster"][EVENT_ID],
        "analyzed_seconds": len(xy) * 0.005,
        "points": len(xy),
        "run_dir": str(RUN_DIR),
        "checkpoint": str(checkpoint),
        "feature": "final encoder layer, end_of_block, coordinate-wise z-score",
        "umap": {"neighbors": 50, "min_dist": 0.0, "metric": "cosine", "seed": SEED},
        "hdbscan": {"min_cluster_size": MIN_CLUSTER_SIZE, "min_samples": MIN_SAMPLES},
        "clusters": len(set(clusters) - {-1}),
        "noise_fraction": float(np.mean(clusters == -1)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
