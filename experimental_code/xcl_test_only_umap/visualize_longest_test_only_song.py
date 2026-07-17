#!/usr/bin/env python3

import csv
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
import umap
from sklearn.cluster import HDBSCAN

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.extract_embedding import _extract_segment_arrays  # noqa: E402
from src.core.utils import load_audio_params, load_model_state, load_spec_slice, normalize_spectrogram  # noqa: E402


DATA = Path("/media/george-vengrovski/disk1/data")
SPEC_DIR = DATA / "XCL_val"
ANNOTATIONS = DATA / "XCL" / "XCL_train_annotations.json"
RUN_DIR = ROOT / "runs/xcl_large_500k_p32x1_c005"
OUT_ROOT = ROOT / "results/xcl_test_only_species_umap/ausfig1/XC519434/30_seconds"

SPECIES_ID = 5839
SPECIES_CODE = "ausfig1"
SPECIES_NAME = "Australasian Figbird"
RECORDING = "XC519434"
MAX_POINTS = 25_000
MAX_TIMEBINS = 6_000
SEED = 42


def recording_row():
    annotations = json.loads(ANNOTATIONS.read_text())
    codes = {
        Path(item["recording"]["filename"]).stem: int(item["recording"]["ebird_code"])
        for item in annotations["recordings"]
    }
    with (DATA / "XCL/shards/index.tsv").open() as handle:
        train = list(csv.DictReader(handle, delimiter="\t"))
    assert all(codes[row["name"]] != SPECIES_ID for row in train)
    with (SPEC_DIR / "shards/index.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    row = next(row for row in rows if row["name"] == RECORDING)
    assert codes[RECORDING] == SPECIES_ID
    return row


def extract_song(row, checkpoint):
    state = load_model_state(RUN_DIR, checkpoint.name)
    params = load_audio_params(SPEC_DIR)
    start = int(row["start"])
    length = min(int(row["end"]) - start, MAX_TIMEBINS)
    spec = load_spec_slice(SPEC_DIR / "shards" / row["shard"], start, start + length)
    stride = max(1, int(np.ceil(length / MAX_POINTS)))
    features = []
    positions = []

    chunk_size = state["model_num_timebins"]
    for chunk_start in range(0, length, chunk_size):
        chunk = spec[:, chunk_start:chunk_start + chunk_size]
        chunk = normalize_spectrogram(chunk, params["mean"], params["std"])
        encoded = _extract_segment_arrays(
            chunk,
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
        chunk_positions = np.arange(chunk_start, chunk_start + len(encoded))
        keep = chunk_positions % stride == 0
        features.append(encoded[keep])
        positions.append(chunk_positions[keep])
        print(f"[{min(chunk_start + chunk_size, length):06d}/{length}] timebins")

    features = np.concatenate(features)
    positions = np.concatenate(positions)
    assert len(features) == len(positions) <= MAX_POINTS
    features -= features.mean(axis=0, keepdims=True)
    features /= np.maximum(features.std(axis=0, keepdims=True), 1e-6)
    return features, positions, spec[:, positions], stride


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
        ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=5, edgecolors="none")
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

    fig.suptitle(f"{SPECIES_NAME} | {RECORDING} | first 30 seconds | {checkpoint.stem}", fontweight="bold")
    fig.savefig(out_dir / "umap_position_hdbscan.png", bbox_inches="tight")
    fig.savefig(out_dir / "umap_position_hdbscan.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    checkpoints = sorted((RUN_DIR / "weights").glob("model_step_*.pth"))
    assert checkpoints
    checkpoint = checkpoints[-1]
    out_dir = OUT_ROOT / checkpoint.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    row = recording_row()
    features, positions, spectrogram, stride = extract_song(row, checkpoint)
    xy = umap.UMAP(
        n_neighbors=50,
        min_dist=0.0,
        metric="cosine",
        random_state=SEED,
        low_memory=True,
    ).fit_transform(features)
    min_cluster_size = max(25, round(len(xy) * 0.005))
    clusters = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10).fit_predict(xy)
    plot(xy, clusters, spectrogram, checkpoint, out_dir)

    np.savez_compressed(out_dir / "umap_points.npz", xy=xy, hdbscan_labels=clusters, timebins=positions)
    source_timebins = int(row["end"]) - int(row["start"])
    summary = {
        "species": SPECIES_NAME,
        "ebird_code": SPECIES_CODE,
        "label_id": SPECIES_ID,
        "recording": RECORDING,
        "recording_duration_minutes": source_timebins * 0.005 / 60,
        "analyzed_minutes": min(source_timebins, MAX_TIMEBINS) * 0.005 / 60,
        "source_timebins": source_timebins,
        "sample_stride": stride,
        "points": len(xy),
        "run_dir": str(RUN_DIR),
        "checkpoint": str(checkpoint),
        "feature": "final encoder layer, end_of_block, coordinate-wise z-score",
        "umap": {"neighbors": 50, "min_dist": 0.0, "metric": "cosine", "seed": SEED},
        "hdbscan": {"min_cluster_size": min_cluster_size, "min_samples": 10},
        "clusters": len(set(clusters) - {-1}),
        "noise_fraction": float(np.mean(clusters == -1)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
