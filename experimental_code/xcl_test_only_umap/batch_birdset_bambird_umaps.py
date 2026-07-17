#!/usr/bin/env python3

import csv
import json
import random
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import librosa
import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import HDBSCAN

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.audio2spec import compute_spectrogram  # noqa: E402
from src.core.extract_embedding import _extract_segment_arrays  # noqa: E402
from src.core.utils import load_audio_params, load_model_state, normalize_spectrogram  # noqa: E402


DATA = Path("/media/george-vengrovski/disk1/data/BirdSet_bambird")
TAXONOMY = Path("/media/george-vengrovski/disk1/avex/avex/data/ebird_taxonomy_v2021.json")
RUN_DIR = ROOT / "runs/xcl_large_500k_p32x1_c005"
OUT_ROOT = ROOT / "results/birdset_bambird_umap_100_singers"
MANIFEST = OUT_ROOT / "manifest.csv"
DATASETS = ("PER", "NES", "UHH", "HSN", "NBP", "POW", "SSW", "SNE")
SEED = 42
N = 100
MIN_SECONDS = 20
MAX_SECONDS = 60
MIN_SAMPLES = 20


def select():
    if MANIFEST.exists():
        with MANIFEST.open() as handle:
            return list(csv.DictReader(handle))

    taxonomy = json.loads(TAXONOMY.read_text())
    candidates = []
    for dataset in DATASETS:
        path = DATA / dataset / "metadata" / f"{dataset}_metadata_train.parquet"
        for row in pd.read_parquet(path).itertuples():
            if "song" not in str(row.call_type).lower():
                continue
            events = [
                (float(end - start), event, float(start), float(end), int(cluster))
                for event, ((start, end), cluster) in enumerate(zip(row.detected_events, row.event_cluster))
                if cluster >= 0 and MIN_SECONDS <= end - start <= MAX_SECONDS
            ]
            if not events:
                continue
            duration, event, start, end, cluster = max(events)
            candidates.append(
                {
                    "dataset": dataset,
                    "recording": Path(row.filepath).name,
                    "ebird_code": row.ebird_code,
                    "common_name": taxonomy[row.ebird_code]["common_name"],
                    "call_type": row.call_type,
                    "order": row.order,
                    "event": event,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "bambird_cluster": cluster,
                }
            )

    rng = random.Random(SEED)
    rng.shuffle(candidates)
    unique_recordings = {row["recording"]: row for row in candidates}
    by_species = defaultdict(list)
    for row in unique_recordings.values():
        by_species[row["ebird_code"]].append(row)
    passerines = [code for code, rows in by_species.items() if rows[0]["order"] == "passeriformes"]
    others = [code for code in by_species if code not in passerines]
    rng.shuffle(passerines)
    rng.shuffle(others)
    species = (passerines + others)[:N]
    assert len(species) == N
    selected = [rng.choice(by_species[code]) for code in species]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected[0])
        writer.writeheader()
        writer.writerows(selected)
    return selected


def extract(row, params, state):
    start = float(row["start"])
    end = float(row["end"])
    audio = DATA / row["dataset"] / "audio" / "train" / row["recording"]
    wav, sr = librosa.load(audio, sr=params["sr"], mono=True, offset=start, duration=end - start)
    region = compute_spectrogram(wav, sr, params["fft"], params["hop_size"], params["mels"])
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
    first = round(start * sr / params["hop_size"])
    return features, region, np.arange(first, first + len(features))


def colors(xy, labels):
    span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 1e-12)
    scaled = (xy - xy.min(axis=0)) / span
    position = np.column_stack([scaled[:, 0], scaled[:, 1], np.full(len(xy), 0.5)])
    clusters = np.full((len(labels), 4), (0.2, 0.2, 0.2, 0.25))
    cmap = plt.get_cmap("tab20")
    for index, label in enumerate(sorted(set(labels) - {-1})):
        clusters[labels == label] = cmap(index % 20)
    return position, clusters


def plot(row, xy, labels, spectrogram, checkpoint, out_dir):
    position, clusters = colors(xy, labels)
    fig = plt.figure(figsize=(12, 9), dpi=300)
    grid = gridspec.GridSpec(4, 2, height_ratios=[3, 2, 0.2, 0.2], hspace=0.25, wspace=0.12)
    for ax, color, title in zip(
        [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
        [position, clusters],
        ["UMAP position", "HDBSCAN labels"],
    ):
        ax.scatter(xy[:, 0], xy[:, 1], c=color, s=8, edgecolors="none")
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
    for grid_row, color, label in [(2, position, "UMAP position"), (3, clusters, "HDBSCAN label")]:
        ax = fig.add_subplot(grid[grid_row, :])
        ax.imshow(color[np.newaxis, :, :], aspect="auto")
        ax.set_xlabel(label, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    title = f"{row['common_name']} | {row['dataset']} {Path(row['recording']).stem} | Bambird event {row['event']} | {checkpoint.stem}"
    fig.suptitle(title, fontweight="bold")
    fig.savefig(out_dir / "umap_position_hdbscan.png", bbox_inches="tight")
    plt.close(fig)


def run(row, index, params, state, checkpoint):
    slug = f"{index:03d}_{row['ebird_code']}_{Path(row['recording']).stem}"
    out_dir = OUT_ROOT / slug / checkpoint.stem
    if (out_dir / "summary.json").exists():
        print(f"skip {index:03d}/100 {slug}", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    features, spectrogram, source_timebins = extract(row, params, state)
    xy = umap.UMAP(
        n_neighbors=50,
        min_dist=0.0,
        metric="cosine",
        random_state=SEED,
        low_memory=True,
    ).fit_transform(features)
    min_cluster_size = max(100, round(len(xy) * 0.015))
    labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=MIN_SAMPLES).fit_predict(xy)
    plot(row, xy, labels, spectrogram, checkpoint, out_dir)
    np.savez_compressed(out_dir / "umap_points.npz", xy=xy, hdbscan_labels=labels, source_timebins=source_timebins)
    summary = {
        **row,
        "selection_index": index,
        "points": len(xy),
        "analyzed_seconds": len(xy) * 0.005,
        "run_dir": str(RUN_DIR),
        "checkpoint": str(checkpoint),
        "feature": "final encoder layer, end_of_block, coordinate-wise z-score",
        "umap": {"neighbors": 50, "min_dist": 0.0, "metric": "cosine", "seed": SEED},
        "hdbscan": {"min_cluster_size": min_cluster_size, "min_samples": MIN_SAMPLES},
        "clusters": len(set(labels) - {-1}),
        "noise_fraction": float(np.mean(labels == -1)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"complete {index:03d}/100 {slug} clusters={summary['clusters']}", flush=True)


def main():
    selected = select()
    checkpoint = sorted((RUN_DIR / "weights").glob("model_step_*.pth"))[-1]
    params = load_audio_params(RUN_DIR)
    state = load_model_state(RUN_DIR, checkpoint.name)
    for index, row in enumerate(selected, 1):
        try:
            run(row, index, params, state, checkpoint)
        except Exception:
            slug = f"{index:03d}_{row['ebird_code']}_{Path(row['recording']).stem}"
            out_dir = OUT_ROOT / slug / checkpoint.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "error.txt").write_text(traceback.format_exc())
            print(f"failed {index:03d}/100 {slug}", flush=True)


if __name__ == "__main__":
    main()
