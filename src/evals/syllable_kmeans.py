#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.evals.syllable_classification import ground_truth, load_units


def load_raster(path, units, count):
    stems = np.asarray(np.load(path / "recording_stem.npy")).astype(str)
    starts = np.rint(np.load(path / "token_start_ms.npy")).astype(np.int64)
    ends = np.rint(np.load(path / "token_end_ms.npy")).astype(np.int64)
    assert stems.shape == starts.shape == ends.shape == (count,)

    labels = []
    vocal_ms = np.zeros(stems.size, dtype=np.int64)
    for i, (stem, start, end) in enumerate(zip(stems, starts, ends)):
        y = ground_truth(units, stem, start, end)
        y = y[y > 0]
        vocal_ms[i] = y.size
        if y.size:
            labels.append(y)
    indices = np.flatnonzero(vocal_ms)
    assert indices.size and labels
    return indices, np.concatenate(labels), vocal_ms[indices]


def evaluate(name, path, units, component_values, cluster_count, seed, out_dir):
    features = np.load(path / "encoded_embeddings.npy", mmap_mode="r")
    indices, labels, vocal_ms = load_raster(path, units, features.shape[0])
    features = np.asarray(features[indices], dtype=np.float32)

    features -= features.mean(axis=0, keepdims=True)
    features /= np.maximum(features.std(axis=0, keepdims=True), 1e-6)
    classes = np.unique(labels)
    rows = []
    for components in component_values:
        reduced = PCA(components, svd_solver="randomized", random_state=seed).fit_transform(features)
        k = cluster_count or classes.size
        clusters = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(reduced)
        raster_clusters = np.repeat(clusters, vocal_ms)
        table = contingency_matrix(labels, raster_clusters)
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
            labels, raster_clusters
        )
        rows.append({
            "model": name,
            "points": int(indices.size),
            "frames": int(labels.size),
            "classes": int(classes.size),
            "clusters": int(k),
            "pca_components": components,
            "purity": float(table.max(axis=0).sum() / labels.size),
            "homogeneity": float(homogeneity),
            "completeness": float(completeness),
            "v_measure": float(v_measure),
            "nmi": float(v_measure),
        })
        np.savez_compressed(
            out_dir / f"{name}_pca{components}.npz",
            token_indices=indices,
            token_clusters=clusters,
            raster_labels=labels,
            raster_clusters=raster_clusters,
        )
    return rows


def plot(rows, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=200)
    ticks = sorted({row["pca_components"] for row in rows})
    for model in dict.fromkeys(row["model"] for row in rows):
        selected = [row for row in rows if row["model"] == model]
        x = [row["pca_components"] for row in selected]
        axes[0].plot(x, [row["purity"] for row in selected], marker="o", label=model)
        axes[1].plot(x, [row["completeness"] for row in selected], marker="o", label=model)
        axes[2].plot(x, [row["v_measure"] for row in selected], marker="o", label=model)
    for ax, title in zip(axes, ["Purity", "Completeness", "V-measure / NMI"]):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, labels=[str(x) for x in ticks])
        ax.set_xlabel("PCA dimensions")
        ax.set_ylabel(title)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "pca_sweep.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="PCA and K-means clustering of labeled syllable embeddings.")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("embeddings", nargs="+", help="NAME=EMBEDDING_DIR")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pca_components", type=int, nargs="+", default=[8])
    parser.add_argument("--clusters", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    units = load_units(args.annotations)
    sources = [(name, Path(path)) for name, path in (item.split("=", 1) for item in args.embeddings)]
    rows = [
        row
        for name, path in sources
        for row in evaluate(
            name, path, units, args.pca_components, args.clusters, args.seed, args.out_dir
        )
    ]
    with (args.out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    plot(rows, args.out_dir)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
