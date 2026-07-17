#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix


def evaluate(name, path, component_values, cluster_count, seed, out_dir):
    features = np.load(path / "encoded_embeddings.npy", mmap_mode="r")
    labels0 = np.load(path / "labels_downsampled.npy", mmap_mode="r")
    indices = np.flatnonzero(labels0 >= 0)
    labels = np.asarray(labels0[indices], dtype=np.int64)
    features = np.asarray(features[indices], dtype=np.float32)

    features -= features.mean(axis=0, keepdims=True)
    features /= np.maximum(features.std(axis=0, keepdims=True), 1e-6)
    classes = np.unique(labels)
    rows = []
    for components in component_values:
        reduced = PCA(components, svd_solver="randomized", random_state=seed).fit_transform(features)
        k = cluster_count or classes.size
        clusters = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(reduced)
        table = contingency_matrix(labels, clusters)
        rows.append({
            "model": name,
            "points": int(labels.size),
            "classes": int(classes.size),
            "clusters": int(k),
            "pca_components": components,
            "purity": float(table.max(axis=0).sum() / labels.size),
            "nmi": float(normalized_mutual_info_score(labels, clusters)),
        })
        np.savez_compressed(
            out_dir / f"{name}_pca{components}.npz", indices=indices, labels=labels, clusters=clusters
        )
    return rows


def plot(rows, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
    ticks = sorted({row["pca_components"] for row in rows})
    for model in dict.fromkeys(row["model"] for row in rows):
        selected = [row for row in rows if row["model"] == model]
        x = [row["pca_components"] for row in selected]
        axes[0].plot(x, [row["purity"] for row in selected], marker="o", label=model)
        axes[1].plot(x, [row["nmi"] for row in selected], marker="o", label=model)
    for ax, title in zip(axes, ["Purity", "NMI"]):
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
    parser.add_argument("--pca_components", type=int, nargs="+", default=[32])
    parser.add_argument("--clusters", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sources = [(name, Path(path)) for name, path in (item.split("=", 1) for item in args.embeddings)]
    rows = [
        row
        for name, path in sources
        for row in evaluate(name, path, args.pca_components, args.clusters, args.seed, args.out_dir)
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
