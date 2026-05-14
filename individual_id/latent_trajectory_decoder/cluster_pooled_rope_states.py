import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score, normalized_mutual_info_score, v_measure_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from individual_id.latent_trajectory_decoder.train_rope_sequence import RopeSequenceDecoder


def load_rows(path):
    data = np.load(path, allow_pickle=True)
    features = data["features"].astype(np.float32)
    birds = data["bird_labels"].astype(str)
    recordings = data["recording_labels"].astype(str)
    by_recording = defaultdict(list)
    for i, recording in enumerate(recordings):
        by_recording[recording].append(i)

    rows = []
    for recording, indices in by_recording.items():
        labels = np.unique(birds[indices])
        assert len(labels) == 1, recording
        if len(indices) > 2:
            rows.append((recording, labels[0], features[indices]))
    return rows, features.shape[1]


def select_rows(rows, individuals, songs_per_individual, seed):
    rng = np.random.default_rng(seed)
    by_bird = defaultdict(list)
    for row in rows:
        by_bird[row[1]].append(row)
    eligible = [bird for bird, bird_rows in by_bird.items() if len(bird_rows) >= songs_per_individual]
    assert len(eligible) >= individuals
    birds = sorted(rng.choice(sorted(eligible), size=individuals, replace=False).tolist())
    selected = []
    for bird in birds:
        bird_rows = sorted(by_bird[bird], key=lambda row: row[0])
        keep = rng.choice(len(bird_rows), size=songs_per_individual, replace=False)
        selected.extend([bird_rows[i] for i in sorted(keep)])
    return selected, birds


def load_model(path, input_dim, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    model = RopeSequenceDecoder(input_dim, int(args["d_model"]), int(args["heads"]), int(args["layers"])).to(device)
    model.load_state_dict(checkpoint["decoder"])
    model.eval()
    return model, args


@torch.no_grad()
def states_for_sequence(model, seq, device):
    x = torch.from_numpy(seq[:-1]).to(device=device, dtype=torch.float32)[None]
    valid = torch.ones((1, x.shape[1]), device=device, dtype=torch.bool)
    _, h = model(x, valid)
    return h.squeeze(0).cpu().numpy().astype(np.float32)


def pool_state(h, mode):
    if mode == "mean":
        return h.mean(axis=0)
    assert mode == "mean_std"
    return np.concatenate([h.mean(axis=0), h.std(axis=0)], axis=0)


def transformed(x, pca_dim, seed):
    scaled = StandardScaler().fit_transform(x)
    n_components = min(pca_dim, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=n_components, whiten=True, random_state=seed)
    z = pca.fit_transform(scaled).astype(np.float32)
    return normalize(z).astype(np.float32), pca.explained_variance_ratio_.astype(float).tolist()


def fit_umap(x, neighbors, min_dist, metric, seed):
    import umap

    reducer = umap.UMAP(
        n_neighbors=min(neighbors, max(2, x.shape[0] - 1)),
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(x).astype(np.float32)


def cluster_hdbscan(x, min_cluster_size, min_samples):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(x).astype(np.int64)


def scores(true_labels, clusters):
    encoded = LabelEncoder().fit_transform(true_labels)
    clustered = clusters >= 0
    cluster_count = len(set(clusters.tolist()) - {-1})
    result = {
        "clusters": int(cluster_count),
        "noise_fraction": float(np.mean(~clustered)),
        "ari_all": float(adjusted_rand_score(encoded, clusters)),
        "nmi_all": float(normalized_mutual_info_score(encoded, clusters)),
    }
    if clustered.any() and cluster_count > 1:
        result.update(
            {
                "homogeneity_clustered": float(homogeneity_score(encoded[clustered], clusters[clustered])),
                "completeness_clustered": float(completeness_score(encoded[clustered], clusters[clustered])),
                "v_measure_clustered": float(v_measure_score(encoded[clustered], clusters[clustered])),
            }
        )
    else:
        result.update({"homogeneity_clustered": 0.0, "completeness_clustered": 0.0, "v_measure_clustered": 0.0})
    return result


def colors(labels):
    unique = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("turbo", len(unique))
    return {label: cmap(i) for i, label in enumerate(unique)}


def plot_labels(xy, labels, title, path):
    color_map = colors(labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    for label in sorted(color_map):
        mask = labels == label
        ax.scatter(xy[mask, 0], xy[mask, 1], s=28, color=color_map[label], alpha=0.85, linewidths=0, label=str(label))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_clusters(xy, clusters, title, path):
    labels = np.asarray([f"c{c}" if c >= 0 else "noise" for c in clusters], dtype=object)
    plot_labels(xy, labels, title, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster full pooled RoPE decoder states and visualize with UMAP.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--model_pt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--individuals", type=int, default=30)
    parser.add_argument("--songs_per_individual", type=int, default=10)
    parser.add_argument("--pca_dim", type=int, default=32)
    parser.add_argument("--umap_neighbors", type=int, default=20)
    parser.add_argument("--umap_min_dist", type=float, default=0.05)
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=5)
    parser.add_argument("--hdbscan_min_samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, input_dim = load_rows(args.features_npz)
    selected, selected_birds = select_rows(rows, args.individuals, args.songs_per_individual, args.seed)
    model, model_args = load_model(args.model_pt, input_dim, device)

    pooled = {"mean": [], "mean_std": []}
    birds = []
    recordings = []
    for recording, bird, seq in selected:
        h = states_for_sequence(model, seq, device)
        pooled["mean"].append(pool_state(h, "mean"))
        pooled["mean_std"].append(pool_state(h, "mean_std"))
        birds.append(bird)
        recordings.append(recording)

    birds = np.asarray(birds, dtype=object)
    recordings = np.asarray(recordings, dtype=object)
    summary = {
        "features_npz": args.features_npz,
        "model_pt": args.model_pt,
        "device": str(device),
        "individuals": args.individuals,
        "songs_per_individual": args.songs_per_individual,
        "selected_birds": selected_birds,
        "songs": len(selected),
        "input_dim": int(input_dim),
        "decoder_state_dim": int(model_args["d_model"]),
        "modes": {},
        "leiden": "skipped: igraph/leidenalg not installed",
    }
    rows_for_csv = []
    arrays = {"birds": birds, "recordings": recordings}
    for mode, vectors in pooled.items():
        x = np.vstack(vectors).astype(np.float32)
        z, evr = transformed(x, args.pca_dim, args.seed)
        xy = fit_umap(z, args.umap_neighbors, args.umap_min_dist, "cosine", args.seed)
        clusters_pca = cluster_hdbscan(z, args.hdbscan_min_cluster_size, args.hdbscan_min_samples)
        clusters_umap = cluster_hdbscan(xy, args.hdbscan_min_cluster_size, args.hdbscan_min_samples)

        true_png = out_dir / f"{mode}_umap_true_birds.png"
        pca_cluster_png = out_dir / f"{mode}_umap_hdbscan_pca32_clusters.png"
        umap_cluster_png = out_dir / f"{mode}_umap_hdbscan_umap2_clusters.png"
        plot_labels(xy, birds, f"{mode}: UMAP of pooled decoder states, colored by bird", true_png)
        plot_clusters(xy, clusters_pca, f"{mode}: HDBSCAN on PCA32, shown on UMAP", pca_cluster_png)
        plot_clusters(xy, clusters_umap, f"{mode}: HDBSCAN on UMAP2", umap_cluster_png)

        summary["modes"][mode] = {
            "pooled_dim": int(x.shape[1]),
            "pca_dim": int(z.shape[1]),
            "pca_explained_variance_ratio_first8": evr[:8],
            "umap_true_birds_png": str(true_png),
            "hdbscan_pca32_png": str(pca_cluster_png),
            "hdbscan_umap2_png": str(umap_cluster_png),
            "hdbscan_pca32": scores(birds, clusters_pca),
            "hdbscan_umap2": scores(birds, clusters_umap),
        }
        for source, clusters in [("pca32", clusters_pca), ("umap2", clusters_umap)]:
            row = {"mode": mode, "cluster_source": source}
            row.update(scores(birds, clusters))
            rows_for_csv.append(row)
        arrays[f"{mode}_pooled"] = x
        arrays[f"{mode}_pca32"] = z
        arrays[f"{mode}_umap"] = xy
        arrays[f"{mode}_hdbscan_pca32"] = clusters_pca
        arrays[f"{mode}_hdbscan_umap2"] = clusters_umap

    np.savez_compressed(out_dir / "cluster_pooled_rope_states.npz", **arrays)
    with (out_dir / "cluster_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_for_csv[0]))
        writer.writeheader()
        writer.writerows(rows_for_csv)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
