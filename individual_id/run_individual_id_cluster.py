#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from numba import njit
from sklearn.cluster import HDBSCAN

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import extract_embedding  # noqa: E402
from run_individual_id_umap import (  # noqa: E402
    _load_recording_stems_by_bird,
    _pick_recordings,
    _pool_embeddings,
    _pool_labels,
    _resolve_run_dir,
)


DEFAULT_RUN_DIR = (
    "/home/george-vengrovski/Documents/projects/TinyBird/"
    "runs/xcm_voronoi_mask_no_normalize_32h_10w_zf_continue10k_bs24_20260302_133416"
)
DEFAULT_SPEC_DIR = "/media/george-vengrovski/disk2/specs/zf_64hop_32khz"
DEFAULT_ANNOTATION_JSON = "/home/george-vengrovski/Documents/projects/TinyBird/files/zf_annotations.json"
DEFAULT_OUT_DIR = str(ROOT / "results" / "individual_id_cluster")


def _load_recording_features(args, bird_id, recording_stem, model_state):
    try:
        extracted = extract_embedding.extract_recording_embeddings_with_state(
            {
                "run_dir": str(args.run_dir),
                "checkpoint": args.checkpoint,
                "spec_dir": str(args.spec_dir),
                "json_path": str(args.annotation_json),
                "bird": bird_id,
                "recording_stem": recording_stem,
                "recording_mode": args.recording_mode,
                "encoder_layer_idx": args.encoder_layer_idx,
                "spec_normalization": args.songmae_input_normalization,
                "normalization_stats_dir": args.songmae_input_normalization_stats_dir,
            },
            model_state,
        )
    except ValueError as exc:
        if str(exc) == "No valid patches extracted for the requested recording set.":
            return None
        raise

    pooled_parts = []
    for segment in extracted["segments"]:
        embedding_key = f"encoded_embeddings_{args.embedding_variant}_pos_removal"
        features = segment[embedding_key]
        labels = segment["labels_downsampled"]
        count = min(features.shape[0], labels.shape[0])
        if count == 0:
            continue

        pooled = _pool_embeddings(
            features[:count],
            args.pool_window,
            args.pool_mode,
            args.pool_hop,
        )
        pooled_labels = _pool_labels(labels[:count], args.pool_window, args.pool_hop)
        count = min(pooled.shape[0], pooled_labels.shape[0])
        if count == 0:
            continue

        pooled = pooled[:count]
        pooled_labels = pooled_labels[:count]
        if args.drop_silence:
            keep = pooled_labels >= 0
            pooled = pooled[keep]
            pooled_labels = pooled_labels[keep]
        if pooled.shape[0] == 0:
            continue
        pooled_parts.append(pooled)

    if not pooled_parts:
        return None

    return np.vstack(pooled_parts).astype(np.float32, copy=False)


def _build_recordings(args, model_state):
    stems_by_bird = _load_recording_stems_by_bird(args.annotation_json)
    bird_ids = sorted(stems_by_bird)
    if args.max_birds > 0:
        bird_ids = bird_ids[: args.max_birds]

    recordings = []
    for bird_id in bird_ids:
        stems = _pick_recordings(
            stems_by_bird[bird_id],
            songs_per_bird=args.songs_per_bird,
            seed=args.seed,
            bird_id=bird_id,
        )
        for recording_stem in stems:
            features = _load_recording_features(args, bird_id, recording_stem, model_state)
            if features is None or features.shape[0] == 0:
                continue
            recordings.append(
                {
                    "bird_id": bird_id,
                    "recording_stem": recording_stem,
                    "features": features,
                }
            )

    assert recordings, "No valid recordings were loaded."
    return recordings


def _stack_recordings(recordings):
    x_parts = []
    bird_labels = []
    recording_indices = []
    point_counts = []
    for recording_index, recording in enumerate(recordings):
        features = recording["features"]
        x_parts.append(features)
        point_counts.append(int(features.shape[0]))
        bird_labels.extend([recording["bird_id"]] * features.shape[0])
        recording_indices.extend([recording_index] * features.shape[0])
    return (
        np.vstack(x_parts).astype(np.float32, copy=False),
        np.asarray(bird_labels, dtype=object),
        np.asarray(recording_indices, dtype=np.int64),
        np.asarray(point_counts, dtype=np.int64),
    )


def _fit_umap(features, neighbors, min_dist, metric, seed):
    used_neighbors = min(int(neighbors), max(1, int(features.shape[0]) - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=used_neighbors,
        min_dist=float(min_dist),
        metric=metric,
        low_memory=True,
        n_jobs=-1,
        random_state=int(seed),
    )
    return reducer.fit_transform(features).astype(np.float32, copy=False)


def _reindex_cluster_ids(labels):
    labels = np.asarray(labels, dtype=np.int64)
    remapped = np.full(labels.shape, -1, dtype=np.int64)

    next_cluster_id = 0
    for raw_label in np.unique(labels):
        if int(raw_label) < 0:
            continue
        remapped[labels == raw_label] = next_cluster_id
        next_cluster_id += 1

    noise_points = labels < 0
    if noise_points.any():
        remapped[noise_points] = next_cluster_id
        next_cluster_id += 1

    assert next_cluster_id > 0
    return remapped, next_cluster_id, int(noise_points.sum())


def _fit_hdbscan(features, min_cluster_size, min_samples):
    point_count = int(features.shape[0])
    used_min_cluster_size = max(2, min(int(min_cluster_size), point_count))
    used_min_samples = None if min_samples is None else min(max(1, int(min_samples)), point_count)
    model = HDBSCAN(
        min_cluster_size=used_min_cluster_size,
        min_samples=used_min_samples,
        n_jobs=-1,
    )
    raw_labels = model.fit_predict(features)
    labels, cluster_count, noise_points = _reindex_cluster_ids(raw_labels)
    return labels.astype(np.int64, copy=False), cluster_count, noise_points


def _cluster_colors(cluster_ids, n_clusters):
    cmap = plt.get_cmap("gist_ncar", max(1, n_clusters))
    return cmap(cluster_ids / max(1, n_clusters - 1))


def _save_cluster_umap(xy, cluster_ids, n_clusters, title, out_base):
    fig = plt.figure(figsize=(9, 7), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=8,
        alpha=0.22,
        c=_cluster_colors(cluster_ids, n_clusters),
        edgecolors="none",
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _build_barcodes(recording_indices, cluster_ids, n_recordings, n_clusters, min_hits):
    barcodes = np.zeros((n_recordings, n_clusters), dtype=bool)
    unique_cluster_counts = np.zeros((n_recordings,), dtype=np.int64)
    barcode_cluster_counts = np.zeros((n_recordings,), dtype=np.int64)
    for recording_index in range(n_recordings):
        ids = cluster_ids[recording_indices == recording_index]
        assert ids.size > 0
        counts = np.bincount(ids, minlength=n_clusters)
        unique_cluster_counts[recording_index] = int((counts > 0).sum())
        barcodes[recording_index] = counts >= int(min_hits)
        if not barcodes[recording_index].any():
            barcodes[recording_index, int(np.argmax(counts))] = True
        barcode_cluster_counts[recording_index] = int(barcodes[recording_index].sum())
    return barcodes, unique_cluster_counts, barcode_cluster_counts


def _summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1
    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p75": float(np.percentile(values, 75)),
        "max": float(values.max()),
    }


def _jaccard_similarity(barcodes):
    overlap = barcodes.astype(np.float32) @ barcodes.astype(np.float32).T
    size = barcodes.sum(axis=1, dtype=np.float32)
    union = size[:, None] + size[None, :] - overlap
    similarity = np.divide(overlap, union, out=np.ones_like(overlap), where=union > 0)
    return similarity.astype(np.float32, copy=False)


def _make_histogram_edges(values, bins):
    values = np.asarray(values, dtype=np.float32)
    assert values.ndim == 1
    assert values.size > 0
    lower = float(values.min())
    upper = float(values.max())
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    return np.linspace(lower, upper, int(bins) + 1, dtype=np.float32)


def _build_recording_umap_histograms(xy, recording_indices, n_recordings, bins):
    assert xy.ndim == 2 and xy.shape[1] == 2
    x_edges = _make_histogram_edges(xy[:, 0], bins)
    y_edges = _make_histogram_edges(xy[:, 1], bins)

    histograms = np.zeros((n_recordings, int(bins), int(bins)), dtype=np.float32)
    point_counts = np.zeros((n_recordings,), dtype=np.int64)
    for recording_index in range(n_recordings):
        points = xy[recording_indices == recording_index]
        assert points.shape[0] > 0
        counts, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[x_edges, y_edges])
        histograms[recording_index] = counts.astype(np.float32, copy=False)
        point_counts[recording_index] = int(points.shape[0])
    return histograms, point_counts


def _bhattacharyya_similarity_matrix(histograms):
    flattened = histograms.reshape(histograms.shape[0], -1).astype(np.float32, copy=False)
    totals = flattened.sum(axis=1, keepdims=True)
    normalized = np.divide(flattened, totals, out=np.zeros_like(flattened), where=totals > 0)

    similarity = np.ones((histograms.shape[0], histograms.shape[0]), dtype=np.float32)
    for i in range(histograms.shape[0]):
        for j in range(i + 1, histograms.shape[0]):
            sim = float(np.sqrt(normalized[i] * normalized[j]).sum())
            similarity[i, j] = sim
            similarity[j, i] = sim
    return similarity


def _build_cluster_sequences(recording_indices, cluster_ids, n_recordings):
    lengths = np.zeros((n_recordings,), dtype=np.int64)
    max_len = 0
    for recording_index in range(n_recordings):
        seq_len = int(np.sum(recording_indices == recording_index))
        lengths[recording_index] = seq_len
        max_len = max(max_len, seq_len)

    sequences = np.full((n_recordings, max_len), -1, dtype=np.int64)
    for recording_index in range(n_recordings):
        seq = cluster_ids[recording_indices == recording_index]
        sequences[recording_index, : seq.shape[0]] = seq
    return sequences, lengths


@njit(cache=True)
def _levenshtein_distance(seq_a, len_a, seq_b, len_b):
    prev = np.arange(len_b + 1, dtype=np.int64)
    curr = np.empty(len_b + 1, dtype=np.int64)
    for i in range(1, len_a + 1):
        curr[0] = i
        token_a = seq_a[i - 1]
        for j in range(1, len_b + 1):
            cost = 0 if token_a == seq_b[j - 1] else 1
            deletion = prev[j] + 1
            insertion = curr[j - 1] + 1
            substitution = prev[j - 1] + cost
            curr[j] = min(deletion, insertion, substitution)
        prev, curr = curr, prev
    return prev[len_b]


@njit(cache=True)
def _levenshtein_similarity_matrix(sequences, lengths):
    n_recordings = lengths.shape[0]
    similarity = np.ones((n_recordings, n_recordings), dtype=np.float32)
    for i in range(n_recordings):
        for j in range(i + 1, n_recordings):
            len_i = int(lengths[i])
            len_j = int(lengths[j])
            norm = max(len_i, len_j)
            if norm == 0:
                sim = np.float32(1.0)
            else:
                dist = _levenshtein_distance(
                    sequences[i],
                    len_i,
                    sequences[j],
                    len_j,
                )
                sim = np.float32(1.0 - (dist / norm))
            similarity[i, j] = sim
            similarity[j, i] = sim
    return similarity


def _connected_components(similarity, threshold):
    adjacency = similarity >= float(threshold)
    group_ids = np.full(adjacency.shape[0], -1, dtype=np.int64)
    next_group = 0
    for start in range(adjacency.shape[0]):
        if group_ids[start] != -1:
            continue
        stack = [start]
        group_ids[start] = next_group
        while stack:
            node = stack.pop()
            neighbors = np.flatnonzero(adjacency[node])
            for neighbor in neighbors:
                if group_ids[neighbor] != -1:
                    continue
                group_ids[neighbor] = next_group
                stack.append(int(neighbor))
        next_group += 1
    return group_ids


def _majority_vote_accuracy(true_labels, pred_groups):
    pred_labels = np.empty(true_labels.shape[0], dtype=object)
    for group_id in sorted(set(pred_groups.tolist())):
        idx = pred_groups == group_id
        values, counts = np.unique(true_labels[idx], return_counts=True)
        pred_labels[idx] = values[np.argmax(counts)]
    return float(np.mean(pred_labels == true_labels))


def _pairwise_threshold_metrics(similarity, true_birds, threshold):
    assert similarity.shape[0] == true_birds.shape[0]
    if similarity.shape[0] < 2:
        empty = _summarize_values(np.zeros((0,), dtype=np.float32))
        return {
            "same_bird_overlap": empty,
            "different_bird_overlap": empty,
            "threshold_metrics": {
                "threshold": float(threshold),
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "precision": None,
                "recall": None,
                "f1": None,
            },
        }

    upper = np.triu_indices(similarity.shape[0], k=1)
    pair_similarity = similarity[upper]
    same_bird = true_birds[upper[0]] == true_birds[upper[1]]
    pred_same = pair_similarity >= float(threshold)

    tp = int(np.sum(pred_same & same_bird))
    fp = int(np.sum(pred_same & ~same_bird))
    tn = int(np.sum(~pred_same & ~same_bird))
    fn = int(np.sum(~pred_same & same_bird))

    precision = None if tp + fp == 0 else float(tp / (tp + fp))
    recall = None if tp + fn == 0 else float(tp / (tp + fn))
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = float(2 * precision * recall / (precision + recall))

    return {
        "same_bird_overlap": _summarize_values(pair_similarity[same_bird]),
        "different_bird_overlap": _summarize_values(pair_similarity[~same_bird]),
        "threshold_metrics": {
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }


def _save_similarity_heatmap(similarity, recordings, title, out_base):
    fig_size = max(8.0, min(18.0, 0.22 * similarity.shape[0]))
    fig = plt.figure(figsize=(fig_size, fig_size), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    image = ax.imshow(similarity, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    label_names = [f'{rec["bird_id"]}:{rec["recording_stem"]}' for rec in recordings]
    if len(label_names) <= 60:
        ax.set_xticks(np.arange(len(label_names)))
        ax.set_yticks(np.arange(len(label_names)))
        ax.set_xticklabels(label_names, rotation=90, fontsize=5)
        ax.set_yticklabels(label_names, fontsize=5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _analyze_clustering(name, xy, recording_indices, recordings, n_clusters, noise_points, args):
    true_birds = np.asarray([recording["bird_id"] for recording in recordings], dtype=object)
    histogram_bins = int(args.overlap_bins)
    # Previous barcode/sequence-based similarity path is intentionally disabled:
    # barcodes, unique_cluster_counts, barcode_cluster_counts = _build_barcodes(...)
    # sequences, sequence_lengths = _build_cluster_sequences(...)
    # similarity = _levenshtein_similarity_matrix(sequences, sequence_lengths)
    histograms, point_counts = _build_recording_umap_histograms(
        xy=xy,
        recording_indices=recording_indices,
        n_recordings=len(recordings),
        bins=histogram_bins,
    )
    similarity = _bhattacharyya_similarity_matrix(histograms)
    group_ids = _connected_components(similarity, args.overlap_threshold)
    return {
        "name": name,
        "histograms": histograms,
        "point_counts": point_counts,
        "similarity": similarity,
        "group_ids": group_ids,
        "summary": {
            "similarity_mode": "umap_bhattacharyya",
            "clusterer": "hdbscan",
            "min_cluster_size": int(args.min_cluster_size),
            "min_samples": None if args.min_samples is None else int(args.min_samples),
            "cluster_label_count": int(n_clusters),
            "noise_points": int(noise_points),
            "overlap_bins": histogram_bins,
            "predicted_individuals": int(len(np.unique(group_ids))),
            "true_individuals": int(len(np.unique(true_birds))),
            "recording_majority_accuracy": _majority_vote_accuracy(true_birds, group_ids),
            "recording_point_count": _summarize_values(point_counts),
            "mean_off_diagonal_overlap": float(
                similarity[~np.eye(similarity.shape[0], dtype=bool)].mean()
            )
            if similarity.shape[0] > 1
            else 1.0,
            **_pairwise_threshold_metrics(similarity, true_birds, args.overlap_threshold),
        },
    }


def _write_summary(out_path, args, recordings, total_points, analyses):
    summary = {
        "species": args.species,
        "model": {
            "run_dir": str(args.run_dir),
            "checkpoint": args.checkpoint,
        },
        "dataset": {
            "recordings": int(len(recordings)),
            "points": int(total_points),
            "birds": int(len({recording["bird_id"] for recording in recordings})),
        },
        "args": {
            "annotation_json": str(args.annotation_json),
            "spec_dir": str(args.spec_dir),
            "recording_mode": args.recording_mode,
            "songs_per_bird": int(args.songs_per_bird),
            "max_birds": int(args.max_birds),
            "seed": int(args.seed),
            "pool_window": int(args.pool_window),
            "pool_hop": int(args.pool_hop),
            "pool_mode": args.pool_mode,
            "embedding_variant": args.embedding_variant,
            "drop_silence": bool(args.drop_silence),
            "encoder_layer_idx": args.encoder_layer_idx,
            "normalization_preset": args.normalization_preset,
            "audio_params_stats_dir": args.audio_params_stats_dir,
            "songmae_input_normalization": args.songmae_input_normalization,
            "songmae_input_normalization_stats_dir": args.songmae_input_normalization_stats_dir,
            "min_cluster_size": int(args.min_cluster_size),
            "min_samples": None if args.min_samples is None else int(args.min_samples),
            "min_cluster_hits": int(args.min_cluster_hits),
            "overlap_threshold": float(args.overlap_threshold),
            "overlap_bins": int(args.overlap_bins),
            "umap_neighbors": int(args.umap_neighbors),
            "umap_min_dist": float(args.umap_min_dist),
            "umap_metric": args.umap_metric,
        },
        "results": {analysis["name"]: analysis["summary"] for analysis in analyses},
        "recordings": [
            {
                "bird_id": recording["bird_id"],
                "recording_stem": recording["recording_stem"],
                "points": int(recording["features"].shape[0]),
                "embedding_group": int(analyses[0]["group_ids"][index]),
                "embedding_overlap_points": int(analyses[0]["point_counts"][index]),
                "umap_group": int(analyses[1]["group_ids"][index]),
                "umap_overlap_points": int(analyses[1]["point_counts"][index]),
            }
            for index, recording in enumerate(recordings)
        ],
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Cluster pooled recording embeddings with HDBSCAN and compare recording barcodes.")
    parser.add_argument("--species", default="Zebra_Finch")
    parser.add_argument("--annotation_json", default=DEFAULT_ANNOTATION_JSON)
    parser.add_argument("--spec_dir", default=DEFAULT_SPEC_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--run_dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--recording_mode", default="events", choices=["events", "full_recordings"])
    parser.add_argument("--songs_per_bird", type=int, default=30)
    parser.add_argument("--max_birds", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool_window", type=int, default=30)
    parser.add_argument("--pool_hop", type=int, default=5)
    parser.add_argument("--pool_mode", default="mean", choices=["mean", "max", "sum"])
    parser.add_argument("--embedding_variant", default="before", choices=["before", "after"])
    parser.add_argument("--drop_silence", action="store_true")
    parser.add_argument("--encoder_layer_idx", type=int, default=None)
    parser.add_argument("--normalization_preset", choices=["vanilla", "zscore", "zscore_rescaled"], default=None)
    parser.add_argument("--audio_params_stats_dir", default=None)
    parser.add_argument(
        "--songmae_input_normalization",
        choices=["none", "per_model_input_zscore", "per_recording_cmvn", "per_recording_cmvn_rescaled_to_target_stats"],
        default="none",
    )
    parser.add_argument("--songmae_input_normalization_stats_dir", default=None)
    parser.add_argument("--min_cluster_size", "--n_clusters", dest="min_cluster_size", type=int, default=100)
    parser.add_argument("--min_samples", type=int, default=None)
    parser.add_argument("--min_cluster_hits", type=int, default=1)
    parser.add_argument("--overlap_threshold", type=float, default=0.3)
    parser.add_argument("--overlap_bins", type=int, default=300)
    parser.add_argument("--umap_neighbors", type=int, default=100)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_metric", default="cosine")
    args = parser.parse_args()

    args.annotation_json = str(Path(args.annotation_json).resolve())
    args.spec_dir = str(Path(args.spec_dir).resolve())
    args.out_dir = str(Path(args.out_dir).resolve())
    args.run_dir = str(_resolve_run_dir(args.run_dir))
    if args.audio_params_stats_dir is not None:
        args.audio_params_stats_dir = str(Path(args.audio_params_stats_dir).resolve())
    if args.songmae_input_normalization_stats_dir is not None:
        args.songmae_input_normalization_stats_dir = str(Path(args.songmae_input_normalization_stats_dir).resolve())

    if args.normalization_preset is not None:
        if args.normalization_preset == "vanilla":
            args.songmae_input_normalization = "none"
        elif args.normalization_preset == "zscore":
            args.songmae_input_normalization = "per_model_input_zscore"
        else:
            assert args.normalization_preset == "zscore_rescaled"
            args.songmae_input_normalization = "per_recording_cmvn_rescaled_to_target_stats"
            args.songmae_input_normalization_stats_dir = args.audio_params_stats_dir

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_state = extract_embedding.load_model_state({"run_dir": str(args.run_dir), "checkpoint": args.checkpoint})
    recordings = _build_recordings(args, model_state)
    features, _, recording_indices, point_counts = _stack_recordings(recordings)
    assert features.shape[0] >= 2, "Need at least two pooled points."

    xy = _fit_umap(
        features,
        neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        seed=args.seed,
    )

    embedding_cluster_ids, embedding_cluster_count, embedding_noise_points = _fit_hdbscan(
        features,
        args.min_cluster_size,
        args.min_samples,
    )
    umap_cluster_ids, umap_cluster_count, umap_noise_points = _fit_hdbscan(
        xy,
        args.min_cluster_size,
        args.min_samples,
    )

    _save_cluster_umap(
        xy,
        embedding_cluster_ids,
        embedding_cluster_count,
        f"{args.species} | HDBSCAN on embeddings",
        out_dir / "embedding_cluster_on_umap",
    )
    _save_cluster_umap(
        xy,
        umap_cluster_ids,
        umap_cluster_count,
        f"{args.species} | HDBSCAN on UMAP",
        out_dir / "umap_cluster_on_umap",
    )

    embedding_analysis = _analyze_clustering(
        "embedding_cluster",
        xy,
        recording_indices,
        recordings,
        embedding_cluster_count,
        embedding_noise_points,
        args,
    )
    umap_analysis = _analyze_clustering(
        "umap_cluster",
        xy,
        recording_indices,
        recordings,
        umap_cluster_count,
        umap_noise_points,
        args,
    )

    _save_similarity_heatmap(
        embedding_analysis["similarity"],
        recordings,
        f"{args.species} | Recording UMAP overlap (Bhattacharyya)",
        out_dir / "embedding_cluster_overlap",
    )
    _save_similarity_heatmap(
        umap_analysis["similarity"],
        recordings,
        f"{args.species} | Recording UMAP overlap (Bhattacharyya)",
        out_dir / "umap_cluster_overlap",
    )

    _write_summary(
        out_dir / "summary.json",
        args,
        recordings,
        int(point_counts.sum()),
        [embedding_analysis, umap_analysis],
    )

    print(f"[cluster] recordings={len(recordings)} points={features.shape[0]}")
    print(
        "[cluster] embedding_cluster: "
        f"predicted={embedding_analysis['summary']['predicted_individuals']} "
        f"true={embedding_analysis['summary']['true_individuals']}"
    )
    print(
        "[cluster] umap_cluster: "
        f"predicted={umap_analysis['summary']['predicted_individuals']} "
        f"true={umap_analysis['summary']['true_individuals']}"
    )


if __name__ == "__main__":
    main()
