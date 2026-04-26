#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import coo_matrix, csgraph
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "individual_id"))

import plot_recording_embedding_similarity as recording_similarity  # noqa: E402


def _parse_ints(text):
    values = sorted({int(x) for x in text.split(",") if x.strip()})
    assert values and min(values) > 0
    return values


def _parse_floats(text):
    values = [float(x) for x in text.split(",") if x.strip()]
    assert values and min(values) > 0
    return values


def _sample_frames(args, rows):
    bird_ids = sorted({row["bird_id"] for row in rows})
    bird_to_code = {bird_id: index for index, bird_id in enumerate(bird_ids)}

    features = []
    point_birds = []
    point_recordings = []
    recording_birds = []
    sampled_counts = []
    for recording_index, row in enumerate(rows):
        x = row["features"]
        if x.shape[0] > args.max_points_per_recording:
            seed = recording_similarity.hashlib.sha1(
                f"{args.seed}|{row['bird_id']}|{row['recording_stem']}|graph".encode("utf-8")
            ).hexdigest()
            rng = np.random.default_rng(int(seed[:8], 16))
            frame_indices = rng.choice(x.shape[0], size=args.max_points_per_recording, replace=False)
            frame_indices.sort()
            x = x[frame_indices]

        bird_code = bird_to_code[row["bird_id"]]
        features.append(x.astype(np.float32, copy=False))
        point_birds.extend([bird_code] * int(x.shape[0]))
        point_recordings.extend([recording_index] * int(x.shape[0]))
        recording_birds.append(bird_code)
        sampled_counts.append(int(x.shape[0]))

    return {
        "features": np.vstack(features).astype(np.float32, copy=False),
        "point_birds": np.asarray(point_birds, dtype=np.int64),
        "point_recordings": np.asarray(point_recordings, dtype=np.int64),
        "recording_birds": np.asarray(recording_birds, dtype=np.int64),
        "bird_ids": np.asarray(bird_ids, dtype=object),
        "sampled_counts": np.asarray(sampled_counts, dtype=np.int64),
    }


def _device(args):
    return torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")


def _normalized_features(sampled):
    x = sampled["features"]
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _knn(args, sampled, k):
    device = _device(args)
    features = torch.from_numpy(_normalized_features(sampled)).to(device=device, dtype=torch.float32)
    point_recordings = torch.from_numpy(sampled["point_recordings"]).to(device=device, dtype=torch.long)
    total_points = int(features.shape[0])
    recording_counts = torch.bincount(point_recordings)
    if args.exclude_same_recording:
        k = min(k, total_points - int(recording_counts.max().item()))
    else:
        k = min(k, total_points - 1)
    assert k > 0

    neighbors = np.empty((total_points, k), dtype=np.int64)
    arange = torch.arange(total_points, device=device)
    for start in range(0, total_points, args.knn_chunk_size):
        end = min(start + args.knn_chunk_size, total_points)
        sims = features[start:end] @ features.T
        if args.exclude_same_recording:
            sims[point_recordings[start:end, None] == point_recordings[None, :]] = -float("inf")
        else:
            sims[torch.arange(end - start, device=device), arange[start:end]] = -float("inf")
        neighbors[start:end] = torch.topk(sims, k=k, dim=1).indices.cpu().numpy()
    return neighbors, str(device), k


def _chance(sampled):
    point_birds = sampled["point_birds"]
    point_recordings = sampled["point_recordings"]
    bird_counts = np.bincount(point_birds)
    if not sampled["exclude_same_recording"]:
        return float(np.mean((bird_counts[point_birds] - 1) / max(point_birds.size - 1, 1)))

    recording_counts = np.bincount(point_recordings)
    same = bird_counts[point_birds] - recording_counts[point_recordings]
    total = point_birds.size - recording_counts[point_recordings]
    return float(np.mean(same / np.maximum(total, 1)))


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _save_purity_plots(out_dir, args, k_values, purity, chance, query_scores, bird_matrix):
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    ax.plot(k_values, purity, marker="o", linewidth=1.6, label="Observed")
    ax.axhline(chance, color="0.35", linestyle="--", linewidth=1.0, label="Chance")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("k nearest neighbors")
    ax.set_ylabel("Same-individual fraction")
    ax.set_title(f"{args.species} | frame kNN purity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "knn_same_individual_purity.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "knn_same_individual_purity.pdf", bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.hist(query_scores, bins=np.linspace(0.0, 1.0, 51), color="tab:orange", alpha=0.75, density=True)
    ax.axvline(chance, color="0.35", linestyle="--", linewidth=1.0, label="Chance")
    ax.axvline(float(query_scores.mean()), color="tab:orange", linestyle="--", linewidth=1.0, label="Mean")
    ax.set_xlabel("Fraction of nearest neighbors from same individual")
    ax.set_ylabel("Density")
    ax.set_title(f"{args.species} | kNN occupancy (k={args.heatmap_k})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "query_knn_same_individual_histogram.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "query_knn_same_individual_histogram.pdf", bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)

    labels = [str(x) for x in args.bird_ids]
    fig_size = max(5.0, min(12.0, 0.35 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=300)
    image = ax.imshow(bird_matrix, cmap="viridis", vmin=0.0, vmax=max(float(bird_matrix.max()), 1e-6))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Neighbor individual")
    ax.set_ylabel("Query individual")
    ax.set_title(f"{args.species} | individual kNN attribution (k={args.heatmap_k})")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Fraction of neighbor slots")
    fig.tight_layout()
    fig.savefig(out_dir / "individual_knn_neighbor_heatmap.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "individual_knn_neighbor_heatmap.pdf", bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _run_purity(args, sampled, out_dir):
    k_values = _parse_ints(args.k_values)
    max_k = max(max(k_values), args.heatmap_k)
    neighbors, device, max_k = _knn(args, sampled, max_k)
    k_values = [k for k in k_values if k <= max_k]
    heatmap_k = min(args.heatmap_k, max_k)

    point_birds = sampled["point_birds"]
    same = point_birds[neighbors] == point_birds[:, None]
    cumulative = np.cumsum(same, axis=1)
    purity = np.asarray([np.mean(cumulative[:, k - 1] / float(k)) for k in k_values], dtype=np.float32)
    query_scores = (cumulative[:, heatmap_k - 1] / float(heatmap_k)).astype(np.float32, copy=False)

    bird_matrix = np.zeros((sampled["bird_ids"].size, sampled["bird_ids"].size), dtype=np.float32)
    for bird in range(sampled["bird_ids"].size):
        query = point_birds == bird
        targets = point_birds[neighbors[query, :heatmap_k]].reshape(-1)
        bird_matrix[bird] = np.bincount(targets, minlength=sampled["bird_ids"].size)
        bird_matrix[bird] /= max(float(query.sum() * heatmap_k), 1.0)

    sampled["exclude_same_recording"] = args.exclude_same_recording
    chance = _chance(sampled)
    args.bird_ids = sampled["bird_ids"]
    _save_purity_plots(out_dir, args, k_values, purity, chance, query_scores, bird_matrix)

    np.savez(
        out_dir / "knn_purity.npz",
        k_values=np.asarray(k_values, dtype=np.int64),
        purity=purity,
        chance=np.asarray(chance, dtype=np.float32),
        query_same_individual_fraction=query_scores,
        individual_neighbor_fraction=bird_matrix,
        bird_ids=sampled["bird_ids"],
    )
    _write_json(
        out_dir / "knn_purity_summary.json",
        {
            "metric": "knn_purity",
            "species": args.species,
            "device": device,
            "points": int(sampled["features"].shape[0]),
            "individuals": int(sampled["bird_ids"].size),
            "chance": chance,
            "heatmap_k": int(heatmap_k),
            "k_values": [int(k) for k in k_values],
            "purity": [float(x) for x in purity],
            "query_same_individual_fraction": recording_similarity._summarize(query_scores),
        },
    )


def _balanced_subset(args, sampled, count, repeat):
    rng = np.random.default_rng(args.seed + 1009 * count + repeat)
    birds = np.sort(rng.choice(sampled["bird_ids"].size, size=count, replace=False))
    indices = []
    for bird in birds:
        bird_indices = np.flatnonzero(sampled["point_birds"] == bird)
        assert bird_indices.size >= args.points_per_individual
        indices.append(rng.choice(bird_indices, size=args.points_per_individual, replace=False))
    indices = np.sort(np.concatenate(indices))
    return {
        "features": sampled["features"][indices],
        "point_birds": sampled["point_birds"][indices],
        "point_recordings": sampled["point_recordings"][indices],
        "bird_ids": sampled["bird_ids"][birds],
    }


def _laplacian_eigenvalues(args, sampled):
    neighbors, device, graph_k = _knn(args, sampled, args.graph_k)
    source = np.repeat(np.arange(neighbors.shape[0]), neighbors.shape[1])
    target = neighbors.reshape(-1)
    graph = coo_matrix(
        (np.ones(source.size, dtype=np.float32), (source, target)),
        shape=(neighbors.shape[0], neighbors.shape[0]),
    ).tocsr()
    graph = graph.maximum(graph.T)
    components = csgraph.connected_components(graph, directed=False, return_labels=False)
    laplacian = csgraph.laplacian(graph, normed=True)
    n_eigs = min(args.num_eigenvalues, graph.shape[0] - 2)
    assert n_eigs > 2
    values = np.sort(eigsh(laplacian, k=n_eigs, which="SM", return_eigenvectors=False))
    return values, int(components), device, int(graph_k)


def _heat_key(scale):
    return f"heat_trace_t{scale:g}".replace(".", "p")


def _fit_line(x, y):
    slope, intercept = np.polyfit(x, y, deg=1)
    predicted = slope * x + intercept
    r2 = 1.0 - float(np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2))
    return float(slope), float(intercept), predicted, r2


def _leave_one_count_out(rows, key):
    y = np.asarray([row["individuals"] for row in rows], dtype=np.float32)
    x = np.asarray([row[key] for row in rows], dtype=np.float32)
    predicted = np.empty_like(y)
    for count in sorted({row["individuals"] for row in rows}):
        train = y != count
        slope, intercept = np.polyfit(x[train], y[train], deg=1)
        predicted[~train] = slope * x[~train] + intercept
    r2 = 1.0 - float(np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2))
    mae = float(np.mean(np.abs(y - predicted)))
    return predicted, r2, mae


def _save_heat_trace_plot(out_dir, args, rows, best_key, slope, intercept, r2):
    counts = np.asarray([row["individuals"] for row in rows], dtype=np.float32)
    values = np.asarray([row[best_key] for row in rows], dtype=np.float32)
    x = np.linspace(float(values.min()), float(values.max()), 100)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    ax.scatter(values, counts, s=34, alpha=0.75)
    ax.plot(x, slope * x + intercept, color="tab:red", linewidth=1.4, label=f"linear fit, R^2={r2:.2f}")
    ax.set_xlabel(best_key.replace("_", " "))
    ax.set_ylabel("Known individuals in subset")
    ax.set_title(f"{args.species} | heat-kernel trace calibration")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "heat_trace_calibration.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "heat_trace_calibration.pdf", bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _run_heat_trace(args, sampled, out_dir):
    counts = [x for x in _parse_ints(args.counts) if x <= sampled["bird_ids"].size]
    assert counts
    scales = _parse_floats(args.heat_scales)

    rows = []
    for count in counts:
        for repeat in range(args.repeats):
            subset = _balanced_subset(args, sampled, count, repeat)
            eigenvalues, components, device, graph_k = _laplacian_eigenvalues(args, subset)
            row = {
                "individuals": int(count),
                "repeat": int(repeat),
                "points": int(subset["features"].shape[0]),
                "connected_components": components,
                "device": device,
                "graph_k": graph_k,
            }
            for scale in scales:
                row[_heat_key(scale)] = float(np.exp(-scale * eigenvalues).sum())
            rows.append(row)
            print(f"[heat-trace] count={count} repeat={repeat} {_heat_key(scales[0])}={rows[-1][_heat_key(scales[0])]:.3f}")

    y = np.asarray([row["individuals"] for row in rows], dtype=np.float32)
    fits = {}
    for scale in scales:
        key = _heat_key(scale)
        x = np.asarray([row[key] for row in rows], dtype=np.float32)
        fits[key] = _fit_line(x, y)

    best_key = max(fits, key=lambda key: fits[key][3])
    slope, intercept, predicted, r2 = fits[best_key]
    loco_predicted, loco_r2, loco_mae = _leave_one_count_out(rows, best_key)
    _save_heat_trace_plot(out_dir, args, rows, best_key, slope, intercept, r2)

    with (out_dir / "heat_trace_calibration.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) + ["predicted_individuals", "loco_predicted_individuals"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    **row,
                    "predicted_individuals": float(predicted[index]),
                    "loco_predicted_individuals": float(loco_predicted[index]),
                }
            )

    _write_json(
        out_dir / "heat_trace_summary.json",
        {
            "metric": "heat_trace",
            "species": args.species,
            "points_per_individual": int(args.points_per_individual),
            "counts": [int(x) for x in counts],
            "repeats": int(args.repeats),
            "graph_k": int(args.graph_k),
            "num_eigenvalues": int(args.num_eigenvalues),
            "heat_scales": [float(x) for x in scales],
            "best_heat_feature": best_key,
            "best_heat_slope": slope,
            "best_heat_intercept": intercept,
            "best_heat_r2": r2,
            "best_heat_leave_one_count_out_r2": loco_r2,
            "best_heat_leave_one_count_out_mae": loco_mae,
            "heat_r2_by_feature": {key: fit[3] for key, fit in fits.items()},
            "rows": rows,
        },
    )


def _load_table(args):
    args.annotation_json = str(Path(args.annotation_json).resolve())
    args.spec_dir = str(Path(args.spec_dir).resolve())
    args.run_dir = str(recording_similarity._resolve_run_dir(args.run_dir))
    args.out_dir = str(Path(args.out_dir).resolve())

    model_state = recording_similarity.extract_embedding.load_model_state(
        {"run_dir": args.run_dir, "checkpoint": args.checkpoint}
    )
    if args.spec_normalization == "auto":
        args.spec_normalization, args.normalization_stats_dir = (
            recording_similarity.extract_embedding.get_native_input_normalization(model_state)
        )
    return recording_similarity._build_recording_table(args, model_state)


def main():
    parser = argparse.ArgumentParser(description="Evaluate kNN graph individual-id metrics.")
    parser.add_argument("--metric", default="all", choices=["all", "purity", "heat_trace"])
    parser.add_argument("--species", required=True)
    parser.add_argument("--annotation_json", required=True)
    parser.add_argument("--spec_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out_dir", default=str(ROOT / "results" / "individual_id_knn_graph_metrics"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--recording_mode", default="full_recordings", choices=["events", "full_recordings"])
    parser.add_argument("--songs_per_bird", type=int, default=30)
    parser.add_argument("--max_birds", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_points_per_recording", type=int, default=200)
    parser.add_argument("--k_values", default="1,2,5,10,20,50,100")
    parser.add_argument("--heatmap_k", type=int, default=50)
    parser.add_argument("--points_per_individual", type=int, default=200)
    parser.add_argument("--counts", default="2,4,8,12,16,20,24,28,32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--graph_k", type=int, default=50)
    parser.add_argument("--heat_scales", default="1,2,5,10,20,50")
    parser.add_argument("--num_eigenvalues", type=int, default=80)
    parser.add_argument("--knn_chunk_size", type=int, default=512)
    parser.add_argument("--exclude_same_recording", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--embedding_variant", default="before", choices=["before", "after"])
    parser.add_argument("--feature_postprocess", default="pca_whiten_l2", choices=["none", "pca_whiten_l2", "whiten_l2"])
    parser.add_argument("--feature_postprocess_dim", type=int, default=1024)
    parser.add_argument("--feature_postprocess_load", default=None)
    parser.add_argument("--feature_postprocess_save", default=None)
    parser.add_argument("--encoder_layer_idx", type=int, default=None)
    parser.add_argument("--drop_silence", action="store_true")
    parser.add_argument(
        "--spec_normalization",
        default="auto",
        choices=[
            "auto",
            "none",
            "audio_params",
            "per_recording_cmvn",
            "per_recording_cmvn_rescaled_to_target_stats",
            "per_model_input_zscore",
        ],
    )
    parser.add_argument("--normalization_stats_dir", default=None)
    args = parser.parse_args()

    assert args.max_points_per_recording > 0
    assert args.heatmap_k > 0
    assert args.points_per_individual > 0
    assert args.repeats > 0
    assert args.graph_k > 0
    assert args.num_eigenvalues > 2

    rows, feature_postprocess = _load_table(args)
    sampled = _sample_frames(args, rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "feature_postprocess.json", recording_similarity._feature_postprocess_summary(feature_postprocess))

    if args.metric in {"all", "purity"}:
        _run_purity(args, sampled, out_dir)
    if args.metric in {"all", "heat_trace"}:
        _run_heat_trace(args, sampled, out_dir)

    print(f"[knn-graph-metrics] metric={args.metric} species={args.species} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
