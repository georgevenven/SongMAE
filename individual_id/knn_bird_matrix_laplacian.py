#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from scipy.sparse import coo_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import svds

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import extract_embedding  # noqa: E402

NAME_ALIASES = {
    "zf": "Zebra finch",
    "bf": "Bengalese finch",
    "canary": "Canary",
    "ovenbird": "Ovenbird",
    "chiffchaff": "Chiffchaff",
    "european_starling": "European starling",
    "tree_pipit": "Tree pipit",
    "little_owl": "Little owl",
}

KNN_CMAP = LinearSegmentedColormap.from_list("knn_overlap", ["#fffdf7", "#ffe66d", "#d7301f"])
KNN_NORM_GAMMA = 0.45

SPECIES = {
    "zf": ("Zebra_Finch", "files/zf_annotations.json", "/media/george-vengrovski/disk2/specs/zf_64hop_32khz", "full_recordings"),
    "bf": ("bf", "files/bf_annotations.json", "/media/george-vengrovski/disk2/specs/bf_64hop_32khz", "full_recordings"),
    "canary": ("canary", "files/canary_annotations_for_individual_id.json", "/media/george-vengrovski/disk2/specs/canary_individual_identification_64hop_32khz", "full_recordings"),
    "ovenbird": ("ovenbird", "files/lapp_ovenbird.json", "/media/george-vengrovski/disk2/specs/ovenbird_lapp_sample_64hop_32khz", "events"),
    "chiffchaff": ("chiffchaff", "files/chiffchaff_annotations.json", "/media/george-vengrovski/disk2/specs/chiffchaff_64hop_32khz", "full_recordings"),
    "european_starling": ("european_starling", "files/european_starling_annotations_unprefixed.json", "/media/george-vengrovski/disk2/specs/european_starling_64hop_32khz", "full_recordings"),
    "tree_pipit": ("tree_pipit", "files/tree_pipit_annotations.json", "/media/george-vengrovski/disk2/specs/tree_pipit_64hop_32khz", "full_recordings"),
    "little_owl": ("little_owl", "files/little_owl_annotations.json", "/media/george-vengrovski/disk2/specs/little_owl_64hop_32khz", "full_recordings"),
}


def _parse_ints(text):
    values = sorted({int(x) for x in text.split(",") if x.strip()})
    assert values and min(values) > 0
    return values


def _subset_counts(text, max_count):
    if text == "all":
        return list(range(1, max_count + 1))
    return [x for x in _parse_ints(text) if x <= max_count]


def _resolve_run_dir(text):
    path = Path(text)
    if path.is_absolute() and path.is_dir():
        return path
    for base in [ROOT, ROOT / "runs"]:
        candidate = base / path
        if candidate.is_dir():
            return candidate.resolve()
    raise SystemExit(f"unable to resolve run_dir: {text}")


def _load_stems(annotation_json):
    data = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
    by_bird = {}
    for item in data["recordings"]:
        recording = item["recording"]
        bird_id = str(recording["bird_id"]).strip()
        stem = Path(recording["filename"]).stem
        by_bird.setdefault(bird_id, set()).add(stem)
    return {bird_id: sorted(stems) for bird_id, stems in by_bird.items()}


def _pick(stems, limit, seed, bird_id):
    if limit <= 0 or len(stems) <= limit:
        return list(stems)
    bird_seed = int(hashlib.sha1(bird_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed + bird_seed)
    indices = np.sort(rng.choice(len(stems), size=limit, replace=False))
    return [stems[index] for index in indices]


def _selected_recordings(args):
    species, annotation_json, spec_dir, recording_mode = SPECIES[args.species_key]
    args.species = species
    args.annotation_json = str(ROOT / annotation_json)
    args.spec_dir = spec_dir
    args.recording_mode = recording_mode
    args.run_dir = str(_resolve_run_dir(args.run_dir))

    rows = []
    for bird_id, stems in sorted(_load_stems(args.annotation_json).items()):
        if len(stems) < args.min_songs_per_bird:
            continue
        for stem in _pick(stems, args.songs_per_bird, args.seed, bird_id):
            rows.append({"bird_id": bird_id, "recording_stem": stem})
    assert rows, f"no singers have at least {args.min_songs_per_bird} recordings"
    return rows


def _feature_key(args):
    if args.embedding_variant == "before":
        return "encoded_embeddings_before_pos_removal"
    assert args.embedding_variant == "after"
    return "encoded_embeddings_after_pos_removal"


def _extract(args, selected):
    model_state = extract_embedding.load_model_state({"run_dir": args.run_dir, "checkpoint": args.checkpoint})
    if args.spec_normalization == "auto":
        args.spec_normalization, args.normalization_stats_dir = extract_embedding.get_native_input_normalization(model_state)
    key = _feature_key(args)
    request = {
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "spec_dir": args.spec_dir,
        "json_path": args.annotation_json,
        "recording_stems": [row["recording_stem"] for row in selected],
        "recording_mode": args.recording_mode,
        "encoder_layer_idx": args.encoder_layer_idx,
        "spec_normalization": args.spec_normalization,
        "normalization_stats_dir": args.normalization_stats_dir,
        "minimal_output": args.embedding_variant == "before",
        "embedding_postprocess": "pca_whiten_l2",
        "embedding_postprocess_dim": 1024,
        "embedding_postprocess_key": key,
        "embedding_postprocess_load": None,
        "embedding_postprocess_save": None,
    }
    extracted = extract_embedding.extract_recording_embeddings_with_state(request, model_state)
    arrays_by_stem = {row["recording_stem"]: [] for row in selected}
    for segment in extracted["segments"]:
        x = segment[key].astype(np.float32, copy=False)
        if x.size:
            arrays_by_stem[segment["recording_stem"]].append(x)

    rows = []
    for row in selected:
        arrays = arrays_by_stem[row["recording_stem"]]
        if arrays:
            rows.append({**row, "features": np.vstack(arrays).astype(np.float32, copy=False)})
    assert rows
    return rows


def _sample(args, rows):
    bird_ids = sorted({row["bird_id"] for row in rows})
    bird_to_code = {bird_id: index for index, bird_id in enumerate(bird_ids)}
    features = []
    point_birds = []
    point_recordings = []
    recording_birds = []
    recording_stems = []
    counts = []
    for recording_index, row in enumerate(rows):
        x = row["features"]
        if args.max_points_per_recording > 0 and x.shape[0] > args.max_points_per_recording:
            key = f"{args.seed}|{row['bird_id']}|{row['recording_stem']}|bird-matrix"
            rng = np.random.default_rng(int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16))
            indices = np.sort(rng.choice(x.shape[0], size=args.max_points_per_recording, replace=False))
            x = x[indices]
        bird = bird_to_code[row["bird_id"]]
        features.append(x)
        point_birds.extend([bird] * x.shape[0])
        point_recordings.extend([recording_index] * x.shape[0])
        recording_birds.append(bird)
        recording_stems.append(row["recording_stem"])
        counts.append(int(x.shape[0]))
    return {
        "features": np.vstack(features).astype(np.float32, copy=False),
        "point_birds": np.asarray(point_birds, dtype=np.int64),
        "point_recordings": np.asarray(point_recordings, dtype=np.int64),
        "recording_birds": np.asarray(recording_birds, dtype=np.int64),
        "recording_stems": np.asarray(recording_stems, dtype=object),
        "bird_ids": np.asarray(bird_ids, dtype=object),
        "sampled_counts": np.asarray(counts, dtype=np.int64),
    }


def _knn(args, sampled, k):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x = sampled["features"]
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    features = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    recordings = torch.from_numpy(sampled["point_recordings"]).to(device=device, dtype=torch.long)
    max_same_recording = int(torch.bincount(recordings).max().item())
    k = min(k, features.shape[0] - max_same_recording if args.exclude_same_recording else features.shape[0] - 1)
    assert k > 0

    neighbors = np.empty((features.shape[0], k), dtype=np.int64)
    arange = torch.arange(features.shape[0], device=device)
    for start in range(0, features.shape[0], args.knn_chunk_size):
        end = min(start + args.knn_chunk_size, features.shape[0])
        sims = features[start:end] @ features.T
        if args.exclude_same_recording:
            sims[recordings[start:end, None] == recordings[None, :]] = -float("inf")
        else:
            sims[torch.arange(end - start, device=device), arange[start:end]] = -float("inf")
        neighbors[start:end] = torch.topk(sims, k=k, dim=1).indices.cpu().numpy()
    return neighbors, str(device), k


def _purity(sampled, neighbors, k_values):
    labels = sampled["point_birds"]
    same = labels[neighbors] == labels[:, None]
    cumulative = np.cumsum(same, axis=1)
    return {k: float(np.mean(cumulative[:, k - 1] / k)) for k in k_values}


def _chance(sampled):
    labels = sampled["point_birds"]
    recordings = sampled["point_recordings"]
    bird_counts = np.bincount(labels)
    if not sampled["exclude_same_recording"]:
        return float(np.mean((bird_counts[labels] - 1) / max(labels.size - 1, 1)))
    recording_counts = np.bincount(recordings)
    same = bird_counts[labels] - recording_counts[recordings]
    total = labels.size - recording_counts[recordings]
    return float(np.mean(same / np.maximum(total, 1)))


def _bird_matrix(sampled, neighbors, k):
    labels = sampled["point_birds"]
    n_birds = sampled["bird_ids"].size
    matrix = np.zeros((n_birds, n_birds), dtype=np.float32)
    for bird in range(n_birds):
        query = labels == bird
        targets = labels[neighbors[query, :k]].reshape(-1)
        matrix[bird] = np.bincount(targets, minlength=n_birds)
        matrix[bird] /= max(float(query.sum() * k), 1.0)
    return matrix


def _recording_matrix(sampled, neighbors, k):
    recordings = sampled["point_recordings"]
    n_recordings = sampled["sampled_counts"].size
    matrix = np.zeros((n_recordings, n_recordings), dtype=np.float32)
    for recording in range(n_recordings):
        query = recordings == recording
        targets = recordings[neighbors[query, :k]].reshape(-1)
        matrix[recording] = np.bincount(targets, minlength=n_recordings)
        matrix[recording] /= max(float(query.sum() * k), 1.0)
    return matrix


def _laplacian_summary(matrix, num_eigenvalues):
    adjacency = (matrix + matrix.T) * 0.5
    np.fill_diagonal(adjacency, 0.0)
    laplacian = csgraph.laplacian(adjacency, normed=True)
    n_eigs = max(2, min(num_eigenvalues, adjacency.shape[0] - 1))
    if adjacency.shape[0] <= 2:
        values = np.linalg.eigvalsh(laplacian)
    else:
        values = np.sort(eigsh(laplacian, k=n_eigs, which="SM", return_eigenvectors=False))
    gaps = np.diff(values)
    estimate = int(np.argmax(gaps) + 1) if gaps.size else 1
    return {
        "eigengap_estimate": estimate,
        "eigenvalues": [float(x) for x in values],
        "gaps": [float(x) for x in gaps],
    }


def _top_k_graph(matrix, top_k):
    rows = []
    cols = []
    vals = []
    for row_index in range(matrix.shape[0]):
        row = matrix[row_index]
        keep = np.flatnonzero(row > 0)
        if keep.size == 0:
            continue
        if keep.size > top_k:
            keep = keep[np.argpartition(row[keep], -top_k)[-top_k:]]
        rows.extend([row_index] * keep.size)
        cols.extend(keep.tolist())
        vals.extend(row[keep].tolist())
    graph = coo_matrix((vals, (rows, cols)), shape=matrix.shape).tocsr()
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def _graph_eigengap(graph, num_eigenvalues):
    laplacian = csgraph.laplacian(graph, normed=True)
    n_eigs = max(2, min(num_eigenvalues, graph.shape[0] - 1))
    values = np.sort(eigsh(laplacian, k=n_eigs, which="SM", return_eigenvectors=False, tol=1e-3))
    gaps = np.diff(values)
    return int(np.argmax(gaps) + 1), values, gaps


def _stable_rank(matrix):
    graph = coo_matrix(matrix).tocsr()
    graph = (graph + graph.T) * 0.5
    graph.setdiag(0)
    graph.eliminate_zeros()
    return _stable_rank_graph(graph)


def _row_normalized_stable_rank(matrix):
    graph = coo_matrix(matrix).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    row_sums = np.asarray(graph.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    graph = graph.multiply(1.0 / row_sums[:, None]).tocsr()
    return _stable_rank_graph(graph)


def _stable_rank_graph(graph):
    if graph.nnz == 0:
        return 0.0
    frobenius_squared = float(graph.multiply(graph).sum())
    try:
        largest = float(svds(graph, k=1, return_singular_vectors=False, tol=1e-3)[0])
    except ValueError:
        largest = float(np.linalg.svd(graph.toarray(), compute_uv=False)[0])
    return frobenius_squared / max(largest**2, 1e-12)


def _linear_fit_r2(rows, prediction_key):
    y = np.asarray([row["true_count"] for row in rows], dtype=np.float64)
    x = np.asarray([row[prediction_key] for row in rows], dtype=np.float64)
    if np.allclose(x, x[0]):
        return 0.0
    slope, intercept = np.polyfit(x, y, deg=1)
    pred = slope * x + intercept
    denom = np.sum((y - y.mean()) ** 2)
    if denom == 0:
        return 0.0
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def _subset_experiment(args, out_dir):
    path = out_dir / "knn_attribution_matrices.npz"
    assert path.exists(), f"run matrix build first: {path}"
    data = np.load(path, allow_pickle=True)
    full_matrix = data["recording_matrix"].astype(np.float32, copy=False)
    recording_birds = data["recording_birds"].astype(np.int64, copy=False)
    bird_ids = data["bird_ids"]
    counts = _subset_counts(args.subset_counts, bird_ids.size)
    recordings_per_bird = [int(x) for x in args.subset_recordings_per_bird.split(",") if x.strip()]
    if args.balanced_max_recordings_per_bird:
        recordings_per_bird = [-2]
    elif args.random_fraction_recordings_per_bird:
        recordings_per_bird = [-3]
    assert counts and recordings_per_bird

    rng = np.random.default_rng(args.seed)
    rows = []
    for true_count in counts:
        for per_bird in recordings_per_bird:
            for repeat in range(args.subset_repeats):
                birds = np.sort(rng.choice(bird_ids.size, size=true_count, replace=False))
                selected_recordings = [np.flatnonzero(recording_birds == bird) for bird in birds]
                if per_bird == -2:
                    n_balanced = min(x.size for x in selected_recordings)
                indices = []
                for bird_recordings in selected_recordings:
                    if per_bird == -2:
                        bird_recordings = np.sort(rng.choice(bird_recordings, size=n_balanced, replace=False))
                    elif per_bird == -3:
                        low = max(1, int(np.ceil(args.min_recording_fraction_per_bird * bird_recordings.size)))
                        n_recordings = rng.integers(low, bird_recordings.size + 1)
                        bird_recordings = np.sort(rng.choice(bird_recordings, size=n_recordings, replace=False))
                    elif per_bird > 0 and bird_recordings.size > per_bird:
                        bird_recordings = np.sort(rng.choice(bird_recordings, size=per_bird, replace=False))
                    indices.append(bird_recordings)
                indices = np.sort(np.concatenate(indices))
                subset = full_matrix[np.ix_(indices, indices)]
                graph = _top_k_graph(subset, args.graph_top_k)
                estimate, values, gaps = _graph_eigengap(graph, args.num_eigenvalues)
                stable_rank = _stable_rank(subset)
                rows.append(
                    {
                        "true_count": int(true_count),
                        "recordings_per_bird": int(per_bird),
                        "repeat": int(repeat),
                        "recordings": int(indices.size),
                        "components": int(csgraph.connected_components(graph, directed=False, return_labels=False)),
                        "eigengap_estimate": int(estimate),
                        "stable_rank": stable_rank,
                        "abs_error": int(abs(estimate - true_count)),
                        "top_gap": float(gaps.max()) if gaps.size else 0.0,
                    }
                )

    csv_path = out_dir / "subset_recording_count_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_recording_count = {}
    for per_bird in recordings_per_bird:
        group = [row for row in rows if row["recordings_per_bird"] == per_bird]
        by_recording_count[str(per_bird)] = {
            "linear_fit_r2": _linear_fit_r2(group, "eigengap_estimate"),
            "stable_rank_r2": _linear_fit_r2(group, "stable_rank"),
            "mae": float(np.mean([row["abs_error"] for row in group])),
            "rows": len(group),
        }
    summary = {
        "method": "recording_matrix_top_k_laplacian_subset_count_sweep",
        "source": str(path),
        "top_k": int(args.graph_top_k),
        "subset_counts": counts,
        "subset_recordings_per_bird": recordings_per_bird,
        "subset_repeats": int(args.subset_repeats),
        "overall_linear_fit_r2": _linear_fit_r2(rows, "eigengap_estimate"),
        "overall_stable_rank_r2": _linear_fit_r2(rows, "stable_rank"),
        "overall_mae": float(np.mean([row["abs_error"] for row in rows])),
        "by_recordings_per_bird": by_recording_count,
    }
    (out_dir / "subset_recording_count_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _read_subset_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append(
                {
                    "true_count": int(row["true_count"]),
                    "recordings_per_bird": int(row["recordings_per_bird"]),
                    "repeat": int(row["repeat"]),
                    "recordings": int(row["recordings"]),
                    "eigengap_estimate": int(row["eigengap_estimate"]),
                    "stable_rank": float(row.get("stable_rank", 0.0)),
                }
            )
    assert rows
    return rows


def _linear_fit(rows):
    x = np.asarray([row["eigengap_estimate"] for row in rows], dtype=np.float64)
    return _linear_fit_xy(rows, x)


def _linear_fit_xy(rows, x):
    y = np.asarray([row["true_count"] for row in rows], dtype=np.float64)
    slope, intercept = np.polyfit(x, y, deg=1)
    predicted = slope * x + intercept
    r2 = 1.0 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2)
    return x, y, float(slope), float(intercept), float(r2)


def _plot_subset_experiment(args, out_dir):
    csv_path = Path(args.plot_subset_csv) if args.plot_subset_csv else out_dir / "subset_recording_count_sweep.csv"
    rows = _read_subset_rows(csv_path)
    x_values = np.asarray([row[args.plot_prediction_key] for row in rows], dtype=np.float64)
    x, y, slope, intercept, r2 = _linear_fit_xy(rows, x_values)

    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=300)
    ax.scatter(x, y, s=34, alpha=0.78, color="#2f6fbb", edgecolor="white", linewidth=0.45)

    x_line = np.linspace(max(0.0, float(x.min()) - 1.0), float(x.max()) + 1.0, 200)
    ax.plot(x_line, slope * x_line + intercept, color="black", linestyle="--", linewidth=1.0)
    ax.text(
        0.04,
        0.96,
        f"$R^2$ = {r2:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "0.82", "boxstyle": "round,pad=0.28", "alpha": 0.92},
    )
    ax.set_xlabel(args.plot_x_label)
    ax.set_ylabel("Known number of singers")
    ax.set_title(args.plot_title or f"{args.species_key} recording count proxy")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_base = Path(args.plot_out) if args.plot_out else out_dir / f"{csv_path.stem}_regression"
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base) + ".png", bbox_inches="tight", dpi=300)
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    summary = {
        "source": str(csv_path),
        "png": str(out_base) + ".png",
        "pdf": str(out_base) + ".pdf",
        "r2": r2,
        "slope": slope,
        "intercept": intercept,
        "prediction_key": args.plot_prediction_key,
        "rows": len(rows),
    }
    print(json.dumps(summary, indent=2))


def _panel_grid(items, figsize):
    fig, axes = plt.subplots(2, 4, figsize=figsize, dpi=300)
    return fig, list(zip(axes.flat, items))


def _save_all_species_heatmaps(root, key, filename):
    species_keys = list(NAME_ALIASES)
    fig, panels = _panel_grid(species_keys, (12, 6.2))
    for ax, species_key in panels:
        data = np.load(root / species_key / "knn_attribution_matrices.npz", allow_pickle=True)
        matrix = data[key].astype(np.float32, copy=False)
        percentile = 97.5 if key == "bird_matrix" else 99.5
        vmax = max(float(np.percentile(matrix, percentile)), 1e-6)
        ax.imshow(matrix, cmap=KNN_CMAP, norm=PowerNorm(gamma=KNN_NORM_GAMMA, vmin=0.0, vmax=vmax))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(NAME_ALIASES[species_key], fontsize=15)
    for ax in fig.axes[::4]:
        ax.set_ylabel("Query", fontsize=14)
    for ax in fig.axes[4:]:
        ax.set_xlabel("Neighbor recording", fontsize=14)
    fig.tight_layout()
    fig.savefig(root / f"{filename}.png", bbox_inches="tight", dpi=300)
    fig.savefig(root / f"{filename}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _save_all_species_scatter(root, prediction_key, x_label, filename):
    species_keys = list(NAME_ALIASES)
    fig, panels = _panel_grid(species_keys, (12, 6.2))
    for ax, species_key in panels:
        path = root / species_key / "subset_recording_count_sweep_all_recordings_all_counts.csv"
        rows = _read_subset_rows(path)
        x_values = np.asarray([row[prediction_key] for row in rows], dtype=np.float64)
        x, y, slope, intercept, r2 = _linear_fit_xy(rows, x_values)
        ax.scatter(x, y, s=14, alpha=0.72, color="#2f6fbb", edgecolor="white", linewidth=0.25)
        x_line = np.linspace(max(0.0, float(x.min()) - 1.0), float(x.max()) + 1.0, 200)
        ax.plot(x_line, slope * x_line + intercept, color="black", linestyle="--", linewidth=0.8)
        ax.text(0.05, 0.94, f"$R^2$ = {r2:.3f}", transform=ax.transAxes, va="top", ha="left", fontsize=9)
        ax.set_title(NAME_ALIASES[species_key], fontsize=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in fig.axes[::4]:
        ax.set_ylabel("Known singers", fontsize=14)
    for ax in fig.axes[4:]:
        ax.set_xlabel(x_label, fontsize=14)
    fig.tight_layout()
    fig.savefig(root / f"{filename}.png", bbox_inches="tight", dpi=300)
    fig.savefig(root / f"{filename}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _variable_recording_rows(root, species_key, min_fraction, repeats, seed):
    data = np.load(root / species_key / "knn_attribution_matrices.npz", allow_pickle=True)
    matrix = data["recording_matrix"].astype(np.float32, copy=False)
    recording_birds = data["recording_birds"].astype(np.int64, copy=False)
    n_birds = len(data["bird_ids"])
    by_bird = [np.flatnonzero(recording_birds == bird) for bird in range(n_birds)]
    max_recordings = min(len(x) for x in by_bird)
    min_recordings = max(1, math.ceil(min_fraction * max_recordings))
    rng = np.random.default_rng(seed)
    rows = []
    for true_count in range(1, n_birds + 1):
        for recordings_per_bird in range(min_recordings, max_recordings + 1):
            for repeat in range(repeats):
                birds = np.sort(rng.choice(n_birds, size=true_count, replace=False))
                indices = []
                for bird in birds:
                    indices.append(rng.choice(by_bird[bird], size=recordings_per_bird, replace=False))
                indices = np.sort(np.concatenate(indices))
                rows.append(
                    {
                        "true_count": int(true_count),
                        "recordings_per_bird": int(recordings_per_bird),
                        "repeat": int(repeat),
                        "recordings": int(indices.size),
                        "stable_rank": _stable_rank(matrix[np.ix_(indices, indices)]),
                        "row_normalized_stable_rank": _row_normalized_stable_rank(matrix[np.ix_(indices, indices)]),
                    }
                )
    return rows, min_recordings, max_recordings


def _write_rows_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_stable_rank_rows(rows, title, out_base):
    x_values = np.asarray([row["stable_rank"] for row in rows], dtype=np.float64)
    x, y, slope, intercept, r2 = _linear_fit_xy(rows, x_values)
    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=300)
    ax.scatter(x, y, s=18, alpha=0.38, color="#2f6fbb", edgecolor="none")
    x_line = np.linspace(float(x.min()), float(x.max()), 200)
    ax.plot(x_line, slope * x_line + intercept, color="black", linestyle="--", linewidth=1.0)
    ax.text(0.04, 0.96, f"$R^2$ = {r2:.3f}", transform=ax.transAxes, va="top", ha="left", fontsize=10)
    ax.set_title(title)
    ax.set_xlabel("Stable rank")
    ax.set_ylabel("Known singers")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(str(out_base) + ".png", bbox_inches="tight", dpi=300)
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return r2


def _plot_variable_collage(root, species_keys, key, csv_stem, filename):
    fig, panels = _panel_grid(species_keys, (12, 6.2))
    for ax, species_key in panels:
        path = root / species_key / f"{csv_stem}.csv"
        with path.open(newline="", encoding="utf-8") as f:
            rows = [{"true_count": int(row["true_count"]), key: float(row[key])} for row in csv.DictReader(f)]
        x_values = np.asarray([row[key] for row in rows], dtype=np.float64)
        x, y, slope, intercept, r2 = _linear_fit_xy(rows, x_values)
        ax.scatter(x, y, s=10, alpha=0.31, color="#2f6fbb", edgecolor="none")
        x_line = np.linspace(float(x.min()), float(x.max()), 200)
        ax.plot(x_line, slope * x_line + intercept, color="black", linestyle="--", linewidth=0.8)
        ax.text(0.05, 0.94, f"$R^2$ = {r2:.3f}", transform=ax.transAxes, va="top", ha="left", fontsize=9)
        ax.set_title(NAME_ALIASES[species_key], fontsize=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in fig.axes[::4]:
        ax.set_ylabel("Known singers", fontsize=14)
    for ax in fig.axes[4:]:
        ax.set_xlabel("Stable rank", fontsize=14)
    fig.tight_layout()
    fig.savefig(root / f"{filename}.png", bbox_inches="tight", dpi=300)
    fig.savefig(root / f"{filename}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _save_variable_recording_stable_rank(root, min_fraction, repeats, seed):
    species_keys = list(NAME_ALIASES)
    summary = []
    stem = f"stable_rank_variable_recordings_per_bird_min{int(min_fraction * 100)}pct"
    for species_key in species_keys:
        rows, min_recordings, max_recordings = _variable_recording_rows(root, species_key, min_fraction, repeats, seed)
        _write_rows_csv(root / species_key / f"{stem}.csv", rows)
        raw_r2 = _plot_stable_rank_rows(rows, NAME_ALIASES[species_key], root / species_key / stem)
        norm_r2 = _plot_stable_rank_rows(
            [{**row, "stable_rank": row["row_normalized_stable_rank"]} for row in rows],
            NAME_ALIASES[species_key],
            root / species_key / f"row_normalized_{stem}",
        )
        summary.append(
            {
                "species": species_key,
                "stable_rank_r2": raw_r2,
                "row_normalized_stable_rank_r2": norm_r2,
                "rows": len(rows),
                "min_recordings_per_bird": min_recordings,
                "max_recordings_per_bird": max_recordings,
            }
        )

    filename = f"all_species_{stem}"
    _plot_variable_collage(root, species_keys, "stable_rank", stem, filename)
    _plot_variable_collage(root, species_keys, "row_normalized_stable_rank", stem, f"all_species_row_normalized_{stem}")
    _write_rows_csv(root / f"stable_rank_variable_recordings_per_bird_min{int(min_fraction * 100)}pct_summary.csv", summary)


def _save_all_species_purity(root):
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=300)
    for species_key in NAME_ALIASES:
        path = root / species_key / "knn_purity.csv"
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        x = np.asarray([int(row["k"]) for row in rows], dtype=np.int64)
        y = np.asarray([float(row["purity"]) for row in rows], dtype=np.float64)
        ax.plot(x, y, marker="o", markersize=3.2, linewidth=1.15, label=NAME_ALIASES[species_key])
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in x])
    ax.set_ylim(0, 1)
    ax.set_xlabel("k nearest neighbors", fontsize=14)
    ax.set_ylabel("Same-singer fraction", fontsize=14)
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(root / "all_species_knn_purity.png", bbox_inches="tight", dpi=300)
    fig.savefig(root / "all_species_knn_purity.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _save_all_species_summary_csv(root):
    rows = []
    for species_key in NAME_ALIASES:
        summary = json.loads((root / species_key / "summary.json").read_text(encoding="utf-8"))
        subset = json.loads(
            (root / species_key / "subset_recording_count_sweep_all_recordings_all_counts_summary.json").read_text(
                encoding="utf-8"
            )
        )
        rows.append(
            {
                "species": species_key,
                "recordings": summary["recordings"],
                "points": summary["points"],
                "full_recording_eigengap": summary["recording_laplacian"]["eigengap_estimate"],
                "subset_r2": subset["overall_linear_fit_r2"],
                "stable_rank_r2": subset["overall_stable_rank_r2"],
                "subset_mae": subset["overall_mae"],
            }
        )
    with (root / "all_species_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_all_species_summary(args):
    root = Path(args.out_dir)
    _save_all_species_heatmaps(root, "bird_matrix", "all_species_bird_knn_heatmaps")
    _save_all_species_heatmaps(root, "recording_matrix", "all_species_recording_knn_heatmaps")
    _save_all_species_scatter(
        root,
        "eigengap_estimate",
        "Eigengap estimate",
        "all_species_recording_count_proxy",
    )
    _save_all_species_scatter(
        root,
        "stable_rank",
        "Stable rank",
        "all_species_stable_rank_proxy",
    )
    _save_variable_recording_stable_rank(root, 0.30, args.variable_recording_repeats, args.seed)
    _save_all_species_purity(root)
    _save_all_species_summary_csv(root)
    print(
        json.dumps(
            {
                "out_dir": str(root),
                "figures": [
                    "all_species_bird_knn_heatmaps.png",
                    "all_species_recording_knn_heatmaps.png",
                    "all_species_recording_count_proxy.png",
                    "all_species_stable_rank_proxy.png",
                    "all_species_stable_rank_variable_recordings_per_bird_min30pct.png",
                    "all_species_row_normalized_stable_rank_variable_recordings_per_bird_min30pct.png",
                    "all_species_knn_purity.png",
                    "all_species_summary.csv",
                ],
            },
            indent=2,
        )
    )


def _save_purity_plot(args, out_dir, k_values, purity, chance):
    x = np.asarray(k_values, dtype=np.int64)
    y = np.asarray([purity[k] for k in k_values], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.8, 4.4), dpi=300)
    ax.plot(x, y, color="#2f6fbb", marker="o", markersize=4.5, linewidth=1.4)
    ax.axhline(chance, color="black", linestyle="--", linewidth=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in x])
    ax.set_ylim(0.0, min(1.0, max(0.2, float(y.max()) + 0.08)))
    ax.set_xlabel("k nearest neighbors", fontsize=14)
    ax.set_ylabel("Same-singer fraction", fontsize=14)
    ax.set_title(NAME_ALIASES[args.species_key], fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "knn_purity.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "knn_purity.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _save_bird_heatmap(args, out_dir, matrix, bird_ids):
    fig_size = max(5.0, min(12.0, 0.35 * len(bird_ids)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=300)
    vmax = max(float(np.percentile(matrix, 97.5)), 1e-6)
    ax.imshow(matrix, cmap=KNN_CMAP, norm=PowerNorm(gamma=KNN_NORM_GAMMA, vmin=0.0, vmax=vmax))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Neighbor recording", fontsize=24)
    ax.set_ylabel("Query", fontsize=24)
    ax.set_title(NAME_ALIASES[args.species_key], fontsize=24)
    fig.tight_layout()
    fig.savefig(out_dir / "bird_knn_attribution.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "bird_knn_attribution.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _save_recording_heatmap(args, out_dir, matrix):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    vmax = max(float(np.percentile(matrix, 99.5)), 1e-6)
    ax.imshow(matrix, cmap=KNN_CMAP, norm=PowerNorm(gamma=KNN_NORM_GAMMA, vmin=0.0, vmax=vmax))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Neighbor recording", fontsize=24)
    ax.set_ylabel("Query", fontsize=24)
    ax.set_title(NAME_ALIASES[args.species_key], fontsize=24)
    fig.tight_layout()
    fig.savefig(out_dir / "recording_knn_attribution.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "recording_knn_attribution.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _write_outputs(args, sampled, neighbors, device, actual_k, out_dir):
    k_values = [k for k in _parse_ints(args.k_values) if k <= actual_k]
    matrix_k = min(args.matrix_k, actual_k)
    sampled["exclude_same_recording"] = args.exclude_same_recording
    purity = _purity(sampled, neighbors, k_values)
    chance = _chance(sampled)
    bird_matrix = _bird_matrix(sampled, neighbors, matrix_k)
    recording_matrix = _recording_matrix(sampled, neighbors, matrix_k)
    recording_laplacian = _laplacian_summary(recording_matrix, args.num_eigenvalues)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_purity_plot(args, out_dir, k_values, purity, chance)
    _save_bird_heatmap(args, out_dir, bird_matrix, sampled["bird_ids"])
    _save_recording_heatmap(args, out_dir, recording_matrix)

    np.savez_compressed(
        out_dir / "knn_attribution_matrices.npz",
        k_values=np.asarray(k_values, dtype=np.int64),
        purity=np.asarray([purity[k] for k in k_values], dtype=np.float32),
        bird_matrix=bird_matrix,
        recording_matrix=recording_matrix,
        bird_ids=sampled["bird_ids"],
        recording_birds=sampled["recording_birds"],
        recording_stems=sampled["recording_stems"],
        recording_eigenvalues=np.asarray(recording_laplacian["eigenvalues"], dtype=np.float32),
        recording_gaps=np.asarray(recording_laplacian["gaps"], dtype=np.float32),
    )
    with (out_dir / "knn_purity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "purity", "chance"])
        writer.writeheader()
        for k in k_values:
            writer.writerow({"k": k, "purity": purity[k], "chance": chance})
    summary = {
        "species_key": args.species_key,
        "pca_dim": 1024,
        "device": device,
        "recordings": int(sampled["sampled_counts"].size),
        "points": int(sampled["features"].shape[0]),
        "max_points_per_recording": int(args.max_points_per_recording),
        "songs_per_bird": int(args.songs_per_bird),
        "matrix_k": int(matrix_k),
        "chance": chance,
        "purity": {str(k): purity[k] for k in k_values},
        "bird_diag_mean": float(np.diag(bird_matrix).mean()),
        "bird_off_diag_mean": float(bird_matrix[~np.eye(bird_matrix.shape[0], dtype=bool)].mean()),
        "recording_diag_mean": float(np.diag(recording_matrix).mean()),
        "recording_off_diag_mean": float(recording_matrix[~np.eye(recording_matrix.shape[0], dtype=bool)].mean()),
        "recording_laplacian": recording_laplacian,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Build bird/recording kNN attribution matrices and run Laplacian eigengaps.")
    parser.add_argument("species_key", choices=sorted(SPECIES))
    parser.add_argument("--run_dir", default="/media/george-vengrovski/Desk SSD/LAMBDA_TRAIN_RUNS/runs/xcl_voronoi_mask_no_normalize_32h_10w_5s_fp8")
    parser.add_argument("--checkpoint", default="model_step_499999.pth")
    parser.add_argument("--out_dir", default=str(ROOT / "results" / "individual_id_knn_graph_metrics" / "bird_knn_matrix_laplacian"))
    parser.add_argument("--songs_per_bird", type=int, default=0)
    parser.add_argument("--min_songs_per_bird", type=int, default=0)
    parser.add_argument("--max_points_per_recording", type=int, default=400)
    parser.add_argument("--k_values", default="1,2,5,10,20,50,100")
    parser.add_argument("--matrix_k", type=int, default=50)
    parser.add_argument("--num_eigenvalues", type=int, default=80)
    parser.add_argument("--knn_chunk_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset_experiment", action="store_true")
    parser.add_argument("--subset_counts", default="all")
    parser.add_argument("--subset_recordings_per_bird", default="10,20,40,80,0")
    parser.add_argument("--subset_repeats", type=int, default=3)
    parser.add_argument("--balanced_max_recordings_per_bird", action="store_true")
    parser.add_argument("--random_fraction_recordings_per_bird", action="store_true")
    parser.add_argument("--min_recording_fraction_per_bird", type=float, default=0.10)
    parser.add_argument("--graph_top_k", type=int, default=20)
    parser.add_argument("--plot_subset_experiment", action="store_true")
    parser.add_argument("--plot_subset_csv", default=None)
    parser.add_argument("--plot_out", default=None)
    parser.add_argument("--plot_title", default=None)
    parser.add_argument("--plot_prediction_key", default="eigengap_estimate", choices=["eigengap_estimate", "stable_rank"])
    parser.add_argument("--plot_x_label", default="Laplacian eigengap component estimate")
    parser.add_argument("--plot_all_species_summary", action="store_true")
    parser.add_argument("--variable_recording_repeats", type=int, default=5)
    parser.add_argument("--exclude_same_recording", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--embedding_variant", default="before", choices=["before", "after"])
    parser.add_argument("--encoder_layer_idx", type=int, default=None)
    parser.add_argument("--spec_normalization", default="auto")
    parser.add_argument("--normalization_stats_dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.species_key
    if args.subset_experiment:
        _subset_experiment(args, out_dir)
        return
    if args.plot_subset_experiment:
        _plot_subset_experiment(args, out_dir)
        return
    if args.plot_all_species_summary:
        _plot_all_species_summary(args)
        return

    selected = _selected_recordings(args)
    rows = _extract(args, selected)
    sampled = _sample(args, rows)
    neighbors, device, actual_k = _knn(args, sampled, max(max(_parse_ints(args.k_values)), args.matrix_k))
    _write_outputs(args, sampled, neighbors, device, actual_k, out_dir)


if __name__ == "__main__":
    main()
