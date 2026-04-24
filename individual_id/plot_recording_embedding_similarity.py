#!/usr/bin/env python3

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import extract_embedding  # noqa: E402


def _resolve_run_dir(run_arg):
    run_path = Path(run_arg)
    if run_path.is_absolute() and run_path.is_dir():
        return run_path
    project_relative = ROOT / run_path
    if project_relative.is_dir():
        return project_relative.resolve()
    runs_relative = ROOT / "runs" / run_path
    if runs_relative.is_dir():
        return runs_relative.resolve()
    raise SystemExit(f"Unable to resolve run_dir: {run_arg}")


def _load_recording_stems_by_bird(annotation_json):
    data = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
    by_bird = {}
    for rec in data.get("recordings", []):
        recording = rec.get("recording", {})
        bird_id = str(recording.get("bird_id", "")).strip()
        stem = Path(str(recording.get("filename", "")).strip()).stem
        if bird_id and stem:
            by_bird.setdefault(bird_id, set()).add(stem)
    return {bird_id: sorted(stems) for bird_id, stems in by_bird.items()}


def _pick_recordings(stems, songs_per_bird, seed, bird_id):
    if songs_per_bird <= 0 or len(stems) <= songs_per_bird:
        return list(stems)
    bird_hash = int(hashlib.sha1(bird_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed + bird_hash)
    indices = rng.choice(len(stems), size=songs_per_bird, replace=False)
    indices.sort()
    return [stems[index] for index in indices]


def _feature_key(variant):
    mapping = {
        "before": "encoded_embeddings_before_pos_removal",
        "after": "encoded_embeddings_after_pos_removal",
    }
    assert variant in mapping, variant
    return mapping[variant]


def _summarize(values):
    values = np.asarray(values, dtype=np.float64)
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


def _extraction_request(args, recording_stems):
    feature_key = _feature_key(args.embedding_variant)
    return {
        "run_dir": str(args.run_dir),
        "checkpoint": args.checkpoint,
        "spec_dir": str(args.spec_dir),
        "json_path": str(args.annotation_json),
        "recording_stems": recording_stems,
        "recording_mode": args.recording_mode,
        "encoder_layer_idx": args.encoder_layer_idx,
        "spec_normalization": args.spec_normalization,
        "normalization_stats_dir": args.normalization_stats_dir,
        "minimal_output": args.embedding_variant == "before",
        "embedding_postprocess": args.feature_postprocess,
        "embedding_postprocess_dim": args.feature_postprocess_dim,
        "embedding_postprocess_key": feature_key,
        "embedding_postprocess_load": args.feature_postprocess_load,
        "embedding_postprocess_save": args.feature_postprocess_save,
    }


def _extract_selected_recordings(args, model_state, selected):
    try:
        stems = [row["recording_stem"] for row in selected]
        extracted = extract_embedding.extract_recording_embeddings_with_state(_extraction_request(args, stems), model_state)
    except ValueError as exc:
        if str(exc) == "No valid patches extracted for the requested recording set.":
            return [], None
        raise

    key = _feature_key(args.embedding_variant)
    arrays_by_stem = {row["recording_stem"]: [] for row in selected}
    for segment in extracted["segments"]:
        stem = segment["recording_stem"]
        features = segment[key]
        if args.drop_silence:
            labels = segment["labels_downsampled"]
            n = min(features.shape[0], labels.shape[0])
            features = features[:n][labels[:n] >= 0]
        if features.shape[0] == 0:
            continue
        arrays_by_stem[stem].append(features.astype(np.float32, copy=False))

    rows = []
    for row in selected:
        arrays = arrays_by_stem[row["recording_stem"]]
        if not arrays:
            continue
        features = np.vstack(arrays).astype(np.float32, copy=False)
        rows.append(
            {
                "bird_id": row["bird_id"],
                "recording_stem": row["recording_stem"],
                "features": features,
                "point_count": int(features.shape[0]),
            }
        )
    return rows, extracted.get("feature_postprocess")


def _build_recording_table(args, model_state):
    stems_by_bird = _load_recording_stems_by_bird(args.annotation_json)
    bird_ids = sorted(stems_by_bird)
    if args.max_birds > 0:
        bird_ids = bird_ids[: args.max_birds]

    selected = []
    for bird_id in bird_ids:
        stems = _pick_recordings(stems_by_bird[bird_id], args.songs_per_bird, args.seed, bird_id)
        for stem in stems:
            selected.append({"bird_id": bird_id, "recording_stem": stem})
    rows, feature_postprocess = _extract_selected_recordings(args, model_state, selected)
    assert rows, "No valid recording embeddings were extracted."
    return rows, feature_postprocess


def _sample_recording_features(args, rows):
    sampled = []
    recording_indices = []
    for row_index, row in enumerate(rows):
        features = row["features"]
        if args.max_points_per_recording > 0 and features.shape[0] > args.max_points_per_recording:
            key = f"{args.seed}|{row['bird_id']}|{row['recording_stem']}"
            row_seed = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(row_seed)
            indices = rng.choice(features.shape[0], size=args.max_points_per_recording, replace=False)
            indices.sort()
            features = features[indices]
        sampled.append(features.astype(np.float32, copy=False))
        recording_indices.extend([row_index] * int(features.shape[0]))
        row["sampled_point_count"] = int(features.shape[0])
        row.pop("features")

    return (
        np.vstack(sampled).astype(np.float32, copy=False),
        np.asarray(recording_indices, dtype=np.int64),
    )


def _recording_pca_histograms(args, rows):
    assert args.pca_dim == 2, "Only 2D PCA histograms are currently supported."
    stacked, recording_indices = _sample_recording_features(args, rows)

    pca = PCA(n_components=args.pca_dim, svd_solver="randomized", random_state=args.seed)
    projected = pca.fit_transform(stacked).astype(np.float32, copy=False)

    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    padding = np.maximum((maxs - mins) * 0.02, 1e-6)
    edges = [
        np.linspace(mins[0] - padding[0], maxs[0] + padding[0], args.overlap_bins + 1),
        np.linspace(mins[1] - padding[1], maxs[1] + padding[1], args.overlap_bins + 1),
    ]

    histograms = np.zeros((len(rows), args.overlap_bins, args.overlap_bins), dtype=np.float32)
    for row_index in range(len(rows)):
        points = projected[recording_indices == row_index]
        hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=edges)
        hist = hist.astype(np.float32, copy=False)
        histograms[row_index] = hist / max(float(hist.sum()), 1.0)

    return histograms, pca, projected


def _bhattacharyya_similarity(histograms):
    roots = np.sqrt(histograms.reshape(histograms.shape[0], -1))
    return np.clip(roots @ roots.T, 0.0, 1.0).astype(np.float32, copy=False)


def _recording_mean_cosine(args, rows):
    means = []
    for row in rows:
        means.append(row["features"].mean(axis=0))
        row["sampled_point_count"] = 1
        row.pop("features")

    x = np.asarray(means, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = (x / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)
    similarity = np.clip(x @ x.T, -1.0, 1.0).astype(np.float32, copy=False)
    return similarity, {"vector_count": int(x.shape[0])}


def _random_window_means(args, rows):
    pooled = []
    recording_indices = []
    for row_index, row in enumerate(rows):
        features = row["features"]
        key = f"{args.seed}|{row['bird_id']}|{row['recording_stem']}|windows"
        row_seed = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(row_seed)

        if features.shape[0] <= args.window_mean_size:
            windows = np.repeat(features.mean(axis=0, keepdims=True), args.windows_per_recording, axis=0)
        else:
            starts = rng.integers(0, features.shape[0] - args.window_mean_size + 1, size=args.windows_per_recording)
            windows = np.asarray(
                [features[start : start + args.window_mean_size].mean(axis=0) for start in starts],
                dtype=np.float32,
            )

        pooled.append(windows.astype(np.float32, copy=False))
        recording_indices.extend([row_index] * int(windows.shape[0]))
        row["sampled_point_count"] = int(windows.shape[0])
        row.pop("features")

    return np.vstack(pooled).astype(np.float32, copy=False), np.asarray(recording_indices, dtype=np.int64)


def _recording_random_window_mean_cosine(args, rows):
    transformed, recording_indices = _random_window_means(args, rows)

    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    transformed = (transformed / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x = torch.from_numpy(transformed).to(device=device, dtype=torch.float32)
    rec = torch.from_numpy(recording_indices).to(device=device, dtype=torch.long)
    n_recordings = len(rows)
    counts = torch.bincount(rec, minlength=n_recordings).to(torch.float32)
    sums = torch.zeros((n_recordings * n_recordings,), device=device, dtype=torch.float32)

    for start in range(0, x.shape[0], int(args.cosine_chunk_size)):
        end = min(start + int(args.cosine_chunk_size), x.shape[0])
        sims = x[start:end] @ x.T
        pair_ids = (rec[start:end, None] * n_recordings + rec[None, :]).reshape(-1)
        sums.scatter_add_(0, pair_ids, sims.reshape(-1))

    similarity = sums.reshape(n_recordings, n_recordings) / torch.clamp(counts[:, None] * counts[None, :], min=1.0)
    similarity.fill_diagonal_(1.0)
    extras = {
        "device": str(device),
        "window_count": int(x.shape[0]),
    }
    return similarity.detach().cpu().numpy().astype(np.float32, copy=False), extras


def _sample_row_features(args, row):
    features = row["features"]
    if args.max_points_per_recording <= 0 or features.shape[0] <= args.max_points_per_recording:
        return features.astype(np.float32, copy=False)

    key = f"{args.seed}|{row['bird_id']}|{row['recording_stem']}|gaussian"
    row_seed = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(row_seed)
    indices = rng.choice(features.shape[0], size=args.max_points_per_recording, replace=False)
    indices.sort()
    return features[indices].astype(np.float32, copy=False)


def _mmd_split_features(args, rows):
    rows_by_bird = {}
    for row in rows:
        rows_by_bird.setdefault(row["bird_id"], []).append(row)

    split_rows = []
    feature_sets = []
    for bird_id in sorted(rows_by_bird):
        bird_rows = rows_by_bird[bird_id]
        if len(bird_rows) < 2:
            continue
        bird_hash = int(hashlib.sha1(bird_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(args.seed + bird_hash)
        order = rng.permutation(len(bird_rows))
        splits = [order[: len(order) // 2], order[len(order) // 2 :]]
        for split_index, split in enumerate(splits):
            sampled = [_sample_row_features(args, bird_rows[index]) for index in split]
            features = np.vstack(sampled).astype(np.float32, copy=False)
            if features.shape[0] > args.mmd_points_per_split:
                indices = rng.choice(features.shape[0], size=args.mmd_points_per_split, replace=False)
                indices.sort()
                features = features[indices]
            split_rows.append(
                {
                    "bird_id": bird_id,
                    "recording_stem": f"{bird_id}_split{split_index}",
                    "point_count": int(sum(bird_rows[index]["point_count"] for index in split)),
                    "sampled_point_count": int(features.shape[0]),
                }
            )
            feature_sets.append(features.astype(np.float32, copy=False))

    assert split_rows, "Need at least one individual with two recordings."
    return split_rows, feature_sets


def _mmd_bandwidth(args, feature_sets, device):
    pooled = np.vstack(feature_sets).astype(np.float32, copy=False)
    if pooled.shape[0] > args.mmd_bandwidth_points:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(pooled.shape[0], size=args.mmd_bandwidth_points, replace=False)
        pooled = pooled[indices]
    x = torch.from_numpy(pooled).to(device=device, dtype=torch.float32)
    distances = torch.pdist(x)
    median = torch.median(distances[distances > 0])
    assert torch.isfinite(median)
    return float(median.item())


def _rbf_mean(x, y, denominator):
    distances = torch.cdist(x, y).square()
    return torch.exp(-distances / denominator).mean()


def _individual_mmd_distances(args, rows):
    split_rows, feature_sets = _mmd_split_features(args, rows)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    sigma = _mmd_bandwidth(args, feature_sets, device) * float(args.mmd_sigma_scale)
    denominator = 2.0 * sigma * sigma
    tensors = [torch.from_numpy(features).to(device=device, dtype=torch.float32) for features in feature_sets]
    self_means = [_rbf_mean(x, x, denominator) for x in tensors]

    n = len(tensors)
    mmd = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            value = self_means[i] + self_means[j] - 2.0 * _rbf_mean(tensors[i], tensors[j], denominator)
            mmd[i, j] = mmd[j, i] = float(torch.clamp(value, min=0.0).item())

    extras = {
        "arrays": {},
        "summary": {
            "mmd_kernel": "rbf",
            "mmd_sigma": sigma,
            "mmd_points_per_split": int(args.mmd_points_per_split),
            "mmd_bandwidth_points": int(args.mmd_bandwidth_points),
            "mmd_feature_dim": int(feature_sets[0].shape[1]),
            "device": str(device),
        },
    }
    return split_rows, mmd, extras


def _sample_mmd_recording_features(args, row):
    features = row["features"]
    key = f"{args.seed}|{row['bird_id']}|{row['recording_stem']}|recording_mmd"
    row_seed = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(row_seed)
    replace = features.shape[0] < args.mmd_points_per_recording
    indices = rng.choice(features.shape[0], size=args.mmd_points_per_recording, replace=replace)
    indices.sort()
    return features[indices].astype(np.float32, copy=False)


def _recording_kernel_matrices(args, rows):
    feature_sets = []
    for row in rows:
        features = _sample_mmd_recording_features(args, row)
        row["sampled_point_count"] = int(features.shape[0])
        row.pop("features")
        feature_sets.append(features)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    sigma = _mmd_bandwidth(args, feature_sets, device) * float(args.mmd_sigma_scale)
    denominator = 2.0 * sigma * sigma
    x = torch.from_numpy(np.stack(feature_sets)).to(device=device, dtype=torch.float32)

    self_means = []
    for start in range(0, x.shape[0], int(args.mmd_pair_batch_size)):
        batch = x[start : start + int(args.mmd_pair_batch_size)]
        kernels = torch.exp(-torch.cdist(batch, batch).square() / denominator)
        self_means.append(kernels.mean(dim=(1, 2)))
    self_means = torch.cat(self_means)

    pair_i, pair_j = np.triu_indices(x.shape[0], k=1)
    cross = np.eye(x.shape[0], dtype=np.float32)
    mmd = np.zeros((x.shape[0], x.shape[0]), dtype=np.float32)
    for start in range(0, pair_i.size, int(args.mmd_pair_batch_size)):
        end = min(start + int(args.mmd_pair_batch_size), pair_i.size)
        i = torch.from_numpy(pair_i[start:end]).to(device=device, dtype=torch.long)
        j = torch.from_numpy(pair_j[start:end]).to(device=device, dtype=torch.long)
        cross_values = torch.exp(-torch.cdist(x[i], x[j]).square() / denominator).mean(dim=(1, 2))
        mmd_values = torch.clamp(self_means[i] + self_means[j] - 2.0 * cross_values, min=0.0)
        cross_values = cross_values.detach().cpu().numpy().astype(np.float32, copy=False)
        mmd_values = mmd_values.detach().cpu().numpy().astype(np.float32, copy=False)
        cross[pair_i[start:end], pair_j[start:end]] = cross_values
        cross[pair_j[start:end], pair_i[start:end]] = cross_values
        mmd[pair_i[start:end], pair_j[start:end]] = mmd_values
        mmd[pair_j[start:end], pair_i[start:end]] = mmd_values

    extras = {
        "arrays": {
            "recording_kernel_cross_mean": cross,
            "recording_mmd_rbf": mmd,
        },
        "summary": {
            "mmd_kernel": "rbf",
            "mmd_sigma": sigma,
            "mmd_sigma_scale": float(args.mmd_sigma_scale),
            "mmd_points_per_recording": int(args.mmd_points_per_recording),
            "mmd_bandwidth_points": int(args.mmd_bandwidth_points),
            "mmd_pair_batch_size": int(args.mmd_pair_batch_size),
            "mmd_feature_dim": int(x.shape[2]),
            "device": str(device),
        },
    }
    self = self_means.detach().cpu().numpy().astype(np.float32, copy=False)
    return cross, self, mmd, extras


def _recording_mmd_distances(args, rows):
    _, _, mmd, extras = _recording_kernel_matrices(args, rows)
    return mmd, extras


def _recording_kernel_overlap(args, rows):
    cross, self, mmd, extras = _recording_kernel_matrices(args, rows)
    normalization = np.sqrt(np.maximum(self[:, None] * self[None, :], 1e-12))
    overlap = np.clip(cross / normalization, 0.0, 1.0).astype(np.float32, copy=False)
    np.fill_diagonal(overlap, 1.0)
    extras["arrays"]["recording_mmd_rbf"] = mmd
    extras["summary"]["kernel_overlap_normalization"] = "cosine"
    return overlap, extras


def _fit_gaussian(features, regularization):
    x = features.astype(np.float64, copy=False)
    mean = x.mean(axis=0)
    centered = x - mean
    covariance = centered.T @ centered / max(x.shape[0] - 1, 1)
    covariance.flat[:: covariance.shape[0] + 1] += regularization
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0
    return mean, covariance, float(logdet)


def _individual_gaussian_distances(args, rows):
    rows_by_bird = {}
    for row in rows:
        rows_by_bird.setdefault(row["bird_id"], []).append(row)

    split_rows = []
    gaussians = []
    for bird_id in sorted(rows_by_bird):
        bird_rows = rows_by_bird[bird_id]
        if len(bird_rows) < 2:
            continue
        bird_hash = int(hashlib.sha1(bird_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(args.seed + bird_hash)
        order = rng.permutation(len(bird_rows))
        splits = [order[: len(order) // 2], order[len(order) // 2 :]]
        for split_index, split in enumerate(splits):
            sampled = [_sample_row_features(args, bird_rows[index]) for index in split]
            features = np.vstack(sampled).astype(np.float32, copy=False)
            split_rows.append(
                {
                    "bird_id": bird_id,
                    "recording_stem": f"{bird_id}_split{split_index}",
                    "point_count": int(sum(bird_rows[index]["point_count"] for index in split)),
                    "sampled_point_count": int(features.shape[0]),
                }
            )
            gaussians.append(_fit_gaussian(features, args.gaussian_regularization))

    assert split_rows, "Need at least one individual with two recordings."
    n = len(split_rows)
    bhattacharyya = np.zeros((n, n), dtype=np.float32)
    symmetric_kl = np.zeros((n, n), dtype=np.float32)
    dim = int(gaussians[0][0].shape[0])

    for i in range(n):
        mean_i, cov_i, logdet_i = gaussians[i]
        for j in range(i + 1, n):
            mean_j, cov_j, logdet_j = gaussians[j]
            diff = mean_j - mean_i
            avg_cov = (cov_i + cov_j) * 0.5
            sign, avg_logdet = np.linalg.slogdet(avg_cov)
            assert sign > 0

            bhatta = 0.125 * diff.dot(np.linalg.solve(avg_cov, diff))
            bhatta += 0.5 * (avg_logdet - 0.5 * (logdet_i + logdet_j))

            solve_ij = np.linalg.solve(cov_j, cov_i)
            solve_ji = np.linalg.solve(cov_i, cov_j)
            kl_ij = np.trace(solve_ij) + diff.dot(np.linalg.solve(cov_j, diff)) - dim + logdet_j - logdet_i
            kl_ji = np.trace(solve_ji) + diff.dot(np.linalg.solve(cov_i, diff)) - dim + logdet_i - logdet_j

            bhattacharyya[i, j] = bhattacharyya[j, i] = float(bhatta)
            symmetric_kl[i, j] = symmetric_kl[j, i] = float(0.25 * (kl_ij + kl_ji))

    extras = {
        "arrays": {
            "gaussian_symmetric_kl": symmetric_kl,
        },
        "summary": {
            "gaussian_covariance": "full",
            "gaussian_regularization": float(args.gaussian_regularization),
            "gaussian_feature_dim": dim,
        },
    }
    return split_rows, bhattacharyya, extras


def _recording_knn_similarity(args, rows):
    transformed, recording_indices = _sample_recording_features(args, rows)

    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    transformed = (transformed / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x = torch.from_numpy(transformed).to(device=device, dtype=torch.float32)
    recording_ids = torch.from_numpy(recording_indices).to(device=device, dtype=torch.long)
    n_recordings = len(rows)
    counts = torch.bincount(recording_ids, minlength=n_recordings).to(torch.float32)
    overlap = torch.zeros((n_recordings, n_recordings), device=device, dtype=torch.float32)

    k = min(int(args.knn_k), int(x.shape[0]) - 1)
    all_indices = torch.arange(x.shape[0], device=device)
    for start in range(0, x.shape[0], int(args.knn_chunk_size)):
        end = min(start + int(args.knn_chunk_size), x.shape[0])
        sims = x[start:end] @ x.T
        sims[torch.arange(end - start, device=device), all_indices[start:end]] = -float("inf")
        neighbors = torch.topk(sims, k=k, dim=1).indices
        source_ids = recording_ids[start:end, None].expand_as(neighbors).reshape(-1)
        target_ids = recording_ids[neighbors].reshape(-1)
        overlap.index_put_((source_ids, target_ids), torch.ones_like(source_ids, dtype=torch.float32), accumulate=True)

    overlap = overlap / torch.clamp(counts[:, None] * float(k), min=1.0)
    similarity = (overlap + overlap.T) / 2.0
    similarity.fill_diagonal_(1.0)
    extras = {
        "device": str(device),
        "knn_k": int(k),
    }
    return similarity.detach().cpu().numpy().astype(np.float32, copy=False), extras


def _recording_neighbor_enrichment(args, rows):
    transformed, recording_indices = _sample_recording_features(args, rows)

    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    transformed = (transformed / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)

    bird_ids = np.asarray([row["bird_id"] for row in rows], dtype=object)
    _, bird_codes = np.unique(bird_ids, return_inverse=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x = torch.from_numpy(transformed).to(device=device, dtype=torch.float32)
    point_recordings = torch.from_numpy(recording_indices).to(device=device, dtype=torch.long)
    recording_birds = torch.from_numpy(bird_codes.astype(np.int64)).to(device=device, dtype=torch.long)
    point_birds = recording_birds[point_recordings]

    n_recordings = len(rows)
    counts = torch.bincount(point_recordings, minlength=n_recordings).to(torch.float32)
    total_points = int(x.shape[0])
    k = min(int(args.knn_k), total_points - int(counts.max().item()))
    assert k > 0, "Need at least two recordings with sampled embedding vectors."

    bird_counts = torch.bincount(point_birds, minlength=int(recording_birds.max().item()) + 1).to(torch.float32)
    same_candidates = bird_counts[recording_birds] - counts
    total_candidates = float(total_points) - counts
    recording_null = same_candidates / torch.clamp(total_candidates, min=1.0)

    overlap = torch.zeros((n_recordings, n_recordings), device=device, dtype=torch.float32)
    query_observed = torch.empty(total_points, device=device, dtype=torch.float32)
    query_null = torch.empty(total_points, device=device, dtype=torch.float32)

    for start in range(0, total_points, int(args.knn_chunk_size)):
        end = min(start + int(args.knn_chunk_size), total_points)
        source_recordings = point_recordings[start:end]
        sims = x[start:end] @ x.T
        sims[source_recordings[:, None] == point_recordings[None, :]] = -float("inf")
        neighbors = torch.topk(sims, k=k, dim=1).indices

        source_birds = recording_birds[source_recordings]
        target_recordings = point_recordings[neighbors]
        target_birds = recording_birds[target_recordings]
        query_observed[start:end] = (target_birds == source_birds[:, None]).to(torch.float32).mean(dim=1)
        query_null[start:end] = recording_null[source_recordings]

        source_ids = source_recordings[:, None].expand_as(target_recordings).reshape(-1)
        target_ids = target_recordings.reshape(-1)
        overlap.index_put_((source_ids, target_ids), torch.ones_like(source_ids, dtype=torch.float32), accumulate=True)

    overlap = overlap / torch.clamp(counts[:, None] * float(k), min=1.0)
    similarity = (overlap + overlap.T) / 2.0
    similarity.fill_diagonal_(0.0)

    per_recording_observed = torch.zeros(n_recordings, device=device, dtype=torch.float32)
    per_recording_observed.scatter_add_(0, point_recordings, query_observed)
    per_recording_observed = per_recording_observed / torch.clamp(counts, min=1.0)

    extras = {
        "device": str(device),
        "knn_k": int(k),
        "query_observed": query_observed.detach().cpu().numpy().astype(np.float32, copy=False),
        "query_null": query_null.detach().cpu().numpy().astype(np.float32, copy=False),
        "query_recording_indices": recording_indices,
        "recording_observed": per_recording_observed.detach().cpu().numpy().astype(np.float32, copy=False),
        "recording_null": recording_null.detach().cpu().numpy().astype(np.float32, copy=False),
    }
    return similarity.detach().cpu().numpy().astype(np.float32, copy=False), extras


def _recording_two_afc(args, rows):
    transformed, recording_indices = _sample_recording_features(args, rows)

    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    transformed = (transformed / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)

    bird_ids = np.asarray([row["bird_id"] for row in rows], dtype=object)
    _, bird_codes = np.unique(bird_ids, return_inverse=True)
    point_birds = bird_codes[recording_indices]
    total_points = int(transformed.shape[0])
    trials = total_points * int(args.afc_trials_per_query)

    rng = np.random.default_rng(args.seed)
    query_indices = np.repeat(np.arange(total_points, dtype=np.int64), int(args.afc_trials_per_query))
    query_recordings = recording_indices[query_indices]
    same_indices = np.empty(trials, dtype=np.int64)
    different_indices = np.empty(trials, dtype=np.int64)

    for recording_index, bird_code in enumerate(bird_codes):
        trial_slots = np.flatnonzero(query_recordings == recording_index)
        if trial_slots.size == 0:
            continue
        same_pool = np.flatnonzero((point_birds == bird_code) & (recording_indices != recording_index))
        different_pool = np.flatnonzero(point_birds != bird_code)
        assert same_pool.size > 0, f"No same-individual candidates outside recording {recording_index}."
        assert different_pool.size > 0, f"No different-individual candidates for recording {recording_index}."
        same_indices[trial_slots] = rng.choice(same_pool, size=trial_slots.size, replace=True)
        different_indices[trial_slots] = rng.choice(different_pool, size=trial_slots.size, replace=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x = torch.from_numpy(transformed).to(device=device, dtype=torch.float32)
    q = torch.from_numpy(query_indices).to(device=device, dtype=torch.long)
    same = torch.from_numpy(same_indices).to(device=device, dtype=torch.long)
    different = torch.from_numpy(different_indices).to(device=device, dtype=torch.long)
    point_recordings = torch.from_numpy(recording_indices).to(device=device, dtype=torch.long)

    n_recordings = len(rows)
    win_matrix = torch.zeros((n_recordings, n_recordings), device=device, dtype=torch.float32)
    count_matrix = torch.zeros((n_recordings, n_recordings), device=device, dtype=torch.float32)
    query_wins = torch.zeros(total_points, device=device, dtype=torch.float32)
    query_counts = torch.zeros(total_points, device=device, dtype=torch.float32)

    for start in range(0, trials, int(args.afc_chunk_size)):
        end = min(start + int(args.afc_chunk_size), trials)
        query = q[start:end]
        same_candidate = same[start:end]
        different_candidate = different[start:end]

        query_vectors = x[query]
        same_similarity = (query_vectors * x[same_candidate]).sum(dim=1)
        different_similarity = (query_vectors * x[different_candidate]).sum(dim=1)
        same_wins = (same_similarity > different_similarity).to(torch.float32)

        source_recordings = point_recordings[query]
        same_recordings = point_recordings[same_candidate]
        different_recordings = point_recordings[different_candidate]
        ones = torch.ones_like(same_wins)

        win_matrix.index_put_((source_recordings, same_recordings), same_wins, accumulate=True)
        count_matrix.index_put_((source_recordings, same_recordings), ones, accumulate=True)
        win_matrix.index_put_((source_recordings, different_recordings), 1.0 - same_wins, accumulate=True)
        count_matrix.index_put_((source_recordings, different_recordings), ones, accumulate=True)
        query_wins.index_put_((query,), same_wins, accumulate=True)
        query_counts.index_put_((query,), ones, accumulate=True)

    pair_wins = win_matrix + win_matrix.T
    pair_counts = count_matrix + count_matrix.T
    similarity = pair_wins / torch.clamp(pair_counts, min=1.0)
    similarity[pair_counts == 0] = 0.0
    similarity.fill_diagonal_(0.0)

    query_accuracy = query_wins / torch.clamp(query_counts, min=1.0)
    recording_counts = torch.bincount(point_recordings, minlength=n_recordings).to(torch.float32)
    recording_accuracy = torch.zeros(n_recordings, device=device, dtype=torch.float32)
    recording_accuracy.scatter_add_(0, point_recordings, query_accuracy)
    recording_accuracy = recording_accuracy / torch.clamp(recording_counts, min=1.0)

    extras = {
        "device": str(device),
        "afc_trials_per_query": int(args.afc_trials_per_query),
        "query_accuracy": query_accuracy.detach().cpu().numpy().astype(np.float32, copy=False),
        "recording_accuracy": recording_accuracy.detach().cpu().numpy().astype(np.float32, copy=False),
    }
    return similarity.detach().cpu().numpy().astype(np.float32, copy=False), extras


def _bird_spans(bird_ids):
    spans = []
    start = 0
    while start < len(bird_ids):
        end = start + 1
        while end < len(bird_ids) and bird_ids[end] == bird_ids[start]:
            end += 1
        spans.append((bird_ids[start], start, end))
        start = end
    return spans


def _pair_scores(similarity, bird_ids):
    upper = np.triu_indices(similarity.shape[0], k=1)
    scores = similarity[upper]
    same = bird_ids[upper[0]] == bird_ids[upper[1]]
    return scores[same], scores[~same]


def _similarity_label(args):
    if args.similarity_mode == "pca_histogram_bhattacharyya":
        return "Bhattacharyya coefficient"
    if args.similarity_mode == "recording_mean_cosine":
        return "Whole-recording mean cosine similarity"
    if args.similarity_mode == "random_window_mean_cosine":
        return "Mean random-window cosine similarity"
    if args.similarity_mode == "individual_gaussian_bhattacharyya":
        return "Individual split Gaussian Bhattacharyya distance"
    if args.similarity_mode == "individual_mmd_rbf":
        return "Individual split RBF MMD^2"
    if args.similarity_mode == "recording_mmd_rbf":
        return "Recording RBF MMD^2"
    if args.similarity_mode == "recording_kernel_overlap":
        return "Recording normalized RBF kernel overlap"
    if args.similarity_mode == "knn_overlap":
        return "Local kNN overlap"
    if args.similarity_mode == "neighbor_enrichment":
        return "Neighbor retrieval probability"
    assert args.similarity_mode == "two_afc"
    return "2AFC target win probability"


def _save_heatmap(similarity, bird_ids, title, out_base, label, vmin, vmax):
    spans = _bird_spans(bird_ids)
    fig_size = max(7.0, min(18.0, 0.22 * similarity.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=300)
    image = ax.imshow(similarity, cmap="viridis", vmin=vmin, vmax=vmax)

    for _, start, end in spans:
        ax.axhline(start - 0.5, color="black", linewidth=0.6)
        ax.axvline(start - 0.5, color="black", linewidth=0.6)
        ax.axhline(end - 0.5, color="black", linewidth=0.6)
        ax.axvline(end - 0.5, color="black", linewidth=0.6)

    centers = [(start + end - 1) / 2 for _, start, end in spans]
    labels = [bird_id for bird_id, _, _ in spans]
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("Recording grouped by individual")
    ax.set_ylabel("Recording grouped by individual")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=label)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _save_distribution_plot(within, between, title, out_base, label, xmin, xmax):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bins = np.linspace(xmin, xmax, 80)
    if between.size:
        ax.hist(between, bins=bins, alpha=0.65, density=True, label="Between individuals")
        ax.axvline(float(between.mean()), color="tab:blue", linestyle="--", linewidth=1.0)
    if within.size:
        ax.hist(within, bins=bins, alpha=0.65, density=True, label="Within individuals")
        ax.axvline(float(within.mean()), color="tab:orange", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(f"Recording {label.lower()}")
    ax.set_ylabel("Density")
    ax.set_xlim(xmin, xmax)
    if within.size or between.size:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _save_observed_null_plot(observed, null, title, out_base, ylabel):
    fig, ax = plt.subplots(figsize=(4.2, 5), dpi=300)
    means = [float(observed.mean()), float(null.mean())]
    ax.bar([0, 1], means, color=["tab:orange", "0.65"], width=0.55)
    for index, values in enumerate([observed, null]):
        x = np.full(values.shape, index, dtype=np.float32)
        ax.scatter(x, values, s=10, alpha=0.55, color="black", linewidths=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Observed", "Null"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.0, max(float(observed.max()), float(null.max()), 0.01) * 1.15)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _feature_postprocess_summary(feature_postprocess):
    if feature_postprocess is None:
        return {
            "mode": "none",
            "dim": None,
            "feature_key": None,
            "load_path": None,
            "save_path": None,
        }
    return {
        "mode": feature_postprocess["mode"],
        "dim": int(feature_postprocess["dim"]),
        "feature_key": feature_postprocess["feature_key"],
        "load_path": feature_postprocess["load_path"],
        "save_path": feature_postprocess["save_path"],
    }


def _write_outputs(args, rows, similarity, within, between, feature_postprocess, extras):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bird_ids = np.asarray([row["bird_id"] for row in rows], dtype=object)
    recording_stems = np.asarray([row["recording_stem"] for row in rows], dtype=object)
    point_counts = np.asarray([row["point_count"] for row in rows], dtype=np.int64)
    title = f"{args.species} | unpooled embedding overlap"
    label = _similarity_label(args)
    upper = similarity[np.triu_indices(similarity.shape[0], k=1)]
    plot_min = 0.0
    plot_max = 1.0
    if args.similarity_mode in {
        "recording_mean_cosine",
        "random_window_mean_cosine",
        "individual_gaussian_bhattacharyya",
        "individual_mmd_rbf",
        "recording_mmd_rbf",
    }:
        plot_min = float(np.percentile(upper, 0.5))
        plot_max = float(np.percentile(upper, 99.5))
    if args.similarity_mode in {"knn_overlap", "neighbor_enrichment"}:
        plot_max = max(float(np.percentile(upper, 99.5)), 1e-6)
    if plot_max <= plot_min:
        plot_max = plot_min + 1e-6

    _save_heatmap(similarity, bird_ids, title, out_dir / "recording_similarity_heatmap", label, plot_min, plot_max)
    _save_distribution_plot(within, between, title, out_dir / "recording_similarity_distributions", label, plot_min, plot_max)
    if "recording_observed" in extras["arrays"]:
        _save_observed_null_plot(
            extras["arrays"]["recording_observed"],
            extras["arrays"]["recording_null"],
            title,
            out_dir / "same_individual_neighbor_enrichment",
            "Same-individual neighbor fraction",
        )
    if "recording_two_afc_accuracy" in extras["arrays"]:
        chance = np.full_like(extras["arrays"]["recording_two_afc_accuracy"], 0.5)
        _save_observed_null_plot(
            extras["arrays"]["recording_two_afc_accuracy"],
            chance,
            title,
            out_dir / "two_alternative_forced_choice",
            "2AFC accuracy",
        )

    np.savez(
        out_dir / "recording_similarity.npz",
        similarity=similarity,
        bird_ids=bird_ids,
        recording_stems=recording_stems,
        point_counts=point_counts,
        sampled_point_counts=np.asarray([row["sampled_point_count"] for row in rows], dtype=np.int64),
        within_scores=within,
        between_scores=between,
        feature_postprocess_mode=np.array(args.feature_postprocess),
        feature_postprocess_dim=np.array(0 if feature_postprocess is None else int(feature_postprocess["dim"])),
        **extras["arrays"],
    )

    summary = {
        "species": args.species,
        "run_dir": str(args.run_dir),
        "checkpoint": args.checkpoint,
        "recording_mode": args.recording_mode,
        "embedding_variant": args.embedding_variant,
        "feature_postprocess": _feature_postprocess_summary(feature_postprocess),
        "similarity_mode": args.similarity_mode,
        **extras["summary"],
        "plot_similarity_vmin": float(plot_min),
        "plot_similarity_vmax": float(plot_max),
        "max_points_per_recording": int(args.max_points_per_recording),
        "drop_silence": bool(args.drop_silence),
        "recordings": int(len(rows)),
        "individuals": int(len(set(bird_ids.tolist()))),
        "within_individual_similarity": _summarize(within),
        "between_individual_similarity": _summarize(between),
        **extras.get("neighbor_enrichment_summary", {}),
        "per_recording_point_count": _summarize(point_counts),
        "recordings_table": [
            {
                "bird_id": row["bird_id"],
                "recording_stem": row["recording_stem"],
                "point_count": int(row["point_count"]),
                "sampled_point_count": int(row["sampled_point_count"]),
            }
            for row in rows
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Plot recording-level embedding similarity grouped by individual.")
    parser.add_argument("--species", required=True)
    parser.add_argument("--annotation_json", required=True)
    parser.add_argument("--spec_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out_dir", default=str(ROOT / "results" / "individual_id_recording_similarity"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--recording_mode", default="events", choices=["events", "full_recordings"])
    parser.add_argument("--songs_per_bird", type=int, default=30)
    parser.add_argument("--max_birds", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_points_per_recording", type=int, default=200)
    parser.add_argument(
        "--similarity_mode",
        default="pca_histogram_bhattacharyya",
        choices=[
            "pca_histogram_bhattacharyya",
            "recording_mean_cosine",
            "random_window_mean_cosine",
            "individual_gaussian_bhattacharyya",
            "individual_mmd_rbf",
            "recording_mmd_rbf",
            "recording_kernel_overlap",
            "knn_overlap",
            "neighbor_enrichment",
            "two_afc",
        ],
    )
    parser.add_argument("--pca_dim", type=int, default=2)
    parser.add_argument("--overlap_bins", type=int, default=100)
    parser.add_argument("--window_mean_size", type=int, default=30)
    parser.add_argument("--windows_per_recording", type=int, default=30)
    parser.add_argument("--cosine_chunk_size", type=int, default=1024)
    parser.add_argument("--knn_k", type=int, default=100)
    parser.add_argument("--knn_chunk_size", type=int, default=512)
    parser.add_argument("--afc_trials_per_query", type=int, default=20)
    parser.add_argument("--afc_chunk_size", type=int, default=262144)
    parser.add_argument("--gaussian_regularization", type=float, default=1e-3)
    parser.add_argument("--mmd_points_per_split", type=int, default=512)
    parser.add_argument("--mmd_points_per_recording", type=int, default=128)
    parser.add_argument("--mmd_bandwidth_points", type=int, default=4096)
    parser.add_argument("--mmd_pair_batch_size", type=int, default=512)
    parser.add_argument("--mmd_sigma_scale", type=float, default=1.0)
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

    args.annotation_json = str(Path(args.annotation_json).resolve())
    args.spec_dir = str(Path(args.spec_dir).resolve())
    args.run_dir = str(_resolve_run_dir(args.run_dir))
    args.out_dir = str(Path(args.out_dir).resolve())
    assert args.window_mean_size > 0
    assert args.windows_per_recording > 0
    assert args.cosine_chunk_size > 0
    assert args.mmd_points_per_split > 0
    assert args.mmd_points_per_recording > 0
    assert args.mmd_bandwidth_points > 0
    assert args.mmd_pair_batch_size > 0
    assert args.mmd_sigma_scale > 0.0
    if args.feature_postprocess_load is not None:
        args.feature_postprocess_load = str(Path(args.feature_postprocess_load).resolve())
    if args.feature_postprocess_save is not None:
        save_path = Path(args.feature_postprocess_save).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        args.feature_postprocess_save = str(save_path)

    model_state = extract_embedding.load_model_state({"run_dir": args.run_dir, "checkpoint": args.checkpoint})
    if args.spec_normalization == "auto":
        args.spec_normalization, args.normalization_stats_dir = extract_embedding.get_native_input_normalization(model_state)

    rows, feature_postprocess = _build_recording_table(args, model_state)
    if args.similarity_mode == "pca_histogram_bhattacharyya":
        histograms, pca, projected = _recording_pca_histograms(args, rows)
        similarity = _bhattacharyya_similarity(histograms)
        extras = {
            "arrays": {
                "histograms": histograms,
                "projected": projected,
                "pca_components": pca.components_.astype(np.float32, copy=False),
                "pca_explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32, copy=False),
            },
            "summary": {
                "pca_dim": int(args.pca_dim),
                "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
                "overlap_bins": int(args.overlap_bins),
            },
        }
    elif args.similarity_mode == "recording_mean_cosine":
        similarity, mean_summary = _recording_mean_cosine(args, rows)
        extras = {
            "arrays": {},
            "summary": {
                "vector_count": mean_summary["vector_count"],
            },
        }
    elif args.similarity_mode == "random_window_mean_cosine":
        similarity, window_summary = _recording_random_window_mean_cosine(args, rows)
        extras = {
            "arrays": {},
            "summary": {
                "window_mean_size": int(args.window_mean_size),
                "windows_per_recording": int(args.windows_per_recording),
                "window_count": window_summary["window_count"],
                "cosine_chunk_size": int(args.cosine_chunk_size),
                "device": window_summary["device"],
            },
        }
    elif args.similarity_mode == "individual_gaussian_bhattacharyya":
        rows, similarity, extras = _individual_gaussian_distances(args, rows)
    elif args.similarity_mode == "individual_mmd_rbf":
        rows, similarity, extras = _individual_mmd_distances(args, rows)
    elif args.similarity_mode == "recording_mmd_rbf":
        similarity, extras = _recording_mmd_distances(args, rows)
    elif args.similarity_mode == "recording_kernel_overlap":
        similarity, extras = _recording_kernel_overlap(args, rows)
    elif args.similarity_mode == "knn_overlap":
        similarity, knn_summary = _recording_knn_similarity(args, rows)
        extras = {
            "arrays": {},
            "summary": {
                "knn_k": knn_summary["knn_k"],
                "knn_chunk_size": int(args.knn_chunk_size),
                "device": knn_summary["device"],
            },
        }
    elif args.similarity_mode == "neighbor_enrichment":
        similarity, enrichment = _recording_neighbor_enrichment(args, rows)
        observed = enrichment["query_observed"]
        null = enrichment["query_null"]
        extras = {
            "arrays": {
                "query_observed_same_individual_fraction": observed,
                "query_null_same_individual_fraction": null,
                "query_recording_indices": enrichment["query_recording_indices"],
                "recording_observed": enrichment["recording_observed"],
                "recording_null": enrichment["recording_null"],
            },
            "summary": {
                "knn_k": enrichment["knn_k"],
                "knn_chunk_size": int(args.knn_chunk_size),
                "device": enrichment["device"],
            },
            "neighbor_enrichment_summary": {
                "query_observed_same_individual_fraction": _summarize(observed),
                "query_null_same_individual_fraction": _summarize(null),
                "query_observed_minus_null_mean": float(observed.mean() - null.mean()),
                "query_observed_over_null_mean": float(observed.mean() / max(float(null.mean()), 1e-12)),
                "recording_observed_same_individual_fraction": _summarize(enrichment["recording_observed"]),
                "recording_null_same_individual_fraction": _summarize(enrichment["recording_null"]),
            },
        }
    else:
        assert args.similarity_mode == "two_afc"
        similarity, afc = _recording_two_afc(args, rows)
        query_accuracy = afc["query_accuracy"]
        recording_accuracy = afc["recording_accuracy"]
        extras = {
            "arrays": {
                "query_two_afc_accuracy": query_accuracy,
                "recording_two_afc_accuracy": recording_accuracy,
            },
            "summary": {
                "afc_trials_per_query": afc["afc_trials_per_query"],
                "afc_chunk_size": int(args.afc_chunk_size),
                "device": afc["device"],
            },
            "neighbor_enrichment_summary": {
                "query_two_afc_accuracy": _summarize(query_accuracy),
                "recording_two_afc_accuracy": _summarize(recording_accuracy),
                "query_two_afc_accuracy_minus_chance": float(query_accuracy.mean() - 0.5),
            },
        }

    bird_ids = np.asarray([row["bird_id"] for row in rows], dtype=object)
    within, between = _pair_scores(similarity, bird_ids)
    _write_outputs(args, rows, similarity, within, between, feature_postprocess, extras)

    print(
        "[recording-similarity] "
        f"recordings={len(rows)} individuals={len(set(bird_ids.tolist()))} "
        f"within_mean={_summarize(within)['mean']} between_mean={_summarize(between)['mean']} "
        f"out_dir={args.out_dir}"
    )


if __name__ == "__main__":
    main()
