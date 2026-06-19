#!/usr/bin/env python3

import argparse
import colorsys
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
import umap.umap_ as umap_
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import completeness_score, homogeneity_score, silhouette_score, v_measure_score
from sklearn.utils import check_random_state

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.external_models import aves  # noqa: E402
from src.external_models import bird_mae  # noqa: E402
from src.core import extract_embedding  # noqa: E402
from src.external_models import hubert  # noqa: E402
try:
    from src.external_models import old_perch as perch  # noqa: E402
except ImportError:
    from src.external_models import perch2 as perch  # noqa: E402


def _default_recording_svd(path):
    return {
        "recording_svd_npz": path,
        "recording_feature_mode": "svd",
        "recording_feature_scope": "full",
        "recording_svd_dim": 15,
        "recording_svd_alpha": 1.0,
        "recording_svd_append": "post",
    }


SPECIES_CONFIGS = {
    "zf": {
        "aliases": ("zebra_finch",),
        "display_name": "Zebra Finch",
        "pool_window": 10,
        "pool_hop": 2,
        "recording_mode": "events",
        "songs_per_bird": 0,
        "feature_postprocess": "pca_whiten_l2",
        "feature_postprocess_dim": 1024,
        **_default_recording_svd("results/individual_id_knn_graph_metrics/bird_knn_matrix_maxdata_uncapped/zf/knn_attribution_matrices.npz"),
    },
    "bf": {
        "aliases": ("bengalese_finch",),
        "display_name": "Bengalese Finch",
        "pool_window": 10,
        "pool_hop": 2,
        "recording_mode": "events",
        "songs_per_bird": 200,
        "feature_postprocess": "pca_whiten_l2",
        "feature_postprocess_dim": 1024,
        **_default_recording_svd("results/individual_id_knn_graph_metrics/bird_knn_matrix_perbird200_uncapped/bf/knn_attribution_matrices.npz"),
    },
    "canary": {
        "aliases": (),
        "display_name": "Canary",
        "pool_window": 30,
        "pool_hop": 5,
        "recording_mode": "events",
        "songs_per_bird": 0,
        "feature_postprocess": "pca_whiten_l2",
        "feature_postprocess_dim": 1024,
        **_default_recording_svd("results/individual_id_knn_graph_metrics/bird_knn_matrix_maxdata_uncapped/canary/knn_attribution_matrices.npz"),
    },
    "chiffchaff": {
        "aliases": (),
        "display_name": "Chiffchaff",
        "pool_window": 30,
        "pool_hop": 5,
        "recording_mode": "events",
        "songs_per_bird": 88,
        "feature_postprocess": "pca_whiten_l2",
        "feature_postprocess_dim": 1024,
        **_default_recording_svd("results/individual_id_knn_graph_metrics/bird_knn_matrix_recordings2000_uncapped/chiffchaff/knn_attribution_matrices.npz"),
    },
    "european_starling": {
        "aliases": ("starling",),
        "display_name": "European Starling",
        "pool_window": 30,
        "pool_hop": 5,
        "recording_mode": "events",
        "songs_per_bird": 30,
        "feature_postprocess": "pca_whiten_l2",
        "feature_postprocess_dim": 1024,
        **_default_recording_svd("results/individual_id_knn_graph_metrics/bird_knn_matrix_laplacian_cap30/european_starling/knn_attribution_matrices.npz"),
    },
    "tree_pipit": {
        "aliases": (),
        "display_name": "Tree Pipit",
        "pool_window": 10,
        "pool_hop": 2,
        "recording_mode": "events",
        "songs_per_bird": 0,
        "feature_postprocess": "pca_whiten_l2",
        "feature_postprocess_dim": 1024,
        **_default_recording_svd("results/individual_id_knn_graph_metrics/bird_knn_matrix_perbird400_uncapped/tree_pipit/knn_attribution_matrices.npz"),
    },
    "little_owl": {
        "aliases": (),
        "display_name": "Little Owl",
        "pool_window": 5,
        "pool_hop": 2,
        "recording_mode": "events",
        "songs_per_bird": 0,
        "feature_postprocess": "pca_whiten_l2",
        "feature_postprocess_dim": 1024,
        **_default_recording_svd("results/individual_id_knn_graph_metrics/bird_knn_matrix_maxdata_uncapped/little_owl/knn_attribution_matrices.npz"),
    },
    "orangutan": {
        "aliases": (),
        "display_name": "Orangutan",
        "pool_window": 250,
        "pool_hop": 50,
        "recording_mode": "full_recordings",
        "songs_per_bird": 0,
        "feature_postprocess": "pca_whiten_l2",
        "feature_postprocess_dim": 1024,
    },
    "ovenbird": {
        "aliases": ("lapp_ovenbird",),
        "display_name": "Ovenbird",
        "pool_window": 5,
        "pool_hop": 2,
        "recording_mode": "events",
        "songs_per_bird": 0,
        "feature_postprocess": "pca_whiten_l2",
        "feature_postprocess_dim": 1024,
        **_default_recording_svd("results/individual_id_knn_graph_metrics/bird_knn_matrix_maxdata_uncapped/ovenbird/knn_attribution_matrices.npz"),
    },
}


def _species_key(species):
    return str(species).strip().lower().replace(" ", "_").replace("-", "_")


def _species_config(species):
    key = _species_key(species)
    for species_key, config in SPECIES_CONFIGS.items():
        if key == species_key or key in config["aliases"]:
            return species_key, config

    return key, {
        "aliases": (),
        "display_name": str(species).strip().replace("_", " ").title(),
        "pool_window": 30,
        "pool_hop": 5,
        "recording_mode": "events",
        "songs_per_bird": 30,
        "feature_postprocess": "pca_whiten_l2",
        "feature_postprocess_dim": 1024,
    }


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
        if not bird_id or not stem:
            continue
        by_bird.setdefault(bird_id, set()).add(stem)
    return {bird_id: sorted(stems) for bird_id, stems in by_bird.items()}


def _load_target_stats(stats_dir):
    audio = extract_embedding.load_audio_params(stats_dir)
    return np.float32(audio["mean"]), np.float32(audio["std"])


def _pick_recordings(stems, songs_per_bird, seed, bird_id):
    if songs_per_bird <= 0:
        return list(stems)
    if len(stems) < songs_per_bird:
        return []
    bird_hash = int(hashlib.sha1(bird_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed + bird_hash)
    indices = rng.choice(len(stems), size=songs_per_bird, replace=False)
    indices.sort()
    return [stems[index] for index in indices]


def _segment_songmae_features(segment, feature_key):
    features = segment.get(feature_key)
    if features is None:
        features = segment["encoded_embeddings"]
    return features


def _songmae_feature_key(feature_source):


    # encoded_before should be default it gives the best features 
    mapping = {
        "encoded_before": "encoded_embeddings_before_pos_removal",
        "encoded_after": "encoded_embeddings_after_pos_removal",
        "patch_pre_pos": "patch_embeddings_before_pos_encoding",
        "patch_before": "patch_embeddings_before_pos_removal",
        "patch_after": "patch_embeddings_after_pos_removal",
    }
    assert feature_source in mapping, feature_source
    return mapping[feature_source]


def _stable_seed(seed, *parts):
    key = "|".join([str(seed), *[str(part) for part in parts]])
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)


def _pool_starts(length, window, hop, layout, seed):
    if length < window:
        return np.zeros((0,), dtype=np.int64)

    starts = np.arange(0, length - window + 1, hop, dtype=np.int64)
    if layout == "sliding" or starts.size <= 1:
        return starts

    assert layout == "shotgun"
    max_start = length - window + 1
    if starts.size >= max_start:
        return np.arange(0, max_start, dtype=np.int64)

    rng = np.random.default_rng(seed)
    sampled = rng.choice(max_start, size=int(starts.size), replace=False)
    sampled.sort()
    return sampled.astype(np.int64, copy=False)


def _window_starts_for_length(length, window, hop, layout, seed):
    assert length >= 0
    if length == 0:
        return np.zeros((0,), dtype=np.int64), False
    if length < window:
        return np.zeros((1,), dtype=np.int64), True
    starts = _pool_starts(length, window, hop, layout, seed)
    assert starts.size > 0
    return starts, False


def _allocate_point_budget(candidates, max_points, seed):
    if max_points <= 0:
        return [None] * len(candidates)

    total_points = sum(candidate["count"] for candidate in candidates)
    if total_points <= max_points:
        return [None] * len(candidates)

    rng = np.random.default_rng(_stable_seed(seed, "max_points", max_points))
    sampled = rng.choice(total_points, size=max_points, replace=False)
    sampled.sort()

    allocations = []
    offset = 0
    sample_start = 0
    for candidate in candidates:
        count = candidate["count"]
        next_offset = offset + count
        sample_end = int(np.searchsorted(sampled, next_offset, side="left"))
        local = sampled[sample_start:sample_end] - offset
        allocations.append(local.astype(np.int64, copy=False))
        offset = next_offset
        sample_start = sample_end
    return allocations


def _concat_window_embeddings(embeddings, window, hop, layout="sliding", seed=0, starts=None, short_segment=False):
    assert embeddings.ndim == 2
    feature_dim = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
    if embeddings.shape[0] == 0:
        return np.zeros((0, window * feature_dim), dtype=np.float32)

    if starts is None:
        starts, short_segment = _window_starts_for_length(embeddings.shape[0], window, hop, layout, seed)

    if short_segment:
        pad = np.zeros((window - embeddings.shape[0], feature_dim), dtype=np.float32)
        chunk = np.vstack([embeddings.astype(np.float32, copy=False), pad])
        return chunk.reshape(1, -1).astype(np.float32, copy=False)

    rows = []
    for start in starts.tolist():
        chunk = embeddings[start : start + window]
        rows.append(chunk.reshape(1, -1))

    if not rows:
        return np.zeros((0, window * feature_dim), dtype=np.float32)
    return np.vstack(rows).astype(np.float32, copy=False)


def _fit_pca(features, target_dim):
    assert features.ndim == 2
    if features.shape[0] == 0:
        return features.astype(np.float32, copy=False)

    n_components = min(int(target_dim), int(features.shape[0]), int(features.shape[1]))
    if n_components <= 0:
        return features.astype(np.float32, copy=False)
    if n_components == features.shape[1]:
        return features.astype(np.float32, copy=False)

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    return pca.fit_transform(features).astype(np.float32, copy=False)


def _recording_feature_aliases(data, stems):
    if "recording_birds" not in data.files or "bird_ids" not in data.files:
        return [(stem,) for stem in stems]
    bird_ids = [str(x) for x in data["bird_ids"].tolist()]
    recording_birds = np.asarray(data["recording_birds"])
    return [
        (stem, f"{bird_ids[int(bird_idx)]}__{stem}")
        for stem, bird_idx in zip(stems, recording_birds)
    ]


def _recording_feature_dict(row_aliases, features):
    result = {}
    for aliases, feature in zip(row_aliases, features):
        for alias in aliases:
            result[alias] = feature
    return result


def _load_recording_features(path, mode, dim, alpha, feature_norm, include_stems=None):
    if path is None:
        return None
    data = np.load(path, allow_pickle=True)
    stems = [str(x) for x in data["recording_stems"].tolist()]
    row_aliases = _recording_feature_aliases(data, stems)
    matrix = data["recording_matrix"].astype(np.float32, copy=False)
    if include_stems is not None:
        index = {
            alias: i
            for i, aliases in enumerate(row_aliases)
            for alias in aliases
        }
        stems = sorted(str(stem) for stem in include_stems)
        missing = [stem for stem in stems if stem not in index]
        assert not missing, missing[:5]
        keep = np.asarray([index[stem] for stem in stems], dtype=np.int64)
        matrix = matrix[np.ix_(keep, keep)]
        row_aliases = [(stem,) for stem in stems]
    matrix = (matrix + matrix.T) * np.float32(0.5)
    np.fill_diagonal(matrix, 0.0)
    if mode == "affinity_row":
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        features = matrix / np.maximum(norms, 1e-12)
        features = features.astype(np.float32, copy=False) * np.float32(alpha)
        return _recording_feature_dict(row_aliases, features)
    if mode == "affinity_prob":
        row_sums = matrix.sum(axis=1, keepdims=True)
        features = matrix / np.maximum(row_sums, 1e-12)
        features = features.astype(np.float32, copy=False) * np.float32(alpha)
        return _recording_feature_dict(row_aliases, features)

    if mode == "pca":
        dim = min(int(dim), int(matrix.shape[0]), int(matrix.shape[1]))
        assert dim > 0
        features = PCA(n_components=dim, svd_solver="full").fit_transform(matrix)
    elif mode == "umap":
        dim = min(int(dim), int(matrix.shape[0]) - 1)
        assert dim > 0
        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        rows = matrix / np.maximum(row_norms, 1e-12)
        features = umap.UMAP(
            n_components=dim,
            n_neighbors=min(15, max(2, matrix.shape[0] - 1)),
            min_dist=0.1,
            metric="cosine",
            random_state=0,
        ).fit_transform(rows)
    else:
        assert mode in {"svd", "svd_u", "svd_us", "normalized_svd", "norm_adj_eig", "norm_adj_eig_skip1", "norm_adj_eig_kmeans"}
    if mode in {"normalized_svd", "norm_adj_eig", "norm_adj_eig_skip1", "norm_adj_eig_kmeans"}:
        degree = matrix.sum(axis=1, keepdims=True)
        scale = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
        matrix = (matrix * scale) * scale.T

    if mode in {"pca", "umap"}:
        pass
    elif mode in {"norm_adj_eig", "norm_adj_eig_skip1", "norm_adj_eig_kmeans"}:
        values, vectors = np.linalg.eigh(matrix.astype(np.float64, copy=False))
        order = np.argsort(values)[::-1]
        start = 1 if mode == "norm_adj_eig_skip1" else 0
        dim = min(int(dim), int(values.shape[0]) - start)
        assert dim > 0
        if mode == "norm_adj_eig_kmeans":
            eig_dim = min(20, int(values.shape[0]) - start)
            assert eig_dim > 0
            eig_features = vectors[:, order[start : start + eig_dim]].astype(np.float32, copy=False)
            eig_norms = np.linalg.norm(eig_features, axis=1, keepdims=True)
            eig_features = eig_features / np.maximum(eig_norms, 1e-12)
            clusters = KMeans(n_clusters=dim, n_init=20, random_state=0).fit_predict(eig_features)
            features = np.eye(dim, dtype=np.float32)[clusters]
        else:
            features = vectors[:, order[start : start + dim]].astype(np.float32, copy=False)
    else:
        u, s, _ = np.linalg.svd(matrix, full_matrices=False)
        dim = min(int(dim), int(s.shape[0]))
        assert dim > 0
        if mode == "svd_u":
            features = u[:, :dim]
        elif mode == "svd_us":
            features = u[:, :dim] * s[:dim][None, :]
        else:
            features = u[:, :dim] * np.sqrt(s[:dim])[None, :]
    assert feature_norm in {"l2", "none"}
    if feature_norm == "l2":
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.maximum(norms, 1e-12)
    features = features.astype(np.float32, copy=False) * np.float32(alpha)
    return _recording_feature_dict(row_aliases, features)


def _combine_recording_features(primary, extra, stems=None):
    if extra is None:
        return primary
    if primary is None:
        return extra
    if stems is None:
        stems = sorted(primary)
    return {
        stem: np.hstack([primary[stem], extra[stem]]).astype(np.float32, copy=False)
        for stem in sorted(stems)
    }


def _recording_feature_name(mode):
    names = {
        "svd": "recsvd",
        "svd_u": "recsvdu",
        "svd_us": "recsvdus",
        "pca": "recpca",
        "umap": "recumap",
        "normalized_svd": "recnormsvd",
        "norm_adj_eig": "recnormeig",
        "norm_adj_eig_skip1": "recnormeigskip1",
        "norm_adj_eig_kmeans": "recnormeigkmeans",
        "affinity_row": "recaffrow",
        "affinity_prob": "recaffprob",
    }
    return names[mode]


def _recording_feature_suffix(mode, dim, alpha, scope):
    feature_name = _recording_feature_name(mode)
    dim_suffix = dim if mode not in {"affinity_row", "affinity_prob"} else "full"
    suffix = f"{feature_name}{dim_suffix}_a{alpha:g}"
    if scope != "full":
        suffix = f"{suffix}_{scope}"
    return suffix


def _apply_feature_postprocess(features, args):
    if args.feature_postprocess == "none":
        return features.astype(np.float32, copy=False), None
    transformed, transform = extract_embedding.maybe_apply_feature_postprocess(
        features,
        mode=args.feature_postprocess,
        dim=args.feature_postprocess_dim,
        load_path=args.feature_postprocess_load,
        save_path=args.feature_postprocess_save,
    )
    return transformed, transform


def _feature_postprocess_kind(feature_postprocess):
    if feature_postprocess is None:
        return None
    kind = feature_postprocess.get("kind")
    if kind is None:
        kind = feature_postprocess.get("mode")
    return kind


def _pool_embeddings(embeddings, window, mode, hop, layout="sliding", seed=0, starts=None, short_segment=False):
    assert embeddings.ndim == 2
    if embeddings.shape[0] == 0:
        return np.zeros((0, embeddings.shape[1]), dtype=np.float32)
    if window <= 1:
        if starts is not None:
            return embeddings[starts].astype(np.float32, copy=False)
        return embeddings.astype(np.float32, copy=False)
    assert mode in {"mean", "stats"}

    if starts is None:
        starts, short_segment = _window_starts_for_length(embeddings.shape[0], window, hop, layout, seed)
    if short_segment:
        return _pool_chunk_stats(embeddings) if mode == "stats" else embeddings.mean(axis=0, keepdims=True).astype(np.float32, copy=False)

    pooled = []
    for start in starts.tolist():
        chunk = embeddings[start : start + window]
        pooled.append(_pool_chunk_stats(chunk)[0] if mode == "stats" else chunk.mean(axis=0))

    if not pooled:
        pooled.append(_pool_chunk_stats(embeddings)[0] if mode == "stats" else embeddings.mean(axis=0))

    return np.asarray(pooled, dtype=np.float32)


def _pool_chunk_stats(chunk):
    quantiles = np.quantile(chunk, [0.25, 0.5, 0.75], axis=0)
    stats = [
        chunk.mean(axis=0),
        chunk.std(axis=0),
        chunk.min(axis=0),
        chunk.max(axis=0),
        quantiles[0],
        quantiles[1],
        quantiles[2],
    ]
    return np.concatenate(stats, axis=0)[None, :].astype(np.float32, copy=False)


def _pool_labels(labels, window, hop, layout="sliding", seed=0, starts=None, short_segment=False):
    assert labels.ndim == 1
    if labels.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if window <= 1:
        if starts is not None:
            return labels[starts].astype(np.int64, copy=False)
        return labels.astype(np.int64, copy=False)

    if starts is None:
        starts, short_segment = _window_starts_for_length(labels.shape[0], window, hop, layout, seed)
    if short_segment:
        values, counts = np.unique(labels, return_counts=True)
        return np.asarray([int(values[np.argmax(counts)])], dtype=np.int64)

    pooled = []
    for start in starts.tolist():
        chunk = labels[start : start + window]
        values, counts = np.unique(chunk, return_counts=True)
        pooled.append(int(values[np.argmax(counts)]))

    if not pooled:
        values, counts = np.unique(labels, return_counts=True)
        pooled.append(int(values[np.argmax(counts)]))

    return np.asarray(pooled, dtype=np.int64)


def _pad_feature_widths(arrays):
    assert arrays
    width = max(int(array.shape[1]) for array in arrays)
    padded = []
    for array in arrays:
        if int(array.shape[1]) == width:
            padded.append(array.astype(np.float32, copy=False))
            continue
        out = np.zeros((int(array.shape[0]), width), dtype=np.float32)
        out[:, : int(array.shape[1])] = array
        padded.append(out)
    return padded


def _mean_pool_spectrogram(spec, window_bins, hop_bins, layout="sliding", seed=0, starts=None, short_segment=False):
    assert spec.ndim == 2
    if spec.shape[1] == 0:
        return np.zeros((0, spec.shape[0]), dtype=np.float32)

    if starts is None:
        starts, short_segment = _window_starts_for_length(spec.shape[1], window_bins, hop_bins, layout, seed)

    if short_segment:
        return np.asarray([spec.mean(axis=1)], dtype=np.float32)

    pooled = []
    for start in starts.tolist():
        chunk = spec[:, start : start + window_bins]
        pooled.append(chunk.mean(axis=1))

    return np.asarray(pooled, dtype=np.float32)


def _fit_umap(features, neighbors, min_dist, metric, random_state, negative_sample_rate):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(neighbors),
        min_dist=float(min_dist),
        metric=metric,
        random_state=random_state,
        negative_sample_rate=int(negative_sample_rate),
        low_memory=True,
        n_jobs=-1,
    )
    return reducer.fit_transform(features)


def _fit_multiview_umap(features, recording_features, neighbors, min_dist, metric, random_state, negative_sample_rate, combine):
    rng = check_random_state(random_state)
    angular = metric in {"cosine", "correlation"}
    stats_graph, _, _ = umap_.fuzzy_simplicial_set(
        features,
        int(neighbors),
        rng,
        metric,
        {},
        angular=angular,
        set_op_mix_ratio=1.0,
    )
    recording_graph, _, _ = umap_.fuzzy_simplicial_set(
        recording_features,
        int(neighbors),
        rng,
        metric,
        {},
        angular=angular,
        set_op_mix_ratio=1.0,
    )
    product = stats_graph.multiply(recording_graph)
    assert combine in {"union", "intersection"}
    if combine == "union":
        graph = stats_graph + recording_graph - product
    else:
        graph = product
    graph.eliminate_zeros()
    graph = umap_.reset_local_connectivity(graph)
    a, b = umap_.find_ab_params(1.0, float(min_dist))
    embedding, _ = umap_.simplicial_set_embedding(
        features,
        graph,
        2,
        1.0,
        a,
        b,
        1.0,
        int(negative_sample_rate),
        None,
        "spectral",
        rng,
        metric,
        {},
        False,
        {},
        False,
        parallel=False,
    )
    return embedding.astype(np.float32, copy=False)


def _recording_view_features(recording_labels, recording_svd_features):
    return np.vstack([recording_svd_features[str(stem)] for stem in recording_labels]).astype(np.float32, copy=False)


def _label_silhouette(xy, labels, sample_size, seed):
    labels = np.asarray(labels)
    valid = np.isfinite(xy).all(axis=1)
    xy = xy[valid]
    labels = labels[valid]
    unique, counts = np.unique(labels, return_counts=True)
    total_points = int(xy.shape[0])
    classes = int(unique.shape[0])
    min_class_points = int(counts.min()) if counts.size else 0
    if xy.shape[0] < 3 or unique.shape[0] < 2 or unique.shape[0] >= xy.shape[0]:
        return {
            "score": None,
            "total_points": total_points,
            "scored_points": 0,
            "classes": classes,
            "min_class_points": min_class_points,
        }

    if sample_size > 0 and xy.shape[0] > sample_size:
        if unique.shape[0] >= sample_size:
            return {
                "score": None,
                "total_points": total_points,
                "scored_points": 0,
                "classes": classes,
                "min_class_points": min_class_points,
            }
        rng = np.random.default_rng(seed)
        keep = []
        for label in unique.tolist():
            label_indices = np.flatnonzero(labels == label)
            keep.append(int(rng.choice(label_indices)))
        remaining = int(sample_size) - len(keep)
        if remaining > 0:
            mask = np.ones((xy.shape[0],), dtype=bool)
            mask[np.asarray(keep, dtype=np.int64)] = False
            extra = rng.choice(np.flatnonzero(mask), size=remaining, replace=False)
            keep.extend(extra.tolist())
        keep = np.asarray(sorted(keep), dtype=np.int64)
        xy = xy[keep]
        labels = labels[keep]
    return {
        "score": float(silhouette_score(xy, labels, metric="euclidean")),
        "total_points": total_points,
        "scored_points": int(xy.shape[0]),
        "classes": classes,
        "min_class_points": min_class_points,
    }


def _syllable_plot_labels(birds, syllables):
    categories = []
    for bird, syllable in zip(birds.tolist(), syllables.tolist()):
        if int(syllable) < 0:
            categories.append("silence")
        else:
            categories.append(f"{bird}:{int(syllable)}")
    return np.asarray(categories, dtype=object)


def _umap_silhouette_scores(xy, bird_labels, syllable_labels, sample_size, seed):
    syllable_labels = np.asarray(syllable_labels)
    non_silence = syllable_labels >= 0
    syllable_categories = _syllable_plot_labels(bird_labels, syllable_labels)
    scores = {
        "bird": _label_silhouette(xy, bird_labels, sample_size, seed),
        "syllable": _label_silhouette(xy, syllable_categories, sample_size, seed),
        "syllable_non_silence": _label_silhouette(
            xy[non_silence],
            syllable_categories[non_silence],
            sample_size,
            seed,
        ),
    }
    print(
        "[umap] silhouette: "
        f"bird={scores['bird']['score']} "
        f"syllable={scores['syllable']['score']} "
        f"syllable_non_silence={scores['syllable_non_silence']['score']}"
    )
    return scores


def _label_metric(labels_true, labels_pred, metric):
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    if labels_true.shape[0] < 2:
        return None
    if np.unique(labels_true).shape[0] < 2 or np.unique(labels_pred).shape[0] < 2:
        return None
    return float(metric(labels_true, labels_pred))


def _median_int(values):
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float32)))


def _recording_cluster_interpretation(cluster_count, recording_homogeneity, median_clusters_per_recording, median_recordings_per_cluster):
    if cluster_count <= 1:
        return "single_cluster"
    if recording_homogeneity is not None and recording_homogeneity >= 0.8:
        return "recording_fracture_risk"
    if (
        median_clusters_per_recording is not None
        and median_recordings_per_cluster is not None
        and median_clusters_per_recording >= 2
        and median_recordings_per_cluster >= 2
    ):
        return "shared_multi_part_structure"
    return "mixed"


def _hdbscan_recording_rows(bird_labels, recording_labels, clusters):
    rows = []
    for bird_id in sorted(set(bird_labels.tolist())):
        bird_mask = bird_labels == bird_id
        bird_clusters = clusters[bird_mask]
        bird_recordings = recording_labels[bird_mask]
        non_noise = bird_clusters >= 0
        cluster_ids = sorted(set(bird_clusters[non_noise].tolist()))
        recording_ids = sorted(set(bird_recordings.tolist()))
        noise_fraction = float(np.mean(~non_noise)) if bird_clusters.size else 0.0

        valid_recordings = bird_recordings[non_noise]
        valid_clusters = bird_clusters[non_noise]
        recording_homogeneity = _label_metric(valid_recordings, valid_clusters, homogeneity_score)
        recording_completeness = _label_metric(valid_recordings, valid_clusters, completeness_score)
        recording_v_measure = _label_metric(valid_recordings, valid_clusters, v_measure_score)

        clusters_per_recording = []
        for recording in recording_ids:
            rec_clusters = bird_clusters[(bird_recordings == recording) & non_noise]
            clusters_per_recording.append(len(set(rec_clusters.tolist())))

        recordings_per_cluster = []
        for cluster_id in cluster_ids:
            cluster_recordings = bird_recordings[bird_clusters == cluster_id]
            recordings_per_cluster.append(len(set(cluster_recordings.tolist())))

        median_clusters_per_recording = _median_int(clusters_per_recording)
        median_recordings_per_cluster = _median_int(recordings_per_cluster)
        rows.append(
            {
                "bird_id": str(bird_id),
                "points": int(bird_clusters.shape[0]),
                "recordings": int(len(recording_ids)),
                "clusters": int(len(cluster_ids)),
                "noise_fraction": noise_fraction,
                "recording_homogeneity": recording_homogeneity,
                "recording_completeness": recording_completeness,
                "recording_v_measure": recording_v_measure,
                "median_clusters_per_recording": median_clusters_per_recording,
                "median_recordings_per_cluster": median_recordings_per_cluster,
                "interpretation": _recording_cluster_interpretation(
                    len(cluster_ids),
                    recording_homogeneity,
                    median_clusters_per_recording,
                    median_recordings_per_cluster,
                ),
            }
        )
    return rows


def _write_hdbscan_recording_csv(path, rows):
    fieldnames = [
        "bird_id",
        "points",
        "recordings",
        "clusters",
        "noise_fraction",
        "recording_homogeneity",
        "recording_completeness",
        "recording_v_measure",
        "median_clusters_per_recording",
        "median_recordings_per_cluster",
        "interpretation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _hdbscan_umap_analysis(xy, bird_labels, syllable_labels, recording_labels, out_dir, rep_name, args):
    import hdbscan

    min_cluster_size = args.hdbscan_min_cluster_size
    if min_cluster_size <= 0:
        min_cluster_size = max(25, int(round(xy.shape[0] * 0.005)))
    min_samples = args.hdbscan_min_samples if args.hdbscan_min_samples > 0 else None

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=min_samples,
    )
    clusters = clusterer.fit_predict(xy)
    non_noise = clusters >= 0
    cluster_ids = sorted(set(clusters[non_noise].tolist()))

    bird_homogeneity = _label_metric(bird_labels[non_noise], clusters[non_noise], homogeneity_score)
    bird_completeness = _label_metric(bird_labels[non_noise], clusters[non_noise], completeness_score)
    bird_v_measure = _label_metric(bird_labels[non_noise], clusters[non_noise], v_measure_score)
    rows = _hdbscan_recording_rows(bird_labels, recording_labels, clusters)

    _scatter_umap(
        xy=xy,
        labels=np.asarray([f"cluster_{int(label)}" if int(label) >= 0 else "noise" for label in clusters], dtype=object),
        title=_plot_title(args.species_display_name, "HDBSCAN"),
        out_base=out_dir / f"{rep_name}_hdbscan",
    )
    np.savez_compressed(
        out_dir / f"{rep_name}_hdbscan_points.npz",
        xy=xy.astype(np.float32, copy=False),
        bird_labels=bird_labels.astype(object, copy=False),
        syllable_labels=syllable_labels.astype(np.int64, copy=False),
        recording_labels=recording_labels.astype(object, copy=False),
        hdbscan_clusters=clusters.astype(np.int64, copy=False),
    )
    _write_hdbscan_recording_csv(out_dir / f"{rep_name}_hdbscan_recording_summary.csv", rows)

    summary = {
        "min_cluster_size": int(min_cluster_size),
        "min_samples": None if min_samples is None else int(min_samples),
        "points": int(xy.shape[0]),
        "clusters": int(len(cluster_ids)),
        "noise_fraction": float(np.mean(~non_noise)) if clusters.size else 0.0,
        "bird_homogeneity": bird_homogeneity,
        "bird_completeness": bird_completeness,
        "bird_v_measure": bird_v_measure,
        "recording_summary": rows,
    }
    (out_dir / f"{rep_name}_hdbscan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _bird_palette(birds):
    birds = sorted(set(birds))
    palette = {}
    for index, bird in enumerate(birds):
        hue = (index * 0.618033988749895) % 1.0
        saturation = 0.72 if index % 2 == 0 else 0.9
        value = 0.86 if (index // 2) % 2 == 0 else 0.68
        palette[bird] = np.asarray(
            colorsys.hsv_to_rgb(hue, saturation, value),
            dtype=np.float32,
        )
    return palette


def _format_extract_embedding_umap(ax):
    ax.set_xlabel("UMAP 1", fontsize=20, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=20, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def _format_umap_title(ax, title):
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)


def _plot_title(display_name, suffix=None):
    title = display_name
    if suffix is None:
        return title
    return f"{title} | {suffix}"


def _scatter_umap(xy, labels, title, out_base):
    birds = sorted(set(labels.tolist()))
    palette = _bird_palette(birds)

    fig = plt.figure(figsize=(5.5, 5.5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    for bird in birds:
        idx = labels == bird
        ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=10,
            alpha=0.15,
            color=palette[bird],
            label=bird,
            edgecolors="none",
        )

    _format_extract_embedding_umap(ax)
    _format_umap_title(ax, title)
    fig.tight_layout()
    fig.savefig(out_base.parent / f"{out_base.name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_base.parent / f"{out_base.name}.pdf", bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _scatter_umap_syllables(xy, syllables, birds, title, out_base):
    assert syllables.shape[0] == xy.shape[0]
    assert birds.shape[0] == xy.shape[0]

    categories = _syllable_plot_labels(birds, syllables).tolist()
    unique = sorted(set(categories))
    non_silence = [label for label in unique if label != "silence"]
    palette = {}
    cmap = plt.get_cmap("gist_ncar", max(1, len(non_silence)))
    for index, label in enumerate(non_silence):
        palette[label] = np.asarray(cmap(index), dtype=np.float32)[:3]
    if "silence" in unique:
        palette["silence"] = np.asarray([0.55, 0.55, 0.55], dtype=np.float32)

    fig = plt.figure(figsize=(5.5, 5.5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    categories_arr = np.asarray(categories, dtype=object)
    if "silence" in unique:
        idx = categories_arr == "silence"
        ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=10,
            alpha=0.1,
            color="#404040",
            edgecolors="none",
        )
    for label in non_silence:
        idx = categories_arr == label
        if idx.any():
            ax.scatter(
                xy[idx, 0],
                xy[idx, 1],
                s=10,
                alpha=0.15,
                color=palette[label],
                label=label,
                edgecolors="none",
            )

    _format_extract_embedding_umap(ax)
    _format_umap_title(ax, title)
    fig.tight_layout()
    fig.savefig(out_base.parent / f"{out_base.name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_base.parent / f"{out_base.name}.pdf", bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _scatter_single_umap(xy, title, out_base, color):
    fig = plt.figure(figsize=(5.5, 5.5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=10,
        alpha=0.15,
        color=color,
        edgecolors="none",
    )
    _format_extract_embedding_umap(ax)
    _format_umap_title(ax, title)
    fig.tight_layout()
    fig.savefig(out_base.parent / f"{out_base.name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_base.parent / f"{out_base.name}.pdf", bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _save_per_bird_umaps(
    per_bird_segments,
    args,
    patch_width,
    out_dir,
):
    birds = sorted(per_bird_segments)
    palette = _bird_palette(birds)
    per_bird_dir = out_dir / "per_bird"
    per_bird_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for bird_id in birds:
        single_bird = {bird_id: per_bird_segments[bird_id]}
        if args.encoder in {"SongMAE", "AVES"}:
            features, _, _, _ = _build_embedding_representation(
                per_bird_segments=single_bird,
                pool_window=args.pool_window,
                pool_hop=args.pool_hop,
                pool_mode=args.pool_mode,
                pool_layout=args.pool_layout,
                seed=args.seed,
                pca_dim=args.concat_pca_dim,
                max_points=args.max_points,
            )
        else:
            features, _, _, _ = _build_spec_representation(
                per_bird_segments=single_bird,
                pool_window=args.pool_window,
                pool_hop=args.pool_hop,
                patch_width=patch_width,
                pool_layout=args.pool_layout,
                seed=args.seed,
                max_points=args.max_points,
            )

        if features.shape[0] < 2:
            continue

        xy = _fit_umap(
            features,
            neighbors=min(args.umap_neighbors, max(1, features.shape[0] - 1)),
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=args.umap_random_state,
            negative_sample_rate=args.umap_negative_sample_rate,
        )
        out_base = per_bird_dir / f"{bird_id}"
        _scatter_single_umap(
            xy=xy,
            title=_plot_title(args.species_display_name, bird_id),
            out_base=out_base,
            color=palette[bird_id],
        )
        saved.append(
            {
                "bird_id": bird_id,
                "points": int(features.shape[0]),
                "out_base": str(out_base),
            }
        )

    return saved


def _load_songmae_segments_by_bird(args, sampled_recordings, model_state):
    stem_to_bird = {}
    recording_stems = []
    for bird_id in sorted(sampled_recordings):
        for recording_stem in sampled_recordings[bird_id]:
            stem_to_bird[recording_stem] = bird_id
            recording_stems.append(recording_stem)

    try:
        extracted = extract_embedding.extract_recording_embeddings_with_state(
            {
                "run_dir": str(args.run_dir),
                "checkpoint": args.checkpoint,
                "spec_dir": str(args.spec_dir),
                "json_path": str(args.annotation_json),
                "recording_stems": recording_stems,
                "recording_mode": args.recording_mode,
                "encoder_layer_idx": args.encoder_layer_idx,
                "spec_normalization": args.songmae_input_normalization,
                "normalization_stats_dir": args.songmae_input_normalization_stats_dir,
                "minimal_output": args.songmae_feature_source == "encoded_before",
                "embedding_postprocess": args.feature_postprocess,
                "embedding_postprocess_dim": args.feature_postprocess_dim,
                "embedding_postprocess_key": _songmae_feature_key(args.songmae_feature_source),
                "embedding_postprocess_load": args.feature_postprocess_load,
                "embedding_postprocess_save": args.feature_postprocess_save,
            },
            model_state,
        )
    except ValueError as exc:
        if str(exc) == "No valid patches extracted for the requested recording set.":
            return {}, None
        raise

    feature_key = _songmae_feature_key(args.songmae_feature_source)
    per_bird_segments = {}
    for segment in extracted["segments"]:
        bird_id = stem_to_bird[segment["recording_stem"]]
        features = _segment_songmae_features(segment, feature_key)
        labels = segment["labels_downsampled"]
        count = min(features.shape[0], labels.shape[0])
        if count == 0:
            continue
        per_bird_segments.setdefault(bird_id, []).append(
            {
                "features": features[:count],
                "labels": labels[:count],
                "recording_stem": segment["recording_stem"],
            }
        )
    return per_bird_segments, extracted.get("feature_postprocess")


def _load_aves_segments(args, bird_id, recording_stem, model_state):
    try:
        extracted = aves.extract_recording_embeddings_with_state(
            {
                "run_dir": str(args.run_dir),
                "checkpoint": args.checkpoint,
                "json_path": str(args.annotation_json),
                "bird": bird_id,
                "recording_stem": recording_stem,
                "recording_mode": args.recording_mode,
                "encoder_layer_idx": args.encoder_layer_idx,
                "wav_root": args.wav_root,
                "wav_manifest": args.wav_manifest,
                "wav_exts": args.wav_exts,
                "audio_sr": args.aves_audio_sr,
                "audio_context_seconds": getattr(args, "audio_context_seconds", 0.0),
                "aves_model_path": args.aves_model_path,
                "aves_config_path": args.aves_config_path,
                "seed": getattr(args, "seed", 0),
                "train_audio_speed_min_pct": getattr(args, "train_audio_speed_min_pct", 0.0),
                "train_audio_speed_max_pct": getattr(args, "train_audio_speed_max_pct", 0.0),
            },
            model_state,
        )
    except ValueError as exc:
        if str(exc) == "No valid AVES tokens extracted for the requested recording set.":
            return []
        raise

    segments = []
    for segment in extracted["segments"]:
        features = _segment_songmae_features(segment, "encoded_embeddings_before_pos_removal")
        labels = segment["labels_downsampled"]
        count = min(features.shape[0], labels.shape[0])
        if count == 0:
            continue
        segments.append(
            {
                "features": features[:count],
                "labels": labels[:count],
            }
        )
    return segments


def _load_hubert_segments(args, bird_id, recording_stem, model_state):
    try:
        extracted = hubert.extract_recording_embeddings_with_state(
            {
                "json_path": str(args.annotation_json),
                "bird": bird_id,
                "recording_stem": recording_stem,
                "recording_mode": args.recording_mode,
                "encoder_layer_idx": args.encoder_layer_idx,
                "wav_root": args.wav_root,
                "wav_manifest": args.wav_manifest,
                "wav_exts": args.wav_exts,
                "audio_sr": args.hubert_audio_sr,
                "audio_context_seconds": getattr(args, "audio_context_seconds", 0.0),
                "seed": getattr(args, "seed", 0),
                "train_audio_speed_min_pct": getattr(args, "train_audio_speed_min_pct", 0.0),
                "train_audio_speed_max_pct": getattr(args, "train_audio_speed_max_pct", 0.0),
            },
            model_state,
        )
    except ValueError as exc:
        if str(exc) == "No valid HuBERT tokens extracted for the requested recording set.":
            return []
        raise

    segments = []
    for segment in extracted["segments"]:
        features = _segment_songmae_features(segment, "encoded_embeddings_before_pos_removal")
        labels = segment["labels_downsampled"]
        count = min(features.shape[0], labels.shape[0])
        if count == 0:
            continue
        segments.append(
            {
                "features": features[:count],
                "labels": labels[:count],
            }
        )
    return segments


def _load_bird_mae_segments(args, bird_id, recording_stem, model_state):
    try:
        extracted = bird_mae.extract_recording_embeddings_with_state(
            {
                "json_path": str(args.annotation_json),
                "bird": bird_id,
                "recording_stem": recording_stem,
                "recording_mode": args.recording_mode,
                "wav_root": args.wav_root,
                "wav_manifest": args.wav_manifest,
                "wav_exts": args.wav_exts,
                "audio_sr": args.bird_mae_audio_sr,
                "audio_context_seconds": getattr(args, "audio_context_seconds", 0.0),
                "seed": getattr(args, "seed", 0),
                "train_audio_speed_min_pct": getattr(args, "train_audio_speed_min_pct", 0.0),
                "train_audio_speed_max_pct": getattr(args, "train_audio_speed_max_pct", 0.0),
            },
            model_state,
        )
    except ValueError as exc:
        if str(exc) == "No valid Bird-MAE embeddings extracted for the requested recording set.":
            return []
        raise

    segments = []
    for segment in extracted["segments"]:
        features = _segment_songmae_features(segment, "encoded_embeddings_before_pos_removal")
        labels = segment["labels_downsampled"]
        count = min(features.shape[0], labels.shape[0])
        if count == 0:
            continue
        segments.append(
            {
                "features": features[:count],
                "labels": labels[:count],
            }
        )
    return segments


def _load_perch_segments(args, bird_id, recording_stem, model_state):
    extracted = perch.extract_recording_embeddings_with_state(
        {
            "json_path": str(args.annotation_json),
            "bird": bird_id,
            "recording_stem": recording_stem,
            "recording_mode": args.recording_mode,
            "wav_root": args.wav_root,
            "wav_manifest": args.wav_manifest,
            "wav_exts": args.wav_exts,
            "perch_audio_sr": args.perch_audio_sr,
            "seed": getattr(args, "seed", 0),
            "train_audio_speed_min_pct": getattr(args, "train_audio_speed_min_pct", 0.0),
            "train_audio_speed_max_pct": getattr(args, "train_audio_speed_max_pct", 0.0),
        },
        model_state,
    )

    segments = []
    for segment in extracted["segments"]:
        features = segment["features"]
        if features.shape[0] == 0:
            continue
        labels = np.full((features.shape[0],), -1, dtype=np.int64)
        segments.append(
            {
                "features": features,
                "labels": labels,
            }
        )
    return segments


def _load_spec_segments(args, bird_id, recording_stem, patch_width):
    loaded = extract_embedding.load_recording_segments(
        {
            "spec_dir": str(args.spec_dir),
            "json_path": str(args.annotation_json),
            "bird": bird_id,
            "recording_stem": recording_stem,
            "recording_mode": args.recording_mode,
        }
    )
    stats_dir = args.spec_normalization_stats_dir or args.spec_dir
    mean, std = _load_target_stats(stats_dir)
    normalized_segments = extract_embedding.normalize_recording_segments(loaded["segments"], mean, std)

    segments = []
    audio_params = loaded["audio_params"]
    context_seconds = float(getattr(args, "audio_context_seconds", 0.0) or 0.0)
    context_timebins = 0
    if context_seconds > 0.0:
        context_timebins = int(round(context_seconds * float(audio_params[0]) / float(audio_params[2])))
        context_timebins = max(patch_width, context_timebins)
        context_timebins -= context_timebins % patch_width
        if context_timebins == 0:
            context_timebins = patch_width
    for segment in normalized_segments:
        spec = segment["spectrogram"]
        labels = segment["labels_original"]
        count = min(spec.shape[1], labels.shape[0])
        if count == 0:
            continue
        spec = spec[:, :count]
        labels = labels[:count]
        if context_timebins <= 0 or spec.shape[1] <= context_timebins:
            segments.append(
                {
                    "features": spec,
                    "labels": labels,
                }
            )
            continue
        for start in range(0, spec.shape[1], context_timebins):
            end = min(start + context_timebins, spec.shape[1])
            if end <= start:
                continue
            segments.append(
                {
                    "features": spec[:, start:end],
                    "labels": labels[start:end],
                }
            )
    return segments


def _build_embedding_representation(
    per_bird_segments,
    pool_window,
    pool_hop,
    pool_mode,
    pool_layout,
    seed,
    pca_dim,
    max_points=0,
    recording_svd_features=None,
    recording_svd_append="post",
):
    candidates = []
    for bird_id in sorted(per_bird_segments):
        for segment_index, segment in enumerate(per_bird_segments[bird_id]):
            length = min(int(segment["features"].shape[0]), int(segment["labels"].shape[0]))
            if length == 0:
                continue
            segment_seed = _stable_seed(seed, bird_id, segment_index)
            starts, short_segment = _window_starts_for_length(length, pool_window, pool_hop, pool_layout, segment_seed)
            candidates.append(
                {
                    "bird_id": bird_id,
                    "features": segment["features"][:length],
                    "labels": segment["labels"][:length],
                    "seed": segment_seed,
                    "starts": starts,
                    "short_segment": short_segment,
                    "count": int(starts.shape[0]),
                    "recording_stem": segment.get("recording_stem"),
                }
            )

    allocations = _allocate_point_budget(candidates, max_points, seed)
    pooled_by_bird = {}
    labels_by_bird = {}
    stems_by_bird = {}
    for candidate, local_indices in zip(candidates, allocations):
        starts = candidate["starts"]
        short_segment = candidate["short_segment"]
        if local_indices is not None:
            if local_indices.size == 0:
                continue
            if not short_segment:
                starts = starts[local_indices]
        candidate_features = candidate["features"]
        if recording_svd_features is not None and recording_svd_append == "frame":
            stem = candidate["recording_stem"]
            assert stem in recording_svd_features, stem
            recording_features = np.repeat(recording_svd_features[stem][None, :], candidate_features.shape[0], axis=0)
            candidate_features = np.hstack([candidate_features, recording_features]).astype(np.float32, copy=False)
        if pool_mode in {"concat", "concat_pca"}:
            pooled = _concat_window_embeddings(
                candidate_features,
                pool_window,
                pool_hop,
                layout=pool_layout,
                seed=candidate["seed"],
                starts=starts,
                short_segment=short_segment,
            )
        else:
            pooled = _pool_embeddings(
                candidate_features,
                pool_window,
                pool_mode,
                pool_hop,
                layout=pool_layout,
                seed=candidate["seed"],
                starts=starts,
                short_segment=short_segment,
            )
        pooled_labels = _pool_labels(
            candidate["labels"],
            pool_window,
            pool_hop,
            layout=pool_layout,
            seed=candidate["seed"],
            starts=starts,
            short_segment=short_segment,
        )
        count = min(pooled.shape[0], pooled_labels.shape[0])
        if count == 0:
            continue
        if recording_svd_features is not None and recording_svd_append == "post":
            stem = candidate["recording_stem"]
            assert stem in recording_svd_features, stem
            recording_features = np.repeat(recording_svd_features[stem][None, :], count, axis=0)
            pooled = np.hstack([pooled[:count], recording_features]).astype(np.float32, copy=False)
        bird_id = candidate["bird_id"]
        pooled_by_bird.setdefault(bird_id, []).append(pooled[:count])
        labels_by_bird.setdefault(bird_id, []).append(pooled_labels[:count])
        stems_by_bird.setdefault(bird_id, []).append(np.repeat(str(candidate["recording_stem"]), count))

    x_parts = []
    y_parts = []
    s_parts = []
    r_parts = []
    for bird_id in sorted(pooled_by_bird):
        bird_features = np.vstack(_pad_feature_widths(pooled_by_bird[bird_id]))
        bird_labels = np.concatenate(labels_by_bird[bird_id], axis=0)
        bird_recordings = np.concatenate(stems_by_bird[bird_id], axis=0)
        x_parts.append(bird_features)
        y_parts.extend([bird_id] * bird_features.shape[0])
        s_parts.append(bird_labels)
        r_parts.append(bird_recordings)

    assert x_parts, "No valid embedding segments were pooled."
    features = np.vstack(_pad_feature_widths(x_parts))
    if pool_mode == "concat_pca":
        features = _fit_pca(features, pca_dim)
    return (
        features,
        np.asarray(y_parts, dtype=object),
        np.concatenate(s_parts, axis=0),
        np.concatenate(r_parts, axis=0).astype(object, copy=False),
    )


def _build_spec_representation(per_bird_segments, pool_window, pool_hop, patch_width, pool_layout, seed, max_points=0):
    window_bins = pool_window * patch_width
    hop_bins = pool_hop * patch_width
    candidates = []
    for bird_id in sorted(per_bird_segments):
        for segment_index, segment in enumerate(per_bird_segments[bird_id]):
            length = min(int(segment["features"].shape[1]), int(segment["labels"].shape[0]))
            if length == 0:
                continue
            segment_seed = _stable_seed(seed, bird_id, segment_index)
            starts, short_segment = _window_starts_for_length(length, window_bins, hop_bins, pool_layout, segment_seed)
            candidates.append(
                {
                    "bird_id": bird_id,
                    "features": segment["features"][:, :length],
                    "labels": segment["labels"][:length],
                    "seed": segment_seed,
                    "starts": starts,
                    "short_segment": short_segment,
                    "count": int(starts.shape[0]),
                    "recording_stem": segment.get("recording_stem"),
                }
            )

    allocations = _allocate_point_budget(candidates, max_points, seed)
    pooled_by_bird = {}
    labels_by_bird = {}
    stems_by_bird = {}
    for candidate, local_indices in zip(candidates, allocations):
        starts = candidate["starts"]
        short_segment = candidate["short_segment"]
        if local_indices is not None:
            if local_indices.size == 0:
                continue
            if not short_segment:
                starts = starts[local_indices]
        pooled = _mean_pool_spectrogram(
            candidate["features"],
            window_bins,
            hop_bins,
            layout=pool_layout,
            seed=candidate["seed"],
            starts=starts,
            short_segment=short_segment,
        )
        pooled_labels = _pool_labels(
            candidate["labels"],
            window_bins,
            hop_bins,
            layout=pool_layout,
            seed=candidate["seed"],
            starts=starts,
            short_segment=short_segment,
        )
        count = min(pooled.shape[0], pooled_labels.shape[0])
        if count == 0:
            continue
        bird_id = candidate["bird_id"]
        pooled_by_bird.setdefault(bird_id, []).append(pooled[:count])
        labels_by_bird.setdefault(bird_id, []).append(pooled_labels[:count])
        stems_by_bird.setdefault(bird_id, []).append(np.repeat(str(candidate["recording_stem"]), count))

    x_parts = []
    y_parts = []
    s_parts = []
    r_parts = []
    for bird_id in sorted(pooled_by_bird):
        bird_features = np.vstack(pooled_by_bird[bird_id])
        bird_labels = np.concatenate(labels_by_bird[bird_id], axis=0)
        bird_recordings = np.concatenate(stems_by_bird[bird_id], axis=0)
        x_parts.append(bird_features)
        y_parts.extend([bird_id] * bird_features.shape[0])
        s_parts.append(bird_labels)
        r_parts.append(bird_recordings)

    assert x_parts, "No valid spectrogram segments were pooled."
    return (
        np.vstack(x_parts),
        np.asarray(y_parts, dtype=object),
        np.concatenate(s_parts, axis=0),
        np.concatenate(r_parts, axis=0).astype(object, copy=False),
    )


def _load_patch_width(run_dir):
    config_path = run_dir / "config.json"
    assert config_path.exists(), f"Missing config.json in run_dir: {run_dir}"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    patch_width = int(config["patch_width"])
    assert patch_width > 0
    return patch_width


def _apply_spec_normalization_preset(args):
    if args.encoder != "Spec" or args.normalization_preset is None:
        return

    stats_dir = args.audio_params_stats_dir
    if args.normalization_preset == "vanilla":
        args.spec_normalization = "none"
        return
    if args.normalization_preset == "zscore":
        args.spec_normalization = "per_recording_cmvn"
        return
    assert args.normalization_preset == "zscore_rescaled"
    args.spec_normalization = "per_recording_cmvn_rescaled_to_target_stats"
    args.spec_normalization_stats_dir = stats_dir


def _songmae_input_normalization(model_state, args):
    mode, stats_dir = "audio_params", model_state["run_dir"]
    if args.audio_params_stats_dir is not None:
        stats_dir = args.audio_params_stats_dir
    return mode, stats_dir


def main():
    parser = argparse.ArgumentParser(description="Individual-ID UMAPs with explicit encoder mode and record-wise pooling.")
    parser.add_argument("--encoder", required=True, choices=["SongMAE", "Spec", "AVES", "HuBERT", "BirdMAE", "Perch"])
    parser.add_argument("--species", required=True)
    parser.add_argument("--annotation_json", required=True)
    parser.add_argument("--spec_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--recording_mode", default=None, choices=["events", "full_recordings"])
    parser.add_argument("--songs_per_bird", type=int, default=None)
    parser.add_argument("--max_birds", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool_window", type=int, default=None)
    parser.add_argument("--pool_hop", type=int, default=None)
    parser.add_argument("--pool_mode", default="mean", choices=["mean", "stats", "concat", "concat_pca"])
    parser.add_argument("--concat_pca_dim", type=int, default=256)
    parser.add_argument("--pool_layout", default="sliding", choices=["sliding", "shotgun"])
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--feature_postprocess", default=None, choices=["none", "pca_whiten_l2", "whiten_l2"])
    parser.add_argument("--feature_postprocess_dim", type=int, default=None)
    parser.add_argument("--feature_postprocess_load", default=None)
    parser.add_argument("--feature_postprocess_save", default=None)
    parser.add_argument("--recording_svd_npz", default=None)
    parser.add_argument("--recording_feature_mode", default="svd", choices=["svd", "svd_u", "svd_us", "pca", "umap", "normalized_svd", "norm_adj_eig", "norm_adj_eig_skip1", "norm_adj_eig_kmeans", "affinity_row", "affinity_prob"])
    parser.add_argument("--recording_feature_scope", default="full", choices=["full", "sampled"])
    parser.add_argument("--recording_svd_dim", type=int, default=32)
    parser.add_argument("--recording_svd_alpha", type=float, default=1.0)
    parser.add_argument("--recording_feature_norm", default="l2", choices=["l2", "none"])
    parser.add_argument("--recording_extra_feature_mode", default=None, choices=["svd", "svd_u", "svd_us", "pca", "umap", "normalized_svd", "norm_adj_eig", "norm_adj_eig_skip1", "norm_adj_eig_kmeans", "affinity_row", "affinity_prob"])
    parser.add_argument("--recording_extra_feature_scope", default="full", choices=["full", "sampled"])
    parser.add_argument("--recording_extra_feature_dim", type=int, default=32)
    parser.add_argument("--recording_extra_feature_alpha", type=float, default=1.0)
    parser.add_argument("--recording_svd_append", default="post", choices=["post", "frame"])
    parser.add_argument("--recording_view_combine", default="concat", choices=["concat", "union", "intersection"])
    parser.add_argument(
        "--songmae_feature_source",
        default="encoded_before",
        choices=["encoded_before", "encoded_after", "patch_pre_pos", "patch_before", "patch_after"],
    )
    parser.add_argument("--per_bird_umaps", action="store_true")
    parser.add_argument("--encoder_layer_idx", type=int, default=None)
    parser.add_argument("--normalization_preset", choices=["vanilla", "zscore", "zscore_rescaled"], default=None)
    parser.add_argument("--audio_params_stats_dir", default=None)
    parser.add_argument(
        "--spec_normalization",
        choices=["none", "audio_params", "per_recording_cmvn", "per_recording_cmvn_rescaled_to_target_stats"],
        default="none",
    )
    parser.add_argument("--spec_normalization_stats_dir", default=None)
    parser.add_argument("--aves_model_path", default=None)
    parser.add_argument("--aves_config_path", default=None)
    parser.add_argument("--wav_root", default=None)
    parser.add_argument("--wav_manifest", default=None)
    parser.add_argument("--wav_exts", default=".wav,.flac,.ogg,.mp3")
    parser.add_argument("--aves_audio_sr", type=int, default=16000)
    parser.add_argument("--hubert_model_name", default="facebook/hubert-base-ls960")
    parser.add_argument("--hubert_audio_sr", type=int, default=16000)
    parser.add_argument("--bird_mae_model_name", default="DBD-research-group/Bird-MAE-Base")
    parser.add_argument("--bird_mae_audio_sr", type=int, default=32000)
    parser.add_argument("--perch_model_name", default="perch_v2")
    parser.add_argument("--perch_audio_sr", type=int, default=32000)
    parser.add_argument("--perch_window_seconds", type=float, default=5.0)
    parser.add_argument("--audio_context_seconds", type=float, default=2.0)
    parser.add_argument("--umap_neighbors", type=int, default=200)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_metric", default="cosine")
    parser.add_argument("--umap_random_state", type=int, default=None)
    parser.add_argument("--umap_negative_sample_rate", type=int, default=5)
    parser.add_argument("--silhouette_sample_size", type=int, default=10000)
    parser.add_argument("--save_umap_features", action="store_true")
    parser.add_argument("--features_only", action="store_true")
    parser.add_argument("--hdbscan_analysis", action="store_true")
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=0)
    parser.add_argument("--hdbscan_min_samples", type=int, default=10)
    args = parser.parse_args()

    species_key, species_config = _species_config(args.species)
    if args.recording_mode is None:
        args.recording_mode = species_config["recording_mode"]
    if args.songs_per_bird is None:
        args.songs_per_bird = species_config["songs_per_bird"]
    if args.pool_window is None:
        args.pool_window = species_config["pool_window"]
    if args.pool_hop is None:
        args.pool_hop = species_config["pool_hop"]
    if args.feature_postprocess is None:
        args.feature_postprocess = species_config["feature_postprocess"]
    if args.feature_postprocess_dim is None:
        args.feature_postprocess_dim = species_config["feature_postprocess_dim"]
    if args.recording_svd_npz is None:
        args.recording_svd_npz = species_config.get("recording_svd_npz")
        args.recording_feature_mode = species_config.get("recording_feature_mode", args.recording_feature_mode)
        args.recording_feature_scope = species_config.get("recording_feature_scope", args.recording_feature_scope)
        args.recording_svd_dim = species_config.get("recording_svd_dim", args.recording_svd_dim)
        args.recording_svd_alpha = species_config.get("recording_svd_alpha", args.recording_svd_alpha)
        args.recording_svd_append = species_config.get("recording_svd_append", args.recording_svd_append)
    args.species_key = species_key
    args.species_display_name = species_config["display_name"]

    annotation_json = Path(args.annotation_json).resolve()
    spec_dir = Path(args.spec_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    run_dir = _resolve_run_dir(args.run_dir)
    args.annotation_json = str(annotation_json)
    args.spec_dir = str(spec_dir)
    args.out_dir = str(out_dir)
    args.run_dir = str(run_dir)
    if args.audio_params_stats_dir is not None:
        args.audio_params_stats_dir = str(Path(args.audio_params_stats_dir).resolve())
    if args.spec_normalization_stats_dir is not None:
        args.spec_normalization_stats_dir = str(Path(args.spec_normalization_stats_dir).resolve())
    if args.aves_model_path is not None:
        args.aves_model_path = str(Path(args.aves_model_path).resolve())
    if args.aves_config_path is not None:
        args.aves_config_path = str(Path(args.aves_config_path).resolve())
    if args.wav_root is not None:
        args.wav_root = str(Path(args.wav_root).resolve())
    if args.wav_manifest is not None:
        args.wav_manifest = str(Path(args.wav_manifest).resolve())
    if args.feature_postprocess_load is not None:
        args.feature_postprocess_load = str(Path(args.feature_postprocess_load).resolve())
    if args.feature_postprocess_save is not None:
        save_path = Path(args.feature_postprocess_save).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        args.feature_postprocess_save = str(save_path)

    assert annotation_json.exists(), f"annotation_json not found: {annotation_json}"
    assert spec_dir.is_dir(), f"spec_dir not found: {spec_dir}"
    assert args.pool_window > 0
    assert args.pool_hop > 0
    if args.encoder == "Spec":
        assert args.pool_mode == "mean", "Spec uses mean pooling only."

    _apply_spec_normalization_preset(args)

    out_dir.mkdir(parents=True, exist_ok=True)
    patch_width = 1 if args.encoder in {"AVES", "HuBERT", "BirdMAE", "Perch"} else _load_patch_width(run_dir)
    model_state = None
    args.songmae_input_normalization = None
    args.songmae_input_normalization_stats_dir = None
    if args.encoder == "SongMAE":
        model_state = extract_embedding.load_model_state(str(args.run_dir), args.checkpoint)
        (
            args.songmae_input_normalization,
            args.songmae_input_normalization_stats_dir,
        ) = _songmae_input_normalization(model_state, args)
    elif args.encoder == "AVES":
        model_state = aves.load_model_state_for_inference(
            {
                "run_dir": str(args.run_dir),
                "checkpoint": args.checkpoint,
                "encoder_layer_idx": args.encoder_layer_idx,
                "wav_root": args.wav_root,
                "wav_manifest": args.wav_manifest,
                "wav_exts": args.wav_exts,
                "audio_sr": args.aves_audio_sr,
                "aves_model_path": args.aves_model_path,
                "aves_config_path": args.aves_config_path,
            }
        )
    elif args.encoder == "HuBERT":
        model_state = hubert.load_model_state_for_inference(
            {
                "run_dir": str(args.run_dir),
                "wav_root": args.wav_root,
                "wav_manifest": args.wav_manifest,
                "wav_exts": args.wav_exts,
                "hubert_model_name": args.hubert_model_name,
                "hubert_audio_sr": args.hubert_audio_sr,
            }
        )
    elif args.encoder == "BirdMAE":
        model_state = bird_mae.load_model_state_for_inference(
            {
                "run_dir": str(args.run_dir),
                "wav_root": args.wav_root,
                "wav_manifest": args.wav_manifest,
                "wav_exts": args.wav_exts,
                "bird_mae_model_name": args.bird_mae_model_name,
                "bird_mae_audio_sr": args.bird_mae_audio_sr,
            }
        )
    elif args.encoder == "Perch":
        model_state = perch.load_model_state_for_inference(
            {
                "run_dir": str(args.run_dir),
                "wav_root": args.wav_root,
                "wav_manifest": args.wav_manifest,
                "wav_exts": args.wav_exts,
                "perch_model_name": args.perch_model_name,
                "perch_audio_sr": args.perch_audio_sr,
                "perch_window_seconds": args.perch_window_seconds,
            }
        )

    stems_by_bird = _load_recording_stems_by_bird(annotation_json)
    bird_ids = sorted(stems_by_bird)
    if args.max_birds > 0:
        bird_ids = bird_ids[: args.max_birds]

    sampled_recordings = {}
    for bird_id in bird_ids:
        picks = _pick_recordings(
            stems_by_bird[bird_id],
            songs_per_bird=args.songs_per_bird,
            seed=args.seed,
            bird_id=bird_id,
        )
        if picks:
            sampled_recordings[bird_id] = picks

    assert sampled_recordings, "No birds have enough recordings after filtering."

    feature_postprocess = None
    per_bird_segments = {}
    if args.encoder == "SongMAE":
        per_bird_segments, feature_postprocess = _load_songmae_segments_by_bird(
            args,
            sampled_recordings,
            model_state,
        )
    else:
        for bird_id in sorted(sampled_recordings):
            bird_segments = []
            for recording_stem in sampled_recordings[bird_id]:
                if args.encoder == "AVES":
                    loaded_segments = _load_aves_segments(args, bird_id, recording_stem, model_state)
                elif args.encoder == "HuBERT":
                    loaded_segments = _load_hubert_segments(args, bird_id, recording_stem, model_state)
                elif args.encoder == "BirdMAE":
                    loaded_segments = _load_bird_mae_segments(args, bird_id, recording_stem, model_state)
                elif args.encoder == "Perch":
                    loaded_segments = _load_perch_segments(args, bird_id, recording_stem, model_state)
                else:
                    loaded_segments = _load_spec_segments(args, bird_id, recording_stem, patch_width)
                bird_segments.extend(loaded_segments)
            if bird_segments:
                per_bird_segments[bird_id] = bird_segments

    assert per_bird_segments, "No valid segments were extracted."
    recording_feature_stems = None
    if args.recording_feature_scope == "sampled" or args.recording_extra_feature_scope == "sampled":
        recording_feature_stems = {
            segment["recording_stem"]
            for segments in per_bird_segments.values()
            for segment in segments
        }
    recording_svd_features = _load_recording_features(
        args.recording_svd_npz,
        args.recording_feature_mode,
        args.recording_svd_dim,
        args.recording_svd_alpha,
        args.recording_feature_norm,
        include_stems=recording_feature_stems if args.recording_feature_scope == "sampled" else None,
    )
    recording_extra_features = None
    if args.recording_extra_feature_mode is not None:
        recording_extra_features = _load_recording_features(
            args.recording_svd_npz,
            args.recording_extra_feature_mode,
            args.recording_extra_feature_dim,
            args.recording_extra_feature_alpha,
            args.recording_feature_norm,
            include_stems=recording_feature_stems if args.recording_extra_feature_scope == "sampled" else None,
        )
    recording_svd_features = _combine_recording_features(
        recording_svd_features,
        recording_extra_features,
        stems=recording_feature_stems,
    )
    if args.recording_view_combine != "concat":
        assert recording_svd_features is not None
        assert args.recording_svd_append == "post"

    if args.encoder in {"SongMAE", "AVES", "HuBERT", "BirdMAE", "Perch"}:
        features, bird_labels, syllable_labels, recording_labels = _build_embedding_representation(
            per_bird_segments=per_bird_segments,
            pool_window=args.pool_window,
            pool_hop=args.pool_hop,
            pool_mode=args.pool_mode,
            pool_layout=args.pool_layout,
            seed=args.seed,
            pca_dim=args.concat_pca_dim,
            max_points=args.max_points,
            recording_svd_features=recording_svd_features if args.recording_view_combine == "concat" else None,
            recording_svd_append=args.recording_svd_append,
        )
        prefix = {"SongMAE": "songmae", "AVES": "aves", "HuBERT": "hubert", "BirdMAE": "birdmae", "Perch": "perch"}[args.encoder]
        source_suffix = ""
        if args.encoder == "SongMAE" and args.songmae_feature_source != "encoded_before":
            source_suffix = f"_{args.songmae_feature_source}"
        pca_suffix = args.concat_pca_dim if args.pool_mode == "concat_pca" else ""
        rep_name = f"{prefix}{source_suffix}_pool_{args.pool_mode}{pca_suffix}_{args.pool_layout}_w{args.pool_window}_h{args.pool_hop}"
    else:
        features, bird_labels, syllable_labels, recording_labels = _build_spec_representation(
            per_bird_segments=per_bird_segments,
            pool_window=args.pool_window,
            pool_hop=args.pool_hop,
            patch_width=patch_width,
            pool_layout=args.pool_layout,
            seed=args.seed,
            max_points=args.max_points,
        )
        rep_name = f"spec_pool_mean_{args.pool_layout}_w{args.pool_window}_h{args.pool_hop}"

    if args.max_points > 0:
        rep_name = f"{rep_name}_maxpts{args.max_points}"
    if recording_svd_features is not None:
        suffix = _recording_feature_suffix(
            args.recording_feature_mode,
            args.recording_svd_dim,
            args.recording_svd_alpha,
            args.recording_feature_scope,
        )
        rep_name = f"{rep_name}_{suffix}"
        if args.recording_feature_norm != "l2":
            rep_name = f"{rep_name}_recnorm{args.recording_feature_norm}"
        if args.recording_extra_feature_mode is not None:
            extra_suffix = _recording_feature_suffix(
                args.recording_extra_feature_mode,
                args.recording_extra_feature_dim,
                args.recording_extra_feature_alpha,
                args.recording_extra_feature_scope,
            )
            rep_name = f"{rep_name}_plus_{extra_suffix}"
        if args.recording_svd_append != "post":
            rep_name = f"{rep_name}_{args.recording_svd_append}"
        if args.recording_view_combine != "concat":
            rep_name = f"{rep_name}_recview{args.recording_view_combine}"

    if args.encoder != "SongMAE":
        features, feature_postprocess = _apply_feature_postprocess(features, args)
    if feature_postprocess is not None:
        rep_name = f"{rep_name}_{_feature_postprocess_kind(feature_postprocess)}{feature_postprocess['dim']}"
    assert features.shape[0] >= 2, "Not enough points for UMAP."
    print(f"[umap] {rep_name}: points={features.shape[0]} dim={features.shape[1]}")
    if args.save_umap_features:
        feature_path = out_dir / f"{rep_name}_features.npz"
        np.savez_compressed(
            feature_path,
            features=features.astype(np.float32, copy=False),
            bird_labels=bird_labels.astype(object, copy=False),
            syllable_labels=syllable_labels.astype(np.int64, copy=False),
            recording_labels=recording_labels.astype(object, copy=False),
        )
    if args.features_only:
        assert args.save_umap_features
        summary = {
            "model": {
                "encoder": args.encoder,
                "run_dir": str(args.run_dir),
                "checkpoint": args.checkpoint,
                "patch_width": int(patch_width),
            },
            "species": args.species,
            "species_key": args.species_key,
            "representation": rep_name,
            "feature_path": str(feature_path),
            "points": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "args": {
                "annotation_json": args.annotation_json,
                "spec_dir": args.spec_dir,
                "recording_mode": args.recording_mode,
                "songs_per_bird": int(args.songs_per_bird),
                "seed": int(args.seed),
                "pool_window": int(args.pool_window),
                "pool_hop": int(args.pool_hop),
                "pool_mode": args.pool_mode,
                "pool_layout": args.pool_layout,
                "max_points": int(args.max_points),
                "feature_postprocess": _feature_postprocess_kind(feature_postprocess) or "none",
                "feature_postprocess_dim": int(feature_postprocess["dim"]) if feature_postprocess is not None else 0,
                "recording_svd_npz": args.recording_svd_npz,
                "recording_feature_mode": args.recording_feature_mode,
                "recording_feature_norm": args.recording_feature_norm,
                "recording_svd_dim": int(args.recording_svd_dim),
                "recording_svd_alpha": float(args.recording_svd_alpha),
                "recording_svd_append": args.recording_svd_append,
                "recording_view_combine": args.recording_view_combine,
                "save_umap_features": bool(args.save_umap_features),
                "features_only": bool(args.features_only),
            },
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return

    if args.recording_view_combine == "concat":
        xy = _fit_umap(
            features,
            neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=args.umap_random_state,
            negative_sample_rate=args.umap_negative_sample_rate,
        )
    else:
        recording_view_features = _recording_view_features(recording_labels, recording_svd_features)
        xy = _fit_multiview_umap(
            features,
            recording_view_features,
            neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=args.umap_random_state,
            negative_sample_rate=args.umap_negative_sample_rate,
            combine=args.recording_view_combine,
        )
    silhouette_scores = _umap_silhouette_scores(
        xy=xy,
        bird_labels=bird_labels,
        syllable_labels=syllable_labels,
        sample_size=args.silhouette_sample_size,
        seed=args.seed,
    )
    hdbscan_summary = None
    if args.hdbscan_analysis:
        hdbscan_summary = _hdbscan_umap_analysis(
            xy=xy,
            bird_labels=bird_labels,
            syllable_labels=syllable_labels,
            recording_labels=recording_labels,
            out_dir=out_dir,
            rep_name=rep_name,
            args=args,
        )

    out_base = out_dir / rep_name
    _scatter_umap(
        xy=xy,
        labels=bird_labels,
        title=_plot_title(args.species_display_name),
        out_base=out_base,
    )
    _scatter_umap_syllables(
        xy=xy,
        syllables=syllable_labels,
        birds=bird_labels,
        title=_plot_title(args.species_display_name, "syllables"),
        out_base=out_dir / f"{rep_name}_syllable",
    )

    per_bird_saved = []
    if args.per_bird_umaps:
        per_bird_saved = _save_per_bird_umaps(
            per_bird_segments=per_bird_segments,
            args=args,
            patch_width=patch_width,
            out_dir=out_dir,
        )

    summary = {
        "model": {
            "encoder": args.encoder,
            "run_dir": str(run_dir),
            "checkpoint": args.checkpoint,
            "patch_width": patch_width,
        },
        "species": args.species,
        "species_key": args.species_key,
        "species_display_name": args.species_display_name,
        "species_defaults": {
            "pool_window": int(species_config["pool_window"]),
            "pool_hop": int(species_config["pool_hop"]),
            "recording_mode": species_config["recording_mode"],
            "songs_per_bird": int(species_config["songs_per_bird"]),
            "feature_postprocess": species_config["feature_postprocess"],
            "feature_postprocess_dim": int(species_config["feature_postprocess_dim"]),
            "recording_svd_npz": species_config.get("recording_svd_npz"),
            "recording_feature_mode": species_config.get("recording_feature_mode"),
            "recording_feature_scope": species_config.get("recording_feature_scope"),
            "recording_feature_norm": species_config.get("recording_feature_norm", "l2"),
            "recording_svd_dim": species_config.get("recording_svd_dim"),
            "recording_svd_alpha": species_config.get("recording_svd_alpha"),
            "recording_svd_append": species_config.get("recording_svd_append"),
        },
        "args": {
            "annotation_json": str(annotation_json),
            "spec_dir": str(spec_dir),
            "recording_mode": args.recording_mode,
            "songs_per_bird": int(args.songs_per_bird),
            "max_birds": int(args.max_birds),
            "seed": int(args.seed),
            "pool_window": int(args.pool_window),
            "pool_hop": int(args.pool_hop),
            "pool_mode": args.pool_mode,
            "pool_layout": args.pool_layout,
            "max_points": int(args.max_points),
            "concat_pca_dim": int(args.concat_pca_dim),
            "feature_postprocess": _feature_postprocess_kind(feature_postprocess) or "none",
            "feature_postprocess_dim": int(feature_postprocess["dim"]) if feature_postprocess is not None else 0,
            "feature_postprocess_load": args.feature_postprocess_load,
            "feature_postprocess_save": args.feature_postprocess_save,
            "recording_svd_npz": args.recording_svd_npz,
            "recording_feature_mode": args.recording_feature_mode,
            "recording_feature_scope": args.recording_feature_scope,
            "recording_feature_norm": args.recording_feature_norm,
            "recording_svd_dim": int(args.recording_svd_dim),
            "recording_svd_alpha": float(args.recording_svd_alpha),
            "recording_extra_feature_mode": args.recording_extra_feature_mode,
            "recording_extra_feature_scope": args.recording_extra_feature_scope,
            "recording_extra_feature_dim": int(args.recording_extra_feature_dim),
            "recording_extra_feature_alpha": float(args.recording_extra_feature_alpha),
            "recording_svd_append": args.recording_svd_append,
            "recording_view_combine": args.recording_view_combine,
            "per_bird_umaps": bool(args.per_bird_umaps),
            "normalization_preset": args.normalization_preset,
            "audio_params_stats_dir": args.audio_params_stats_dir,
            "spec_normalization": args.spec_normalization,
            "spec_normalization_stats_dir": args.spec_normalization_stats_dir,
            "songmae_input_normalization": args.songmae_input_normalization,
            "songmae_input_normalization_stats_dir": args.songmae_input_normalization_stats_dir,
            "songmae_feature_source": args.songmae_feature_source,
            "aves_model_path": args.aves_model_path,
            "aves_config_path": args.aves_config_path,
            "wav_root": args.wav_root,
            "wav_manifest": args.wav_manifest,
            "wav_exts": args.wav_exts,
            "aves_audio_sr": int(args.aves_audio_sr),
            "hubert_model_name": args.hubert_model_name,
            "hubert_audio_sr": int(args.hubert_audio_sr),
            "bird_mae_model_name": args.bird_mae_model_name,
            "bird_mae_audio_sr": int(args.bird_mae_audio_sr),
            "perch_model_name": args.perch_model_name,
            "perch_audio_sr": int(args.perch_audio_sr),
            "perch_window_seconds": float(args.perch_window_seconds),
            "audio_context_seconds": float(args.audio_context_seconds),
            "encoder_layer_idx": args.encoder_layer_idx,
            "umap_neighbors": int(args.umap_neighbors),
            "umap_min_dist": float(args.umap_min_dist),
            "umap_metric": args.umap_metric,
            "umap_random_state": args.umap_random_state,
            "umap_negative_sample_rate": int(args.umap_negative_sample_rate),
            "silhouette_sample_size": int(args.silhouette_sample_size),
            "save_umap_features": bool(args.save_umap_features),
            "hdbscan_analysis": bool(args.hdbscan_analysis),
            "hdbscan_min_cluster_size": int(args.hdbscan_min_cluster_size),
            "hdbscan_min_samples": int(args.hdbscan_min_samples),
        },
        "silhouette_scores": silhouette_scores,
        "hdbscan_summary": hdbscan_summary,
        "per_bird_umaps": per_bird_saved,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
