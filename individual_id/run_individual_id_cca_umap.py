#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.core import extract_embedding  # noqa: E402
from src.core.utils import load_audio_params, load_json_events, resolve_single_spec_path  # noqa: E402
from individual_id.run_individual_id_umap import (  # noqa: E402
    _build_embedding_representation,
    _fit_umap,
    _hdbscan_umap_analysis,
    _load_recording_features,
    _load_recording_stems_by_bird,
    _load_songmae_segments_by_bird,
    _pick_recordings,
    _pool_chunk_stats,
    _bird_palette,
    _syllable_plot_labels,
    _songmae_input_normalization,
    _umap_silhouette_scores,
)


def _recording_view(recording_labels, recording_features):
    return np.vstack([recording_features[str(label)] for label in recording_labels]).astype(np.float32, copy=False)


def _load_features_npz(path):
    data = np.load(path, allow_pickle=True)
    return (
        data["features"].astype(np.float32, copy=False),
        data["bird_labels"].astype(object, copy=False),
        data["syllable_labels"].astype(np.int64, copy=False),
        data["recording_labels"].astype(object, copy=False),
        None,
    )


def _limit_points(features, bird_labels, syllable_labels, recording_labels, max_points, seed):
    if max_points <= 0 or features.shape[0] <= max_points:
        return features, bird_labels, syllable_labels, recording_labels
    rng = np.random.default_rng(seed)
    keep = rng.choice(features.shape[0], size=max_points, replace=False)
    keep.sort()
    return features[keep], bird_labels[keep], syllable_labels[keep], recording_labels[keep]


def _display_name(species):
    return str(species).replace("_", " ").title()


def _format_umap(ax, title):
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_xticks([])
    ax.set_yticks([])


def _scatter_cca_umap(xy, labels, title, out_base):
    birds = sorted(set(labels.tolist()))
    palette = _bird_palette(birds)
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)
    for bird in birds:
        idx = labels == bird
        ax.scatter(xy[idx, 0], xy[idx, 1], s=10, alpha=0.15, color=palette[bird], edgecolors="none")
    _format_umap(ax, title)
    fig.tight_layout()
    fig.savefig(out_base.parent / f"{out_base.name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_base.parent / f"{out_base.name}.pdf", bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _scatter_cca_syllables(xy, syllables, birds, title, out_base):
    categories = np.asarray(_syllable_plot_labels(birds, syllables).tolist(), dtype=object)
    unique = sorted(set(categories.tolist()))
    non_silence = [label for label in unique if label != "silence"]
    cmap = plt.get_cmap("gist_ncar", max(1, len(non_silence)))

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)
    if "silence" in unique:
        idx = categories == "silence"
        ax.scatter(xy[idx, 0], xy[idx, 1], s=10, alpha=0.1, color="#404040", edgecolors="none")
    for index, label in enumerate(non_silence):
        idx = categories == label
        ax.scatter(xy[idx, 0], xy[idx, 1], s=10, alpha=0.15, color=cmap(index), edgecolors="none")
    _format_umap(ax, title)
    fig.tight_layout()
    fig.savefig(out_base.parent / f"{out_base.name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_base.parent / f"{out_base.name}.pdf", bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _recording_means(features, recording_labels, recording_features):
    rows = []
    svd_rows = []
    for recording in sorted(set(recording_labels.tolist())):
        rows.append(features[recording_labels == recording].mean(axis=0))
        svd_rows.append(recording_features[str(recording)])
    return np.vstack(rows).astype(np.float32, copy=False), np.vstack(svd_rows).astype(np.float32, copy=False)


def _standardize(x):
    mean = x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    return (x - mean) / scale, mean, scale


def _thin_svd_cca(a, b, components):
    a, a_mean, a_scale = _standardize(a.astype(np.float64, copy=False))
    b, b_mean, b_scale = _standardize(b.astype(np.float64, copy=False))
    ua, sa, vta = np.linalg.svd(a, full_matrices=False)
    ub, sb, vtb = np.linalg.svd(b, full_matrices=False)
    keep_a = sa > 1e-8
    keep_b = sb > 1e-8
    ua, sa, vta = ua[:, keep_a], sa[keep_a], vta[keep_a]
    ub, sb, vtb = ub[:, keep_b], sb[keep_b], vtb[keep_b]
    left, corrs, right_t = np.linalg.svd(ua.T @ ub, full_matrices=False)
    k = min(int(components), left.shape[1], right_t.shape[0])
    assert k > 0
    a_coef = (vta.T / sa[None, :]) @ left[:, :k]
    b_coef = (vtb.T / sb[None, :]) @ right_t.T[:, :k]
    return {
        "a_mean": a_mean.astype(np.float32, copy=False),
        "a_scale": a_scale.astype(np.float32, copy=False),
        "b_mean": b_mean.astype(np.float32, copy=False),
        "b_scale": b_scale.astype(np.float32, copy=False),
        "a_coef": a_coef.astype(np.float32, copy=False),
        "b_coef": b_coef.astype(np.float32, copy=False),
        "correlations": [float(value) for value in corrs[:k]],
    }


def _rcca_fit(a, b, components, ridge_a, ridge_b):
    a, a_mean, a_scale = _standardize(a.astype(np.float64, copy=False))
    b, b_mean, b_scale = _standardize(b.astype(np.float64, copy=False))
    ua, sa, vta = np.linalg.svd(a, full_matrices=False)
    ub, sb, vtb = np.linalg.svd(b, full_matrices=False)
    keep_a = sa > 1e-8
    keep_b = sb > 1e-8
    ua, sa, vta = ua[:, keep_a], sa[keep_a], vta[keep_a]
    ub, sb, vtb = ub[:, keep_b], sb[keep_b], vtb[keep_b]
    n = max(a.shape[0] - 1, 1)
    da = sa / np.sqrt(sa**2 + n * float(ridge_a))
    db = sb / np.sqrt(sb**2 + n * float(ridge_b))
    left, corrs, right_t = np.linalg.svd((da[:, None] * (ua.T @ ub)) * db[None, :], full_matrices=False)
    k = min(int(components), left.shape[1], right_t.shape[0])
    assert k > 0
    a_coef = (vta.T / np.sqrt(sa[None, :] ** 2 / n + float(ridge_a))) @ left[:, :k]
    b_coef = (vtb.T / np.sqrt(sb[None, :] ** 2 / n + float(ridge_b))) @ right_t.T[:, :k]
    model = {
        "a_mean": a_mean.astype(np.float32, copy=False),
        "a_scale": a_scale.astype(np.float32, copy=False),
        "b_mean": b_mean.astype(np.float32, copy=False),
        "b_scale": b_scale.astype(np.float32, copy=False),
        "a_coef": a_coef.astype(np.float32, copy=False),
        "b_coef": b_coef.astype(np.float32, copy=False),
        "correlations": [float(value) for value in corrs[:k]],
        "ridge_a": float(ridge_a),
        "ridge_b": float(ridge_b),
    }
    train_a = a @ model["a_coef"]
    train_b = b @ model["b_coef"]
    model["a_score_scale"] = np.maximum(train_a.std(axis=0, keepdims=True), 1e-6).astype(np.float32, copy=False)
    model["b_score_scale"] = np.maximum(train_b.std(axis=0, keepdims=True), 1e-6).astype(np.float32, copy=False)
    return model


def _heldout_cca_score(a, b):
    scores = []
    for idx in range(a.shape[1]):
        if a[:, idx].std() > 1e-6 and b[:, idx].std() > 1e-6:
            scores.append(abs(float(np.corrcoef(a[:, idx], b[:, idx])[0, 1])))
    assert scores
    return float(np.mean(scores))


def _rcca_cv(a, b, components, ridge_values, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(a.shape[0])
    split = max(2, int(round(a.shape[0] * 0.8)))
    train = order[:split]
    valid = order[split:]
    assert valid.size
    rows = []
    for ridge in ridge_values:
        model = _rcca_fit(a[train], b[train], components, 0.0, ridge)
        valid_a, valid_b = _transform_cca(a[valid], b[valid], model)
        rows.append({"ridge_b": float(ridge), "heldout_score": _heldout_cca_score(valid_a, valid_b)})
    best = max(rows, key=lambda row: row["heldout_score"])
    model = _rcca_fit(a, b, components, 0.0, best["ridge_b"])
    model["ridge_cv"] = rows
    return model


def _transform_cca(features, recording_view, model, normalize=True):
    a = ((features - model["a_mean"]) / model["a_scale"]) @ model["a_coef"]
    b = ((recording_view - model["b_mean"]) / model["b_scale"]) @ model["b_coef"]
    if normalize and "a_score_scale" in model:
        a = a / model["a_score_scale"]
        b = b / model["b_score_scale"]
    return a.astype(np.float32, copy=False), b.astype(np.float32, copy=False)


def _block_normalize_views(a, b, recording_weight):
    a = (a - a.mean(axis=0, keepdims=True)) / np.maximum(a.std(axis=0, keepdims=True), 1e-6)
    b = (b - b.mean(axis=0, keepdims=True)) / np.maximum(b.std(axis=0, keepdims=True), 1e-6)
    a = a / np.sqrt(a.shape[1])
    b = b / np.sqrt(b.shape[1])
    return a.astype(np.float32, copy=False), (b * float(recording_weight)).astype(np.float32, copy=False)


def _parse_floats(text):
    return [float(value) for value in str(text).split(",") if value]


def _majority_label(labels):
    values, counts = np.unique(labels, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _short_stem(stem, limit=34):
    stem = str(stem)
    if len(stem) <= limit:
        return stem
    return f"{stem[: limit - 3]}..."


def _spectrogram_preview(spec_dir, stem, max_cols=1200):
    path = Path(spec_dir) / f"{stem}.npy"
    if not path.exists():
        return None
    spec = np.load(path, mmap_mode="r")
    step = max(1, int(np.ceil(spec.shape[1] / max_cols)))
    preview = np.asarray(spec[:, ::step], dtype=np.float32)
    lo, hi = np.percentile(preview, [2, 99])
    return np.clip((preview - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _write_recording_cluster_spectrograms(out_dir, rep_name, spec_dir, bird_labels, recording_labels, clusters):
    records = []
    for recording in sorted(set(recording_labels.tolist())):
        idx = recording_labels == recording
        non_noise = clusters[idx]
        non_noise = non_noise[non_noise >= 0]
        if non_noise.size == 0:
            continue
        bird = _majority_label(bird_labels[idx]) if np.issubdtype(bird_labels.dtype, np.integer) else str(np.unique(bird_labels[idx])[0])
        cluster = _majority_label(non_noise)
        records.append(
            {
                "bird": str(bird),
                "recording": str(recording),
                "cluster": int(cluster),
                "points": int(idx.sum()),
            }
        )

    by_bird = {}
    for record in records:
        by_bird.setdefault(record["bird"], []).append(record)
    split_birds = [
        (bird, rows)
        for bird, rows in by_bird.items()
        if len({row["cluster"] for row in rows}) > 1
    ]
    split_birds.sort(key=lambda item: (len({row["cluster"] for row in item[1]}), len(item[1])), reverse=True)
    split_birds = split_birds[:6]
    if not split_birds:
        return None

    selections = []
    for bird, rows in split_birds:
        by_cluster = {}
        for row in rows:
            current = by_cluster.get(row["cluster"])
            if current is None or row["points"] > current["points"]:
                by_cluster[row["cluster"]] = row
        chosen = [by_cluster[cluster] for cluster in sorted(by_cluster)[:4]]
        selections.append((bird, chosen))

    cols = max(len(rows) for _, rows in selections)
    fig, axes = plt.subplots(len(selections), cols, figsize=(3.8 * cols, 1.9 * len(selections)), squeeze=False)
    for row_idx, (bird, rows) in enumerate(selections):
        for col_idx in range(cols):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx >= len(rows):
                ax.axis("off")
                continue
            row = rows[col_idx]
            preview = _spectrogram_preview(spec_dir, row["recording"])
            if preview is None:
                ax.text(0.5, 0.5, "missing spec", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.imshow(preview, origin="lower", aspect="auto", cmap="magma", interpolation="nearest")
            ax.set_title(
                f"bird {bird} | cluster {row['cluster']}\n{_short_stem(row['recording'])}",
                fontsize=8,
            )
    fig.suptitle("Same individual, recordings assigned to different HDBSCAN clusters", fontsize=11, fontweight="bold")
    fig.tight_layout()
    png_path = out_dir / f"{rep_name}_recording_cluster_spectrograms.png"
    pdf_path = out_dir / f"{rep_name}_recording_cluster_spectrograms.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=220)
    fig.savefig(pdf_path, bbox_inches="tight", dpi=220)
    plt.close(fig)

    json_path = out_dir / f"{rep_name}_recording_cluster_spectrograms.json"
    json_path.write_text(json.dumps({"selections": selections}, indent=2), encoding="utf-8")
    return str(png_path)


def _pick_recordings_up_to(stems, songs_per_bird, seed, bird_id):
    if songs_per_bird <= 0 or len(stems) <= songs_per_bird:
        return list(stems)
    return _pick_recordings(stems, songs_per_bird, seed, bird_id)


def _pool_count_from_tokens(tokens, pool_window, pool_hop):
    tokens = int(tokens)
    if tokens <= 0:
        return 0
    if tokens < pool_window:
        return 1
    return 1 + (tokens - pool_window) // pool_hop


def _pool_count_from_timebins(timebins, patch_width, pool_window, pool_hop):
    tokens = (int(timebins) + int(patch_width) - 1) // int(patch_width)
    return _pool_count_from_tokens(tokens, pool_window, pool_hop)


def _estimated_pool_count(args, stem, patch_width, event_map):
    path = resolve_single_spec_path(args.spec_dir, stem)
    spec = np.load(path, mmap_mode="r")
    rounded_length = int(spec.shape[1]) - (int(spec.shape[1]) % int(patch_width))
    if rounded_length <= 0:
        return 0
    if args.recording_mode == "full_recordings":
        return _pool_count_from_timebins(rounded_length, patch_width, args.pool_window, args.pool_hop)

    total = 0
    for event in event_map.get(stem, []):
        start = max(0, min(int(event["on_timebins"]), rounded_length))
        end = max(start, min(int(event["off_timebins"]), rounded_length))
        total += _pool_count_from_timebins(end - start, patch_width, args.pool_window, args.pool_hop)
    return total


def _fit_recordings_to_point_budget(stems_by_bird, args, model_state):
    if args.songs_per_bird > 0 or args.max_points <= 0:
        return {
            bird_id: _pick_recordings_up_to(stems, args.songs_per_bird, args.seed, bird_id)
            for bird_id, stems in stems_by_bird.items()
        }
    patch_width = int(model_state["patch_width"])
    audio = load_audio_params(args.spec_dir, require_stats=False)
    audio_params = (
        audio["sr"],
        audio["mels"],
        audio["hop_size"],
        audio["fft"],
    )
    event_map = load_json_events(args.annotation_json, audio_params=audio_params)
    queues = {}
    total = 0
    for bird_id, stems in stems_by_bird.items():
        rows = [
            (count, stem)
            for stem in stems
            for count in [_estimated_pool_count(args, stem, patch_width, event_map)]
            if count > 0
        ]
        rows.sort()
        queues[bird_id] = rows
        total += sum(count for count, _ in rows)
    if total <= args.max_points:
        return {bird_id: [stem for _, stem in rows] for bird_id, rows in queues.items()}

    sampled = {bird_id: [] for bird_id in queues}
    used = 0
    while True:
        changed = False
        for bird_id in sorted(queues):
            if not queues[bird_id]:
                continue
            count, stem = queues[bird_id][0]
            if used + count > args.max_points:
                continue
            queues[bird_id].pop(0)
            sampled[bird_id].append(stem)
            used += count
            changed = True
        if not changed:
            break
    return sampled


def _build_recording_stats_representation(per_bird_segments):
    features = []
    bird_labels = []
    syllable_labels = []
    recording_labels = []
    for bird_id in sorted(per_bird_segments):
        by_recording = {}
        for segment in per_bird_segments[bird_id]:
            stem = str(segment["recording_stem"])
            by_recording.setdefault(stem, []).append(segment["features"])
        for stem in sorted(by_recording):
            x = np.vstack(by_recording[stem]).astype(np.float32, copy=False)
            features.append(_pool_chunk_stats(x)[0])
            bird_labels.append(bird_id)
            syllable_labels.append(-1)
            recording_labels.append(stem)
    assert features
    return (
        np.vstack(features).astype(np.float32, copy=False),
        np.asarray(bird_labels, dtype=object),
        np.asarray(syllable_labels, dtype=np.int64),
        np.asarray(recording_labels, dtype=object),
    )


def _load_songmae_points(args, allowed_recordings):
    model_state = extract_embedding.load_model_state(str(args.run_dir), args.checkpoint)
    norm_mode, norm_stats_dir = _songmae_input_normalization(model_state, args)
    args.songmae_input_normalization = norm_mode
    args.songmae_input_normalization_stats_dir = norm_stats_dir

    stems_by_bird = _load_recording_stems_by_bird(args.annotation_json)
    if allowed_recordings is not None:
        stems_by_bird = {
            bird_id: [stem for stem in stems if str(stem) in allowed_recordings]
            for bird_id, stems in stems_by_bird.items()
        }
    sampled = {
        bird_id: picks
        for bird_id, picks in _fit_recordings_to_point_budget(stems_by_bird, args, model_state).items()
        if picks
    }
    assert sampled

    per_bird_segments, feature_postprocess = _load_songmae_segments_by_bird(args, sampled, model_state)
    if args.recording_level_stats_pool:
        features, bird_labels, syllable_labels, recording_labels = _build_recording_stats_representation(per_bird_segments)
    else:
        features, bird_labels, syllable_labels, recording_labels = _build_embedding_representation(
            per_bird_segments=per_bird_segments,
            pool_window=args.pool_window,
            pool_hop=args.pool_hop,
            pool_mode=args.pool_mode,
            pool_layout=args.pool_layout,
            seed=args.seed,
            pca_dim=args.concat_pca_dim,
            max_points=args.max_points,
        )
    return features, bird_labels, syllable_labels, recording_labels, feature_postprocess


def main():
    parser = argparse.ArgumentParser(description="CCA UMAP for SongMAE pooled features and recording-level affinity features.")
    parser.add_argument("--features_npz", default="results/individual_id_umap/european_starling_metric_learning_source_stats_maxpts25000/songmae_pool_stats_sliding_w30_h5_maxpts25000_recsvd15_a1_recviewunion_pca_whiten_l21024_features.npz")
    parser.add_argument("--species", default="european_starling")
    parser.add_argument("--species_display_name", default=None)
    parser.add_argument("--annotation_json", default="files/annotation jsons/european_starling_annotations.json")
    parser.add_argument("--spec_dir", default="/media/george-vengrovski/disk2/specs/european_starling_64hop_32khz_prefixed")
    parser.add_argument("--run_dir", default=str(ROOT / "github_assets/xcl_voronoi_mask_no_normalize_32h_10w_5s_fp8"))
    parser.add_argument("--checkpoint", default="model_step_499999.pth")
    parser.add_argument("--recording_svd_npz", default="results/individual_id_knn_graph_metrics/bird_knn_matrix_laplacian_cap30/european_starling/knn_attribution_matrices.npz")
    parser.add_argument("--recording_feature_mode", default="svd", choices=["svd", "pca", "umap", "affinity_row"])
    parser.add_argument("--recording_svd_dim", type=int, default=15)
    parser.add_argument("--cca_mode", default="thin", choices=["thin", "rcca_cv", "rcca_fixed"])
    parser.add_argument("--cca_ridge_b", type=float, default=0.1)
    parser.add_argument("--cca_ridge_values", default="0.01,0.1,1.0")
    parser.add_argument("--out_dir", default="results/individual_id_umap/european_starling_cca_recsvd15_maxpts25000")
    parser.add_argument("--recording_mode", default="events")
    parser.add_argument("--songs_per_bird", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool_window", type=int, default=30)
    parser.add_argument("--pool_hop", type=int, default=5)
    parser.add_argument("--pool_mode", default="stats", choices=["mean", "stats", "concat", "concat_pca"])
    parser.add_argument("--pool_layout", default="sliding", choices=["sliding", "shotgun"])
    parser.add_argument("--recording_level_stats_pool", action="store_true")
    parser.add_argument("--concat_pca_dim", type=int, default=256)
    parser.add_argument("--max_points", type=int, default=25000)
    parser.add_argument("--feature_postprocess", default="pca_whiten_l2")
    parser.add_argument("--feature_postprocess_dim", type=int, default=1024)
    parser.add_argument("--feature_postprocess_load", default=None)
    parser.add_argument("--feature_postprocess_save", default=None)
    parser.add_argument("--songmae_feature_source", default="encoded_before")
    parser.add_argument("--encoder_layer_idx", type=int, default=None)
    parser.add_argument("--audio_params_stats_dir", default="/media/george-vengrovski/disk2/specs/european_starling_64hop_32khz_prefixed")
    parser.add_argument("--spec_normalization", default="none")
    parser.add_argument("--spec_normalization_stats_dir", default=None)
    parser.add_argument("--cca_components", type=int, default=15)
    parser.add_argument("--cca_block_normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recording_cca_weight", type=float, default=1.0)
    parser.add_argument("--umap_neighbors", type=int, default=50)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_metric", default="cosine")
    parser.add_argument("--umap_negative_sample_rate", type=int, default=5)
    parser.add_argument("--silhouette_sample_size", type=int, default=10000)
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=0)
    parser.add_argument("--hdbscan_min_samples", type=int, default=10)
    args = parser.parse_args()

    args.features_npz = str((ROOT / args.features_npz).resolve()) if args.features_npz else None
    args.annotation_json = str((ROOT / args.annotation_json).resolve())
    args.recording_svd_npz = str((ROOT / args.recording_svd_npz).resolve())
    args.spec_dir = str(Path(args.spec_dir).resolve())
    args.audio_params_stats_dir = str(Path(args.audio_params_stats_dir).resolve())
    args.run_dir = str(Path(args.run_dir).resolve())
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    recording_features = _load_recording_features(
        args.recording_svd_npz,
        args.recording_feature_mode,
        args.recording_svd_dim,
        1.0,
        "l2",
    )
    if args.features_npz:
        songmae_features, bird_labels, syllable_labels, recording_labels, feature_postprocess = _load_features_npz(args.features_npz)
        songmae_features, bird_labels, syllable_labels, recording_labels = _limit_points(
            songmae_features,
            bird_labels,
            syllable_labels,
            recording_labels,
            args.max_points,
            args.seed,
        )
    else:
        songmae_features, bird_labels, syllable_labels, recording_labels, feature_postprocess = _load_songmae_points(args, set(recording_features))
    recording_view = _recording_view(recording_labels, recording_features)
    recording_songmae, recording_fit_view = _recording_means(songmae_features, recording_labels, recording_features)
    if args.cca_mode == "thin":
        cca = _thin_svd_cca(recording_songmae, recording_fit_view, args.cca_components)
    elif args.cca_mode == "rcca_cv":
        cca = _rcca_cv(recording_songmae, recording_fit_view, args.cca_components, _parse_floats(args.cca_ridge_values), args.seed)
    else:
        cca = _rcca_fit(recording_songmae, recording_fit_view, args.cca_components, 0.0, args.cca_ridge_b)
    songmae_cca, recording_cca = _transform_cca(songmae_features, recording_view, cca)
    if args.cca_block_normalize:
        songmae_cca, recording_cca = _block_normalize_views(songmae_cca, recording_cca, args.recording_cca_weight)
    features = np.hstack([songmae_cca, recording_cca]).astype(np.float32, copy=False)
    if args.recording_feature_mode == "affinity_row":
        mode_tag = "recaffrow"
    elif args.recording_feature_mode == "pca":
        mode_tag = f"recpca{args.recording_svd_dim}"
    elif args.recording_feature_mode == "umap":
        mode_tag = f"recumap{args.recording_svd_dim}"
    else:
        mode_tag = f"recsvd{args.recording_svd_dim}"
    cca_tag = "cca" if args.cca_mode == "thin" else args.cca_mode
    pool_tag = "recording_stats" if args.recording_level_stats_pool else f"{args.pool_mode}_{args.pool_layout}_w{args.pool_window}_h{args.pool_hop}"
    norm_tag = "_blocknorm" if args.cca_block_normalize else ""
    if args.cca_block_normalize and float(args.recording_cca_weight) != 1.0:
        weight = str(args.recording_cca_weight).replace(".", "p")
        norm_tag = f"{norm_tag}_recw{weight}"
    rep_name = f"songmae_pool_{pool_tag}_maxpts{args.max_points}_{mode_tag}_{cca_tag}{features.shape[1]}{norm_tag}"

    feature_path = out_dir / f"{rep_name}_features.npz"
    np.savez_compressed(
        feature_path,
        features=features,
        bird_labels=bird_labels.astype(object, copy=False),
        syllable_labels=syllable_labels.astype(np.int64, copy=False),
        recording_labels=recording_labels.astype(object, copy=False),
        songmae_cca=songmae_cca.astype(np.float32, copy=False),
        recording_cca=recording_cca.astype(np.float32, copy=False),
    )

    xy = _fit_umap(
        features,
        neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=None,
        negative_sample_rate=args.umap_negative_sample_rate,
    )
    hdbscan_args = SimpleNamespace(
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        species_display_name=args.species_display_name or _display_name(args.species),
    )
    hdbscan_summary = _hdbscan_umap_analysis(
        xy=xy,
        bird_labels=bird_labels,
        syllable_labels=syllable_labels,
        recording_labels=recording_labels,
        out_dir=out_dir,
        rep_name=rep_name,
        args=hdbscan_args,
    )
    hdbscan_points = np.load(out_dir / f"{rep_name}_hdbscan_points.npz", allow_pickle=True)
    clusters = hdbscan_points["hdbscan_clusters"]
    non_noise = clusters >= 0
    ari = adjusted_rand_score(bird_labels, clusters)
    ari_non_noise = adjusted_rand_score(bird_labels[non_noise], clusters[non_noise]) if non_noise.any() else None
    recording_cluster_spectrogram_path = _write_recording_cluster_spectrograms(
        out_dir,
        rep_name,
        args.spec_dir,
        bird_labels,
        recording_labels,
        clusters,
    )
    plot_title = args.species_display_name or _display_name(args.species)
    cluster_labels = np.asarray([f"cluster_{int(label)}" if int(label) >= 0 else "noise" for label in clusters], dtype=object)
    _scatter_cca_umap(xy, cluster_labels, plot_title, out_dir / f"{rep_name}_hdbscan")
    _scatter_cca_umap(xy, bird_labels, plot_title, out_dir / rep_name)
    _scatter_cca_syllables(xy, syllable_labels, bird_labels, plot_title, out_dir / f"{rep_name}_syllable")
    silhouette_scores = _umap_silhouette_scores(xy, bird_labels, syllable_labels, args.silhouette_sample_size, args.seed)

    summary = {
        "species": args.species,
        "representation": rep_name,
        "feature_path": str(feature_path),
        "points": int(features.shape[0]),
        "songmae_dim": int(songmae_features.shape[1]),
        "recording_feature_mode": args.recording_feature_mode,
        "recording_view_dim": int(recording_view.shape[1]),
        "recording_svd_dim": int(recording_view.shape[1]),
        "recording_level_stats_pool": bool(args.recording_level_stats_pool),
        "cca_components_per_view": int(features.shape[1] // 2),
        "cca_mode": args.cca_mode,
        "cca_ridge_b": cca.get("ridge_b"),
        "cca_ridge_cv": cca.get("ridge_cv"),
        "cca_fit_recordings": int(recording_songmae.shape[0]),
        "canonical_correlations": cca["correlations"],
        "args": vars(args),
        "feature_postprocess": feature_postprocess,
        "silhouette_scores": silhouette_scores,
        "hdbscan_summary": hdbscan_summary,
        "recording_cluster_spectrogram_path": recording_cluster_spectrogram_path,
        "hdbscan_adjusted_rand_index": float(ari),
        "hdbscan_adjusted_rand_index_non_noise": None if ari_non_noise is None else float(ari_non_noise),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
