#!/usr/bin/env python3

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

import aves  # noqa: E402
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


def _songmae_feature_key(feature_source):
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
        return embeddings.astype(np.float32, copy=False)
    assert mode == "mean"

    if starts is None:
        starts, short_segment = _window_starts_for_length(embeddings.shape[0], window, hop, layout, seed)
    if short_segment:
        return embeddings.mean(axis=0, keepdims=True).astype(np.float32, copy=False)

    pooled = []
    for start in starts.tolist():
        chunk = embeddings[start : start + window]
        pooled.append(chunk.mean(axis=0))

    if not pooled:
        pooled.append(embeddings.mean(axis=0))

    return np.asarray(pooled, dtype=np.float32)


def _pool_labels(labels, window, hop, layout="sliding", seed=0, starts=None, short_segment=False):
    assert labels.ndim == 1
    if labels.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if window <= 1:
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


def _fit_umap(features, neighbors, min_dist, metric):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(neighbors),
        min_dist=float(min_dist),
        metric=metric,
        low_memory=True,
        n_jobs=-1,
    )
    return reducer.fit_transform(features)


def _bird_palette(birds):
    birds = sorted(set(birds))
    cmap = plt.get_cmap("tab20", max(1, len(birds)))
    palette = {}
    for index, bird in enumerate(birds):
        palette[bird] = np.asarray(cmap(index), dtype=np.float32)[:3]
    return palette


def _point_alpha(num_points):
    low_points = 5000
    high_points = 100000
    low_alpha = 0.35
    high_alpha = 0.1

    if num_points <= low_points:
        return low_alpha
    if num_points >= high_points:
        return high_alpha

    t = (float(num_points) - float(low_points)) / float(high_points - low_points)
    return low_alpha + t * (high_alpha - low_alpha)


def _scatter_umap(xy, labels, title, out_base):
    birds = sorted(set(labels.tolist()))
    palette = _bird_palette(birds)
    alpha = _point_alpha(int(xy.shape[0]))

    fig = plt.figure(figsize=(9.5, 7.5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    for bird in birds:
        idx = labels == bird
        ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=10,
            alpha=alpha,
            color=palette[bird],
            label=bird,
            edgecolors="none",
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
        markerscale=1.6,
    )
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _scatter_umap_syllables(xy, syllables, birds, title, out_base):
    assert syllables.shape[0] == xy.shape[0]
    assert birds.shape[0] == xy.shape[0]
    alpha = _point_alpha(int(xy.shape[0]))

    categories = []
    for bird, syllable in zip(birds.tolist(), syllables.tolist()):
        if int(syllable) < 0:
            categories.append("silence")
        else:
            categories.append(f"{bird}:{int(syllable)}")

    unique = sorted(set(categories))
    non_silence = [label for label in unique if label != "silence"]
    palette = {}
    cmap = plt.get_cmap("gist_ncar", max(1, len(non_silence)))
    for index, label in enumerate(non_silence):
        palette[label] = np.asarray(cmap(index), dtype=np.float32)[:3]
    if "silence" in unique:
        palette["silence"] = np.asarray([0.55, 0.55, 0.55], dtype=np.float32)

    fig = plt.figure(figsize=(9.5, 7.5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    categories_arr = np.asarray(categories, dtype=object)
    for label in unique:
        idx = categories_arr == label
        ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=10,
            alpha=alpha,
            color=palette[label],
            label=label,
            edgecolors="none",
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=7,
        markerscale=1.6,
        ncol=1 if len(unique) <= 30 else 2,
        title="Syllable",
    )
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _scatter_single_umap(xy, title, out_base, color):
    alpha = _point_alpha(int(xy.shape[0]))
    fig = plt.figure(figsize=(9.5, 7.5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=10,
        alpha=alpha,
        color=color,
        edgecolors="none",
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300, format="pdf")
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
            features, _, _ = _build_embedding_representation(
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
            features, _, _ = _build_spec_representation(
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
        )
        out_base = per_bird_dir / f"{bird_id}"
        _scatter_single_umap(
            xy=xy,
            title=f"{args.species} | {bird_id}",
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
                "minimal_output": True,
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
        features = segment[feature_key]
        labels = segment["labels_downsampled"]
        count = min(features.shape[0], labels.shape[0])
        if count == 0:
            continue
        per_bird_segments.setdefault(bird_id, []).append(
            {
                "features": features[:count],
                "labels": labels[:count],
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
        features = segment["encoded_embeddings_before_pos_removal"]
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


def _load_spec_segments(args, bird_id, recording_stem, patch_width):
    if float(getattr(args, "train_audio_speed_max_pct", 0.0)) > 0.0:
        loaded = extract_embedding.load_recording_segments_from_audio(
            {
                "spec_dir": str(args.spec_dir),
                "json_path": str(args.annotation_json),
                "bird": bird_id,
                "recording_stem": recording_stem,
                "recording_mode": args.recording_mode,
                "wav_root": args.wav_root,
                "wav_manifest": args.wav_manifest,
                "wav_exts": args.wav_exts,
                "seed": getattr(args, "seed", 0),
                "train_audio_speed_min_pct": getattr(args, "train_audio_speed_min_pct", 0.0),
                "train_audio_speed_max_pct": getattr(args, "train_audio_speed_max_pct", 0.0),
            },
            patch_width=patch_width,
        )
    else:
        loaded = extract_embedding.load_recording_segments(
            {
                "spec_dir": str(args.spec_dir),
                "json_path": str(args.annotation_json),
                "bird": bird_id,
                "recording_stem": recording_stem,
                "recording_mode": args.recording_mode,
            },
            patch_width=patch_width,
        )
    target_stats = None
    if args.spec_normalization == "per_recording_cmvn_rescaled_to_target_stats":
        stats_dir = args.spec_normalization_stats_dir or args.spec_dir
        target_stats = _load_target_stats(stats_dir)
    normalized_segments = extract_embedding.normalize_recording_segments(
        loaded["segments"],
        args.spec_normalization,
        target_stats=target_stats,
    )

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


def _build_embedding_representation(per_bird_segments, pool_window, pool_hop, pool_mode, pool_layout, seed, pca_dim, max_points=0):
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
                }
            )

    allocations = _allocate_point_budget(candidates, max_points, seed)
    pooled_by_bird = {}
    labels_by_bird = {}
    for candidate, local_indices in zip(candidates, allocations):
        starts = candidate["starts"]
        short_segment = candidate["short_segment"]
        if local_indices is not None:
            if local_indices.size == 0:
                continue
            if not short_segment:
                starts = starts[local_indices]
        if pool_mode == "concat_pca":
            pooled = _concat_window_embeddings(
                candidate["features"],
                pool_window,
                pool_hop,
                layout=pool_layout,
                seed=candidate["seed"],
                starts=starts,
                short_segment=short_segment,
            )
        else:
            pooled = _pool_embeddings(
                candidate["features"],
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
        bird_id = candidate["bird_id"]
        pooled_by_bird.setdefault(bird_id, []).append(pooled[:count])
        labels_by_bird.setdefault(bird_id, []).append(pooled_labels[:count])

    x_parts = []
    y_parts = []
    s_parts = []
    for bird_id in sorted(pooled_by_bird):
        bird_features = np.vstack(_pad_feature_widths(pooled_by_bird[bird_id]))
        bird_labels = np.concatenate(labels_by_bird[bird_id], axis=0)
        x_parts.append(bird_features)
        y_parts.extend([bird_id] * bird_features.shape[0])
        s_parts.append(bird_labels)

    assert x_parts, "No valid embedding segments were pooled."
    features = np.vstack(_pad_feature_widths(x_parts))
    if pool_mode == "concat_pca":
        features = _fit_pca(features, pca_dim)
    return (
        features,
        np.asarray(y_parts, dtype=object),
        np.concatenate(s_parts, axis=0),
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
                }
            )

    allocations = _allocate_point_budget(candidates, max_points, seed)
    pooled_by_bird = {}
    labels_by_bird = {}
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

    x_parts = []
    y_parts = []
    s_parts = []
    for bird_id in sorted(pooled_by_bird):
        bird_features = np.vstack(pooled_by_bird[bird_id])
        bird_labels = np.concatenate(labels_by_bird[bird_id], axis=0)
        x_parts.append(bird_features)
        y_parts.extend([bird_id] * bird_features.shape[0])
        s_parts.append(bird_labels)

    assert x_parts, "No valid spectrogram segments were pooled."
    return (
        np.vstack(x_parts),
        np.asarray(y_parts, dtype=object),
        np.concatenate(s_parts, axis=0),
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


def main():
    parser = argparse.ArgumentParser(description="Individual-ID UMAPs with explicit encoder mode and record-wise pooling.")
    parser.add_argument("--encoder", required=True, choices=["SongMAE", "Spec", "AVES"])
    parser.add_argument("--species", required=True)
    parser.add_argument("--annotation_json", required=True)
    parser.add_argument("--spec_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--recording_mode", required=True, choices=["events", "full_recordings"])
    parser.add_argument("--songs_per_bird", type=int, required=True)
    parser.add_argument("--max_birds", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool_window", type=int, default=50)
    parser.add_argument("--pool_hop", type=int, default=10)
    parser.add_argument("--pool_mode", default="mean", choices=["mean", "concat_pca"])
    parser.add_argument("--concat_pca_dim", type=int, default=256)
    parser.add_argument("--pool_layout", default="sliding", choices=["sliding", "shotgun"])
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--feature_postprocess", default="whiten_l2", choices=["none", "pca_whiten_l2", "whiten_l2"])
    parser.add_argument("--feature_postprocess_dim", type=int, default=256)
    parser.add_argument("--feature_postprocess_load", default=None)
    parser.add_argument("--feature_postprocess_save", default=None)
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
    parser.add_argument("--umap_neighbors", type=int, default=100)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_metric", default="cosine")
    args = parser.parse_args()

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
    patch_width = 1 if args.encoder == "AVES" else _load_patch_width(run_dir)
    model_state = None
    args.songmae_input_normalization = None
    args.songmae_input_normalization_stats_dir = None
    if args.encoder == "SongMAE":
        model_state = extract_embedding.load_model_state(
            {
                "run_dir": str(args.run_dir),
                "checkpoint": args.checkpoint,
            }
        )
        (
            args.songmae_input_normalization,
            args.songmae_input_normalization_stats_dir,
        ) = extract_embedding.get_native_input_normalization(model_state)
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
                else:
                    loaded_segments = _load_spec_segments(args, bird_id, recording_stem, patch_width)
                bird_segments.extend(loaded_segments)
            if bird_segments:
                per_bird_segments[bird_id] = bird_segments

    assert per_bird_segments, "No valid segments were extracted."

    if args.encoder in {"SongMAE", "AVES"}:
        features, bird_labels, syllable_labels = _build_embedding_representation(
            per_bird_segments=per_bird_segments,
            pool_window=args.pool_window,
            pool_hop=args.pool_hop,
            pool_mode=args.pool_mode,
            pool_layout=args.pool_layout,
            seed=args.seed,
            pca_dim=args.concat_pca_dim,
            max_points=args.max_points,
        )
        prefix = "songmae" if args.encoder == "SongMAE" else "aves"
        source_suffix = ""
        if args.encoder == "SongMAE" and args.songmae_feature_source != "encoded_before":
            source_suffix = f"_{args.songmae_feature_source}"
        rep_name = f"{prefix}{source_suffix}_pool_{args.pool_mode}{args.concat_pca_dim if args.pool_mode == 'concat_pca' else ''}_{args.pool_layout}_w{args.pool_window}_h{args.pool_hop}"
    else:
        features, bird_labels, syllable_labels = _build_spec_representation(
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

    if args.encoder != "SongMAE":
        features, feature_postprocess = _apply_feature_postprocess(features, args)
    if feature_postprocess is not None:
        rep_name = f"{rep_name}_{_feature_postprocess_kind(feature_postprocess)}{feature_postprocess['dim']}"

    assert features.shape[0] >= 2, "Not enough points for UMAP."
    print(f"[umap] {rep_name}: points={features.shape[0]} dim={features.shape[1]}")

    xy = _fit_umap(
        features,
        neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
    )

    out_base = out_dir / rep_name
    _scatter_umap(
        xy=xy,
        labels=bird_labels,
        title=f"{args.species} | {rep_name}",
        out_base=out_base,
    )
    _scatter_umap_syllables(
        xy=xy,
        syllables=syllable_labels,
        birds=bird_labels,
        title=f"{args.species} | {rep_name} | syllable",
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
            "encoder_layer_idx": args.encoder_layer_idx,
            "umap_neighbors": int(args.umap_neighbors),
            "umap_min_dist": float(args.umap_min_dist),
            "umap_metric": args.umap_metric,
        },
        "per_bird_umaps": per_bird_saved,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
