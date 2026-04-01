#!/usr/bin/env python3

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap

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


def _pool_embeddings(embeddings, window, mode, hop):
    assert embeddings.ndim == 2
    if embeddings.shape[0] == 0:
        return np.zeros((0, embeddings.shape[1]), dtype=np.float32)
    if window <= 1:
        return embeddings.astype(np.float32, copy=False)

    pooled = []
    for start in range(0, embeddings.shape[0] - window + 1, hop):
        chunk = embeddings[start : start + window]
        if mode == "max":
            pooled.append(chunk.max(axis=0))
        elif mode == "sum":
            pooled.append(chunk.sum(axis=0))
        else:
            pooled.append(chunk.mean(axis=0))

    if not pooled:
        chunk = embeddings
        if mode == "max":
            pooled.append(chunk.max(axis=0))
        elif mode == "sum":
            pooled.append(chunk.sum(axis=0))
        else:
            pooled.append(chunk.mean(axis=0))

    return np.asarray(pooled, dtype=np.float32)


def _pool_labels(labels, window, hop):
    assert labels.ndim == 1
    if labels.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if window <= 1:
        return labels.astype(np.int64, copy=False)

    pooled = []
    for start in range(0, labels.shape[0] - window + 1, hop):
        chunk = labels[start : start + window]
        values, counts = np.unique(chunk, return_counts=True)
        pooled.append(int(values[np.argmax(counts)]))

    if not pooled:
        values, counts = np.unique(labels, return_counts=True)
        pooled.append(int(values[np.argmax(counts)]))

    return np.asarray(pooled, dtype=np.int64)


def _mean_pool_spectrogram(spec, window_bins, hop_bins):
    assert spec.ndim == 2
    if spec.shape[1] == 0:
        return np.zeros((0, spec.shape[0]), dtype=np.float32)

    pooled = []
    if spec.shape[1] < window_bins:
        pooled.append(spec.mean(axis=1))
        return np.asarray(pooled, dtype=np.float32)

    for start in range(0, spec.shape[1] - window_bins + 1, hop_bins):
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


def _scatter_umap(xy, labels, title, out_base):
    birds = sorted(set(labels.tolist()))
    palette = _bird_palette(birds)

    fig = plt.figure(figsize=(9.5, 7.5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    for bird in birds:
        idx = labels == bird
        ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=10,
            alpha=0.1,
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
            alpha=0.1,
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
    fig = plt.figure(figsize=(9.5, 7.5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=10,
        alpha=0.1,
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
        if args.encoder == "SongMAE":
            features, _, _ = _build_songmae_representation(
                per_bird_segments=single_bird,
                pool_window=args.pool_window,
                pool_hop=args.pool_hop,
                pool_mode=args.pool_mode,
            )
        else:
            features, _, _ = _build_spec_representation(
                per_bird_segments=single_bird,
                pool_window=args.pool_window,
                pool_hop=args.pool_hop,
                patch_width=patch_width,
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


def _load_songmae_segments(args, bird_id, recording_stem, model_state):
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
    for segment in normalized_segments:
        spec = segment["spectrogram"]
        labels = segment["labels_original"]
        count = min(spec.shape[1], labels.shape[0])
        if count == 0:
            continue
        segments.append(
            {
                "features": spec[:, :count],
                "labels": labels[:count],
            }
        )
    return segments


def _build_songmae_representation(per_bird_segments, pool_window, pool_hop, pool_mode):
    x_parts = []
    y_parts = []
    s_parts = []

    for bird_id in sorted(per_bird_segments):
        pooled_parts = []
        label_parts = []
        for segment in per_bird_segments[bird_id]:
            pooled = _pool_embeddings(segment["features"], pool_window, pool_mode, pool_hop)
            pooled_labels = _pool_labels(segment["labels"], pool_window, pool_hop)
            count = min(pooled.shape[0], pooled_labels.shape[0])
            if count == 0:
                continue
            pooled_parts.append(pooled[:count])
            label_parts.append(pooled_labels[:count])

        if not pooled_parts:
            continue

        bird_features = np.vstack(pooled_parts)
        bird_labels = np.concatenate(label_parts, axis=0)
        x_parts.append(bird_features)
        y_parts.extend([bird_id] * bird_features.shape[0])
        s_parts.append(bird_labels)

    assert x_parts, "No valid SongMAE segments were pooled."
    return (
        np.vstack(x_parts),
        np.asarray(y_parts, dtype=object),
        np.concatenate(s_parts, axis=0),
    )


def _build_spec_representation(per_bird_segments, pool_window, pool_hop, patch_width):
    window_bins = pool_window * patch_width
    hop_bins = pool_hop * patch_width
    x_parts = []
    y_parts = []
    s_parts = []

    for bird_id in sorted(per_bird_segments):
        pooled_parts = []
        label_parts = []
        for segment in per_bird_segments[bird_id]:
            pooled = _mean_pool_spectrogram(segment["features"], window_bins, hop_bins)
            pooled_labels = _pool_labels(segment["labels"], window_bins, hop_bins)
            count = min(pooled.shape[0], pooled_labels.shape[0])
            if count == 0:
                continue
            pooled_parts.append(pooled[:count])
            label_parts.append(pooled_labels[:count])

        if not pooled_parts:
            continue

        bird_features = np.vstack(pooled_parts)
        bird_labels = np.concatenate(label_parts, axis=0)
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


def _apply_normalization_preset(args):
    if args.normalization_preset is None:
        return

    stats_dir = args.audio_params_stats_dir
    if args.encoder == "SongMAE":
        assert args.normalization_preset in {"vanilla", "zscore", "zscore_rescaled"}
        if args.normalization_preset == "vanilla":
            args.songmae_input_normalization = "none"
            return
        if args.normalization_preset == "zscore":
            args.songmae_input_normalization = "per_model_input_zscore"
            return
        args.songmae_input_normalization = "per_recording_cmvn_rescaled_to_target_stats"
        args.songmae_input_normalization_stats_dir = stats_dir
        return

    assert args.encoder == "Spec"
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
    parser.add_argument("--pool_window", type=int, required=True)
    parser.add_argument("--pool_hop", type=int, required=True)
    parser.add_argument("--pool_mode", default="mean", choices=["mean", "max", "sum"])
    parser.add_argument("--per_bird_umaps", action="store_true")
    parser.add_argument("--encoder_layer_idx", type=int, default=None)
    parser.add_argument("--normalization_preset", choices=["vanilla", "zscore", "zscore_rescaled"], default=None)
    parser.add_argument("--audio_params_stats_dir", default=None)
    parser.add_argument(
        "--spec_normalization",
        choices=["none", "per_recording_cmvn", "per_recording_cmvn_rescaled_to_target_stats"],
        default="none",
    )
    parser.add_argument("--spec_normalization_stats_dir", default=None)
    parser.add_argument(
        "--songmae_input_normalization",
        choices=["none", "per_model_input_zscore", "per_recording_cmvn", "per_recording_cmvn_rescaled_to_target_stats"],
        default="none",
    )
    parser.add_argument("--songmae_input_normalization_stats_dir", default=None)
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
    if args.songmae_input_normalization_stats_dir is not None:
        args.songmae_input_normalization_stats_dir = str(Path(args.songmae_input_normalization_stats_dir).resolve())

    assert annotation_json.exists(), f"annotation_json not found: {annotation_json}"
    assert spec_dir.is_dir(), f"spec_dir not found: {spec_dir}"
    assert args.pool_window > 0
    assert args.pool_hop > 0
    if args.encoder == "AVES":
        raise SystemExit("encoder=AVES is not implemented yet.")

    if args.encoder == "Spec":
        assert args.pool_mode == "mean", "Spec uses mean pooling only."

    _apply_normalization_preset(args)

    out_dir.mkdir(parents=True, exist_ok=True)
    patch_width = _load_patch_width(run_dir)
    model_state = None
    if args.encoder == "SongMAE":
        model_state = extract_embedding.load_model_state(
            {
                "run_dir": str(args.run_dir),
                "checkpoint": args.checkpoint,
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

    per_bird_segments = {}
    for bird_id in sorted(sampled_recordings):
        bird_segments = []
        for recording_stem in sampled_recordings[bird_id]:
            if args.encoder == "SongMAE":
                loaded_segments = _load_songmae_segments(args, bird_id, recording_stem, model_state)
            else:
                loaded_segments = _load_spec_segments(args, bird_id, recording_stem, patch_width)
            bird_segments.extend(loaded_segments)
        if bird_segments:
            per_bird_segments[bird_id] = bird_segments

    assert per_bird_segments, "No valid segments were extracted."

    if args.encoder == "SongMAE":
        features, bird_labels, syllable_labels = _build_songmae_representation(
            per_bird_segments=per_bird_segments,
            pool_window=args.pool_window,
            pool_hop=args.pool_hop,
            pool_mode=args.pool_mode,
        )
        rep_name = f"songmae_pool_{args.pool_mode}_w{args.pool_window}_h{args.pool_hop}"
    else:
        features, bird_labels, syllable_labels = _build_spec_representation(
            per_bird_segments=per_bird_segments,
            pool_window=args.pool_window,
            pool_hop=args.pool_hop,
            patch_width=patch_width,
        )
        rep_name = f"spec_pool_mean_w{args.pool_window}_h{args.pool_hop}"

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
            "per_bird_umaps": bool(args.per_bird_umaps),
            "normalization_preset": args.normalization_preset,
            "audio_params_stats_dir": args.audio_params_stats_dir,
            "spec_normalization": args.spec_normalization,
            "spec_normalization_stats_dir": args.spec_normalization_stats_dir,
            "songmae_input_normalization": args.songmae_input_normalization,
            "songmae_input_normalization_stats_dir": args.songmae_input_normalization_stats_dir,
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
