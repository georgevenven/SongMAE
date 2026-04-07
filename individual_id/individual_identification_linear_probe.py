#!/usr/bin/env python3

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import extract_embedding
from run_individual_id_umap import (
    _load_patch_width,
    _load_recording_stems_by_bird,
    _load_spec_segments,
    _mean_pool_spectrogram,
    _pick_recordings,
    _pool_embeddings,
    _resolve_run_dir,
)


def _split_recordings(recording_stems, val_fraction, seed, bird_id):
    assert len(recording_stems) >= 2
    bird_hash = int(hashlib.sha1(bird_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed + bird_hash)
    order = rng.permutation(len(recording_stems))
    val_count = max(1, int(round(len(recording_stems) * val_fraction)))
    val_count = min(val_count, len(recording_stems) - 1)
    val_idx = np.sort(order[:val_count])
    train_idx = np.sort(order[val_count:])
    assert train_idx.size > 0
    assert val_idx.size > 0
    train = [recording_stems[i] for i in train_idx]
    val = [recording_stems[i] for i in val_idx]
    return train, val


def _build_recording_splits(args, stems_by_bird):
    bird_ids = sorted(stems_by_bird)
    sampled_recordings = {}
    for bird_id in bird_ids:
        stems = _pick_recordings(
            stems_by_bird[bird_id],
            songs_per_bird=args.songs_per_bird,
            seed=args.seed,
            bird_id=bird_id,
        )
        if len(stems) >= 2:
            sampled_recordings[bird_id] = stems

    valid_birds = sorted(sampled_recordings)
    if args.max_birds > 0:
        valid_birds = valid_birds[: args.max_birds]
    assert len(valid_birds) >= 2, "Need at least two birds with at least two recordings each."
    train_recordings = {}
    val_recordings = {}
    for bird_id in valid_birds:
        train_split, val_split = _split_recordings(
            sampled_recordings[bird_id],
            val_fraction=args.val_fraction,
            seed=args.seed,
            bird_id=bird_id,
        )
        train_recordings[bird_id] = train_split
        val_recordings[bird_id] = val_split
    return train_recordings, val_recordings


def _load_target_stats(stats_dir):
    audio = extract_embedding.load_audio_params(stats_dir)
    return np.float32(audio["mean"]), np.float32(audio["std"])


def _normalize_spec_segments(segments, mode, target_stats=None):
    if mode == "none":
        return segments

    assert mode in {"per_recording_cmvn", "per_recording_cmvn_rescaled_to_target_stats"}
    features = [segment["features"] for segment in segments if segment["features"].shape[1] > 0]
    if not features:
        return segments
    recording = np.concatenate(features, axis=1).astype(np.float32, copy=False)
    mean = recording.mean(axis=1, keepdims=True)
    std = recording.std(axis=1, keepdims=True)
    std = np.maximum(std, 1e-6)

    normalized = []
    for segment in segments:
        features = ((segment["features"] - mean) / std).astype(np.float32, copy=False)
        if mode == "per_recording_cmvn_rescaled_to_target_stats":
            assert target_stats is not None
            target_mean, target_std = target_stats
            features = (features * target_std + target_mean).astype(np.float32, copy=False)
        normalized.append(
            {
                "features": features,
                "labels": segment["labels"],
            }
        )
    return normalized


def _normalize_embedding_features(features, mode):
    if mode == "none":
        return features

    assert mode == "per_recording_zscore"
    assert features.ndim == 2
    assert features.shape[0] > 0
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return ((features - mean) / std).astype(np.float32, copy=False)


def _pool_recording(args, bird_id, recording_stem, patch_width, model_state):
    if args.encoder == "SongMAE":
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
                return np.zeros((0, 0), dtype=np.float32)
            raise
        pooled_parts = []
        embedding_key = f"encoded_embeddings_{args.songmae_embedding_variant}_pos_removal"
        for segment in extracted["segments"]:
            features = _normalize_embedding_features(
                segment[embedding_key],
                args.songmae_feature_normalization,
            )
            pooled = _pool_embeddings(
                features,
                args.pool_window,
                args.pool_mode,
                args.pool_hop,
            )
            if pooled.shape[0] > 0:
                pooled_parts.append(pooled)
    elif args.encoder == "Spec":
        segments = _load_spec_segments(args, bird_id, recording_stem, patch_width)
        target_stats = None
        if args.spec_normalization == "per_recording_cmvn_rescaled_to_target_stats":
            stats_dir = args.spec_normalization_stats_dir or args.spec_dir
            target_stats = _load_target_stats(stats_dir)
        segments = _normalize_spec_segments(segments, args.spec_normalization, target_stats=target_stats)
        pooled_parts = []
        for segment in segments:
            if args.pool_window == 0:
                pooled = segment["features"][:, :: args.pool_hop].T.astype(np.float32, copy=False)
            else:
                pooled = _mean_pool_spectrogram(
                    segment["features"],
                    args.pool_window * patch_width,
                    args.pool_hop * patch_width,
                )
            if pooled.shape[0] > 0:
                pooled_parts.append(pooled)
    else:
        raise SystemExit("encoder=AVES is not implemented yet.")

    if not pooled_parts:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(pooled_parts).astype(np.float32, copy=False)


def _build_split_matrix(args, split_recordings, patch_width, model_state):
    x_parts = []
    y_parts = []
    recording_counts = {}
    example_counts = {}

    for bird_id in sorted(split_recordings):
        bird_recordings = split_recordings[bird_id]
        recording_counts[bird_id] = len(bird_recordings)
        example_counts[bird_id] = 0
        for recording_stem in bird_recordings:
            pooled = _pool_recording(args, bird_id, recording_stem, patch_width, model_state)
            if pooled.shape[0] == 0:
                continue
            x_parts.append(pooled)
            y_parts.extend([bird_id] * pooled.shape[0])
            example_counts[bird_id] += int(pooled.shape[0])

    assert x_parts, "No pooled examples were extracted."
    return (
        np.vstack(x_parts),
        np.asarray(y_parts, dtype=object),
        recording_counts,
        example_counts,
    )


def _plot_confusion(y_true, y_pred, class_names, out_base):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)), normalize="true")
    fig = plt.figure(figsize=(9, 8), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    image = ax.imshow(cm, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Validation Confusion Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _apply_normalization_preset(args):
    if args.normalization_preset is None:
        return

    stats_dir = args.audio_params_stats_dir
    if args.encoder == "SongMAE":
        assert args.normalization_preset in {"vanilla", "zscore", "zscore_rescaled"}
        args.songmae_feature_normalization = "none"
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
    parser = argparse.ArgumentParser(description="Train an individual-identification linear probe on pooled per-recording features.")
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
    parser.add_argument("--encoder_layer_idx", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--normalization_preset", choices=["vanilla", "zscore", "zscore_rescaled"], default=None)
    parser.add_argument("--audio_params_stats_dir", default=None)
    parser.add_argument(
        "--spec_normalization",
        choices=["none", "per_recording_cmvn", "per_recording_cmvn_rescaled_to_target_stats"],
        default="none",
    )
    parser.add_argument("--spec_normalization_stats_dir", default=None)
    parser.add_argument("--songmae_embedding_variant", choices=["before", "after"], default="after")
    parser.add_argument("--songmae_feature_normalization", choices=["none", "per_recording_zscore"], default="none")
    parser.add_argument(
        "--songmae_input_normalization",
        choices=["none", "per_model_input_zscore", "per_recording_cmvn_rescaled_to_target_stats"],
        default="none",
    )
    parser.add_argument("--songmae_input_normalization_stats_dir", default=None)
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

    assert annotation_json.exists(), f"annotation_json not found: {annotation_json}"
    assert spec_dir.is_dir(), f"spec_dir not found: {spec_dir}"
    assert 0.0 < args.val_fraction < 1.0
    assert args.pool_hop > 0
    if args.encoder == "SongMAE":
        assert args.pool_window > 0
    else:
        assert args.pool_window >= 0

    if args.encoder == "Spec":
        assert args.pool_mode == "mean", "Spec uses mean pooling only."

    _apply_normalization_preset(args)

    out_dir.mkdir(parents=True, exist_ok=True)
    patch_width = _load_patch_width(run_dir)
    model_state = None
    if args.encoder == "SongMAE":
        model_state = extract_embedding.load_model_state(
            {
                "run_dir": str(run_dir),
                "checkpoint": args.checkpoint,
            }
        )

    stems_by_bird = _load_recording_stems_by_bird(annotation_json)
    train_recordings, val_recordings = _build_recording_splits(args, stems_by_bird)

    x_train, y_train_raw, train_recording_counts, train_example_counts = _build_split_matrix(
        args,
        train_recordings,
        patch_width,
        model_state,
    )
    x_val, y_val_raw, val_recording_counts, val_example_counts = _build_split_matrix(
        args,
        val_recordings,
        patch_width,
        model_state,
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    keep_val = np.isin(y_val_raw, label_encoder.classes_)
    x_val = x_val[keep_val]
    y_val_raw = y_val_raw[keep_val]
    assert x_val.shape[0] > 0, "Validation split has no examples for the trained classes."
    y_val = label_encoder.transform(y_val_raw)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=args.c,
            max_iter=args.max_iter,
            solver="lbfgs",
        ),
    )
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)

    train_accuracy = float(accuracy_score(y_train, train_pred))
    val_accuracy = float(accuracy_score(y_val, val_pred))
    val_macro_f1 = float(f1_score(y_val, val_pred, average="macro"))

    print(
        f"[linear_probe] train_examples={x_train.shape[0]} val_examples={x_val.shape[0]} "
        f"train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f} val_macro_f1={val_macro_f1:.4f}"
    )

    _plot_confusion(y_val, val_pred, label_encoder.classes_.tolist(), out_dir / "val_confusion_matrix")

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
            "encoder_layer_idx": args.encoder_layer_idx,
            "val_fraction": float(args.val_fraction),
            "c": float(args.c),
            "max_iter": int(args.max_iter),
            "normalization_preset": args.normalization_preset,
            "audio_params_stats_dir": args.audio_params_stats_dir,
            "spec_normalization": args.spec_normalization,
            "spec_normalization_stats_dir": args.spec_normalization_stats_dir,
            "songmae_embedding_variant": args.songmae_embedding_variant,
            "songmae_feature_normalization": args.songmae_feature_normalization,
            "songmae_input_normalization": args.songmae_input_normalization,
            "songmae_input_normalization_stats_dir": args.songmae_input_normalization_stats_dir,
        },
    }
    metrics = {
        "birds": label_encoder.classes_.tolist(),
        "train_examples": int(x_train.shape[0]),
        "val_examples": int(x_val.shape[0]),
        "feature_dim": int(x_train.shape[1]),
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "train_recordings_per_bird": train_recording_counts,
        "val_recordings_per_bird": val_recording_counts,
        "train_examples_per_bird": train_example_counts,
        "val_examples_per_bird": val_example_counts,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
