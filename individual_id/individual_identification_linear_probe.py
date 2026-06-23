#!/usr/bin/env python3

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.core import extract_embedding
from run_individual_id_umap import (
    _load_aves_segments,
    _load_bird_mae_segments,
    _load_hubert_segments,
    _load_patch_width,
    _load_recording_stems_by_bird,
    _load_spec_segments,
    _mean_pool_spectrogram,
    _pick_recordings,
    _pool_embeddings,
    _resolve_run_dir,
)
from src.external_models import aves
from src.external_models import bird_mae
from src.external_models import hubert
try:
    from src.external_models import old_perch as perch
except ImportError:
    from src.external_models import perch2 as perch

# Perch uses a separate TensorFlow stack. Run this script with:
# /home/george-vengrovski/anaconda3/envs/perch/bin/python


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


def _audio_speed_args(args, apply_audio_speed_augmentation):
    if not apply_audio_speed_augmentation:
        return {
            "train_audio_speed_min_pct": 0.0,
            "train_audio_speed_max_pct": 0.0,
        }
    return {
        "train_audio_speed_min_pct": args.train_audio_speed_min_pct,
        "train_audio_speed_max_pct": args.train_audio_speed_max_pct,
    }


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


def _mean_pool_segment(features):
    assert features.ndim == 2
    assert features.shape[0] > 0
    return features.mean(axis=0, keepdims=True).astype(np.float32, copy=False)


def _mean_pool_songmae_windows(features, tokens_per_window):
    assert features.ndim == 2
    assert tokens_per_window > 0
    pooled = []
    for start in range(0, features.shape[0], tokens_per_window):
        chunk = features[start : start + tokens_per_window]
        if chunk.shape[0] == 0:
            continue
        pooled.append(chunk.mean(axis=0))
    return np.asarray(pooled, dtype=np.float32)


def _concat_fixed_windows(features, tokens_per_window):
    assert features.ndim == 2
    assert tokens_per_window > 0
    rows = []
    feature_dim = int(features.shape[1])
    for start in range(0, features.shape[0], tokens_per_window):
        chunk = features[start : start + tokens_per_window]
        if chunk.shape[0] == 0:
            continue
        if chunk.shape[0] < tokens_per_window:
            pad = np.zeros((tokens_per_window - chunk.shape[0], feature_dim), dtype=np.float32)
            chunk = np.vstack([chunk, pad])
        else:
            chunk = chunk[:tokens_per_window]
        rows.append(chunk.reshape(1, -1))
    if not rows:
        return np.zeros((0, tokens_per_window * feature_dim), dtype=np.float32)
    return np.vstack(rows).astype(np.float32, copy=False)


def _token_windows(features, tokens_per_window, group_prefix):
    assert features.ndim == 2
    assert tokens_per_window > 0
    x_parts = []
    group_ids = []
    window_count = 0
    for start in range(0, features.shape[0], tokens_per_window):
        chunk = features[start : start + tokens_per_window]
        if chunk.shape[0] == 0:
            continue
        x_parts.append(chunk.astype(np.float32, copy=False))
        group_ids.extend([f"{group_prefix}:{window_count}"] * int(chunk.shape[0]))
        window_count += 1
    if not x_parts:
        return np.zeros((0, features.shape[1]), dtype=np.float32), np.asarray([], dtype=object), 0
    return np.vstack(x_parts).astype(np.float32, copy=False), np.asarray(group_ids, dtype=object), window_count


def _resolve_aves_concat_tokens(args, model_state):
    cached = model_state.get("concat_tokens_per_window")
    if cached is not None:
        return int(cached)
    context_seconds = float(args.audio_context_seconds)
    assert context_seconds > 0.0
    model = model_state["model"]
    device = model_state["device"]
    audio_sr = int(model_state["audio_sr"])
    context_samples = max(1, int(round(context_seconds * audio_sr)))
    wav = torch.zeros((1, context_samples), device=device, dtype=torch.float32)
    lengths = torch.tensor([context_samples], device=device, dtype=torch.long)
    with torch.no_grad():
        _, out_lengths = model._extract_features(wav, lengths)
    token_count = int(out_lengths[0].item())
    token_count = max(1, token_count)
    model_state["concat_tokens_per_window"] = token_count
    return token_count


def _aggregate_group_predictions(group_ids, y_true, pred_probs):
    assert group_ids.shape[0] == y_true.shape[0] == pred_probs.shape[0]
    order = {}
    grouped_indices = []
    for index, group_id in enumerate(group_ids.tolist()):
        if group_id not in order:
            order[group_id] = len(grouped_indices)
            grouped_indices.append([])
        grouped_indices[order[group_id]].append(index)

    agg_true = []
    agg_pred = []
    for indices in grouped_indices:
        indices = np.asarray(indices, dtype=np.int64)
        labels = y_true[indices]
        assert np.all(labels == labels[0])
        agg_true.append(int(labels[0]))
        scores = pred_probs[indices].mean(axis=0)
        agg_pred.append(int(np.argmax(scores)))

    return (
        np.asarray(agg_true, dtype=np.int64),
        np.asarray(agg_pred, dtype=np.int64),
    )


def _aggregate_group_probabilities(group_ids, y_true, pred_probs):
    assert group_ids.shape[0] == y_true.shape[0] == pred_probs.shape[0]
    order = {}
    grouped_indices = []
    for index, group_id in enumerate(group_ids.tolist()):
        if group_id not in order:
            order[group_id] = len(grouped_indices)
            grouped_indices.append([])
        grouped_indices[order[group_id]].append(index)

    agg_true = []
    agg_probs = []
    agg_group_ids = []
    for group_id, indices in zip(order.keys(), grouped_indices):
        indices = np.asarray(indices, dtype=np.int64)
        labels = y_true[indices]
        assert np.all(labels == labels[0])
        agg_true.append(int(labels[0]))
        agg_probs.append(pred_probs[indices].mean(axis=0))
        agg_group_ids.append(group_id)

    return (
        np.asarray(agg_true, dtype=np.int64),
        np.asarray(agg_probs, dtype=np.float32),
        np.asarray(agg_group_ids, dtype=object),
    )


def _multiclass_ovr_auc_present_classes(y_true, pred_probs):
    present = np.unique(y_true)
    if present.size < 2:
        return float("nan")

    sliced = pred_probs[:, present].astype(np.float32, copy=False)
    row_sums = sliced.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    sliced = sliced / row_sums

    remapped = np.searchsorted(present, y_true)
    return float(
        roc_auc_score(
            remapped,
            sliced,
            multi_class="ovr",
            average="macro",
        )
    )


def _pool_recording(
    args,
    bird_id,
    recording_stem,
    patch_width,
    model_state,
    apply_audio_speed_augmentation,
):
    audio_speed_args = _audio_speed_args(args, apply_audio_speed_augmentation)
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
                    "seed": args.seed,
                    "wav_root": args.wav_root,
                    "wav_manifest": args.wav_manifest,
                    "wav_exts": args.wav_exts,
                    **audio_speed_args,
                },
                model_state,
            )
        except ValueError as exc:
            if str(exc) == "No valid patches extracted for the requested recording set.":
                return np.zeros((0, 0), dtype=np.float32), np.asarray([], dtype=object), 0
            raise
        pooled_parts = []
        group_parts = []
        grouped_example_count = 0
        embedding_key = f"encoded_embeddings_{args.songmae_embedding_variant}_pos_removal"
        tokens_per_window = int(extracted["num_patches_time"])
        for segment_index, segment in enumerate(extracted["segments"]):
            features = segment.get(embedding_key)
            if features is None:
                features = segment["encoded_embeddings"]
            if args.window_token_probe:
                pooled, group_ids, grouped_count = _token_windows(
                    features,
                    tokens_per_window,
                    f"{recording_stem}:{segment_index}",
                )
                grouped_example_count += grouped_count
            elif args.window_mean_pool:
                pooled = _mean_pool_songmae_windows(features, tokens_per_window)
            elif args.window_concat_pool:
                pooled = _concat_fixed_windows(features, tokens_per_window)
            else:
                pooled = _pool_embeddings(
                    features,
                    args.pool_window,
                    args.pool_mode,
                    args.pool_hop,
                )
            if pooled.shape[0] > 0:
                pooled_parts.append(pooled)
                if args.window_token_probe:
                    group_parts.append(group_ids)
    elif args.encoder == "Spec":
        segments = _load_spec_segments(args, bird_id, recording_stem, patch_width)
        target_stats = None
        if args.spec_normalization == "per_recording_cmvn_rescaled_to_target_stats":
            stats_dir = args.spec_normalization_stats_dir or args.spec_dir
            target_stats = _load_target_stats(stats_dir)
        segments = _normalize_spec_segments(segments, args.spec_normalization, target_stats=target_stats)
        pooled_parts = []
        group_parts = []
        grouped_example_count = 0
        for segment_index, segment in enumerate(segments):
            if args.window_mean_pool:
                pooled = _mean_pool_segment(segment["features"].T)
            elif args.window_token_probe:
                token_features = _mean_pool_spectrogram(
                    segment["features"],
                    patch_width,
                    patch_width,
                )
                pooled, group_ids, grouped_count = _token_windows(
                    token_features,
                    int(token_features.shape[0]),
                    f"{recording_stem}:{segment_index}",
                )
                grouped_example_count += grouped_count
            elif args.pool_window == 0:
                pooled = segment["features"][:, :: args.pool_hop].T.astype(np.float32, copy=False)
            else:
                pooled = _mean_pool_spectrogram(
                    segment["features"],
                    args.pool_window * patch_width,
                    args.pool_hop * patch_width,
                )
            if pooled.shape[0] > 0:
                pooled_parts.append(pooled)
                if args.window_token_probe:
                    group_parts.append(group_ids)
    elif args.encoder in {"AVES", "HuBERT", "BirdMAE"}:
        aves_args = argparse.Namespace(**vars(args))
        aves_args.train_audio_speed_min_pct = audio_speed_args["train_audio_speed_min_pct"]
        aves_args.train_audio_speed_max_pct = audio_speed_args["train_audio_speed_max_pct"]
        if args.encoder == "AVES":
            segments = _load_aves_segments(aves_args, bird_id, recording_stem, model_state)
        elif args.encoder == "HuBERT":
            segments = _load_hubert_segments(aves_args, bird_id, recording_stem, model_state)
        else:
            segments = _load_bird_mae_segments(aves_args, bird_id, recording_stem, model_state)
        pooled_parts = []
        group_parts = []
        grouped_example_count = 0
        concat_tokens = None
        if args.window_concat_pool:
            concat_tokens = _resolve_aves_concat_tokens(args, model_state)
        for segment_index, segment in enumerate(segments):
            if args.window_token_probe:
                pooled, group_ids, grouped_count = _token_windows(
                    segment["features"],
                    int(segment["features"].shape[0]),
                    f"{recording_stem}:{segment_index}",
                )
                grouped_example_count += grouped_count
            elif args.window_mean_pool:
                pooled = _mean_pool_segment(segment["features"])
            elif args.window_concat_pool:
                pooled = _concat_fixed_windows(segment["features"], concat_tokens)
            else:
                pooled = _pool_embeddings(
                    segment["features"],
                    args.pool_window,
                    args.pool_mode,
                    args.pool_hop,
                )
            if pooled.shape[0] > 0:
                pooled_parts.append(pooled)
                if args.window_token_probe:
                    group_parts.append(group_ids)
    elif args.encoder == "Perch":
        perch_args = {
            "json_path": args.annotation_json,
            "bird": bird_id,
            "recording_stem": recording_stem,
            "recording_mode": args.recording_mode,
            "wav_root": args.wav_root,
            "wav_manifest": args.wav_manifest,
            "wav_exts": args.wav_exts,
            "perch_audio_sr": args.perch_audio_sr,
            "train_audio_speed_min_pct": audio_speed_args["train_audio_speed_min_pct"],
            "train_audio_speed_max_pct": audio_speed_args["train_audio_speed_max_pct"],
            "seed": args.seed,
        }
        extracted = perch.extract_recording_embeddings_with_state(perch_args, model_state)
        pooled_parts = []
        for segment in extracted["segments"]:
            features = segment["features"].astype(np.float32, copy=False)
            if features.shape[0] > 0:
                pooled_parts.append(features)
    else:
        raise SystemExit(f"Unsupported encoder: {args.encoder}")

    if not pooled_parts:
        return np.zeros((0, 0), dtype=np.float32), np.asarray([], dtype=object), 0
    if args.window_token_probe:
        return (
            np.vstack(pooled_parts).astype(np.float32, copy=False),
            np.concatenate(group_parts, axis=0),
            grouped_example_count,
        )
    pooled = np.vstack(pooled_parts).astype(np.float32, copy=False)
    group_ids = np.asarray([f"{recording_stem}:{idx}" for idx in range(int(pooled.shape[0]))], dtype=object)
    return pooled, group_ids, int(pooled.shape[0])


def _build_split_matrix(
    args,
    split_recordings,
    patch_width,
    model_state,
    apply_audio_speed_augmentation,
):
    x_parts = []
    y_parts = []
    group_parts = []
    recording_counts = {}
    example_counts = {}

    for bird_id in sorted(split_recordings):
        bird_recordings = split_recordings[bird_id]
        recording_counts[bird_id] = len(bird_recordings)
        example_counts[bird_id] = 0
        for recording_stem in bird_recordings:
            pooled, group_ids, grouped_count = _pool_recording(
                args,
                bird_id,
                recording_stem,
                patch_width,
                model_state,
                apply_audio_speed_augmentation,
            )
            if pooled.shape[0] == 0:
                continue
            x_parts.append(pooled)
            y_parts.extend([bird_id] * pooled.shape[0])
            group_parts.append(group_ids)
            example_counts[bird_id] += int(grouped_count)

    assert x_parts, "No pooled examples were extracted."
    return (
        np.vstack(x_parts),
        np.asarray(y_parts, dtype=object),
        np.concatenate(group_parts, axis=0),
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


def _apply_train_feature_postprocess(x_train, x_val, args):
    if args.feature_postprocess == "none":
        return (
            x_train.astype(np.float32, copy=False),
            x_val.astype(np.float32, copy=False),
            None,
        )

    transform = extract_embedding.fit_feature_postprocess(
        x_train,
        mode=args.feature_postprocess,
        dim=args.feature_postprocess_dim,
    )
    return (
        extract_embedding.apply_feature_postprocess_transform(x_train, transform),
        extract_embedding.apply_feature_postprocess_transform(x_val, transform),
        transform,
    )


class _ThreeLayerMlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def _fit_logistic_regression(x_train, y_train, x_val, args):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=args.c,
            max_iter=args.max_iter,
            solver="lbfgs",
        ),
    )
    model.fit(x_train, y_train)
    return model.predict(x_train), model.predict(x_val), model.predict_proba(x_val).astype(np.float32, copy=False)


def _mlp_probabilities(model, x, args, device):
    loader = DataLoader(
        torch.from_numpy(x),
        batch_size=int(args.mlp_batch_size),
        shuffle=False,
    )
    parts = []
    with torch.no_grad():
        for xb in loader:
            logits = model(xb.to(device, non_blocking=True))
            parts.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(parts).astype(np.float32, copy=False)


def _fit_mlp(x_train, y_train, x_val, args, num_classes):
    torch.manual_seed(int(args.seed))
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32, copy=False)
    x_val = scaler.transform(x_val).astype(np.float32, copy=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _ThreeLayerMlp(
        input_dim=int(x_train.shape[1]),
        hidden_dim=int(args.mlp_hidden_dim),
        num_classes=int(num_classes),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.mlp_lr),
        weight_decay=float(args.mlp_weight_decay),
    )
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train.astype(np.int64, copy=False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.mlp_batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(args.seed)),
    )

    model.train()
    for _ in range(int(args.mlp_epochs)):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    train_pred = _mlp_probabilities(model, x_train, args, device).argmax(axis=1)
    val_probs = _mlp_probabilities(model, x_val, args, device)
    val_pred = val_probs.argmax(axis=1)
    return train_pred, val_pred, val_probs


def _fit_probe(x_train, y_train, x_val, args, num_classes):
    if args.probe_model == "logistic_regression":
        return _fit_logistic_regression(x_train, y_train, x_val, args)
    if args.probe_model == "mlp":
        return _fit_mlp(x_train, y_train, x_val, args, num_classes)
    raise SystemExit(f"Unsupported probe_model: {args.probe_model}")


def main():
    parser = argparse.ArgumentParser(description="Train an individual-identification probe on pooled per-recording features.")
    parser.add_argument("--encoder", required=True, choices=["SongMAE", "Spec", "AVES", "Perch", "HuBERT", "BirdMAE"])
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
    parser.add_argument("--pool_mode", default="mean", choices=["mean"])
    parser.add_argument("--window_mean_pool", action="store_true")
    parser.add_argument("--window_concat_pool", action="store_true")
    parser.add_argument("--window_token_probe", action="store_true")
    parser.add_argument("--encoder_layer_idx", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--probe_model", default="logistic_regression", choices=["logistic_regression", "mlp"])
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--mlp_epochs", type=int, default=100)
    parser.add_argument("--mlp_batch_size", type=int, default=1024)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-4)
    parser.add_argument("--feature_postprocess", default="none", choices=["none", "pca_whiten_l2", "whiten_l2"])
    parser.add_argument("--feature_postprocess_dim", type=int, default=256)
    parser.add_argument("--normalization_preset", choices=["vanilla", "zscore", "zscore_rescaled"], default=None)
    parser.add_argument("--audio_params_stats_dir", default=None)
    parser.add_argument(
        "--spec_normalization",
        choices=["none", "per_recording_cmvn", "per_recording_cmvn_rescaled_to_target_stats"],
        default="none",
    )
    parser.add_argument("--spec_normalization_stats_dir", default=None)
    parser.add_argument("--songmae_embedding_variant", choices=["before", "after"], default="before")
    parser.add_argument("--aves_model_path", default=None)
    parser.add_argument("--aves_config_path", default=None)
    parser.add_argument("--wav_root", default=None)
    parser.add_argument("--wav_manifest", default=None)
    parser.add_argument("--wav_exts", default=".wav,.flac,.ogg,.mp3")
    parser.add_argument("--aves_audio_sr", type=int, default=16000)
    parser.add_argument("--perch_model_name", default="perch_v2")
    parser.add_argument("--perch_audio_sr", type=int, default=32000)
    parser.add_argument("--perch_window_seconds", type=float, default=5.0)
    parser.add_argument("--hubert_model_name", default="facebook/hubert-large-ll60k")
    parser.add_argument("--hubert_audio_sr", type=int, default=16000)
    parser.add_argument("--bird_mae_model_name", default="DBD-research-group/Bird-MAE-Base")
    parser.add_argument("--bird_mae_audio_sr", type=int, default=32000)
    parser.add_argument("--audio_context_seconds", type=float, default=2.0)
    parser.add_argument("--train_audio_speed_min_pct", type=float, default=0.0)
    parser.add_argument("--train_audio_speed_max_pct", type=float, default=0.0)
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
    if args.aves_model_path is not None:
        args.aves_model_path = str(Path(args.aves_model_path).resolve())
    if args.aves_config_path is not None:
        args.aves_config_path = str(Path(args.aves_config_path).resolve())
    if args.wav_root is not None:
        args.wav_root = str(Path(args.wav_root).resolve())
    if args.wav_manifest is not None:
        args.wav_manifest = str(Path(args.wav_manifest).resolve())

    assert annotation_json.exists(), f"annotation_json not found: {annotation_json}"
    assert spec_dir.is_dir(), f"spec_dir not found: {spec_dir}"
    assert 0.0 < args.val_fraction < 1.0
    assert args.audio_context_seconds > 0.0
    assert args.perch_window_seconds > 0.0
    assert args.mlp_hidden_dim > 0
    assert args.mlp_epochs > 0
    assert args.mlp_batch_size > 0
    assert args.mlp_lr > 0.0
    assert args.mlp_weight_decay >= 0.0
    assert 0.0 <= args.train_audio_speed_min_pct <= args.train_audio_speed_max_pct < 1.0
    assert args.pool_hop > 0
    if args.encoder == "SongMAE":
        assert args.pool_window > 0
    else:
        assert args.pool_window >= 0
    if args.window_mean_pool:
        assert args.pool_mode == "mean"
    if args.window_concat_pool:
        assert args.pool_mode == "mean"
    if args.window_token_probe:
        assert args.pool_mode == "mean"
    active_window_modes = int(args.window_mean_pool) + int(args.window_concat_pool) + int(args.window_token_probe)
    assert active_window_modes <= 1, "Choose at most one window aggregation mode."
    if args.train_audio_speed_max_pct > 0.0:
        assert args.encoder in {"SongMAE", "AVES", "Spec", "Perch", "HuBERT", "BirdMAE"}
        assert args.wav_root is not None or args.wav_manifest is not None

    if args.encoder == "Spec":
        assert args.pool_mode == "mean", "Spec uses mean pooling only."
        assert not args.window_concat_pool, "Spec concat pooling is not supported."
    if args.encoder == "Perch":
        assert not args.window_mean_pool, "Perch emits one embedding per fixed window; no pooling needed."
        assert not args.window_concat_pool, "Perch emits one embedding per fixed window; no concat pooling needed."
        assert not args.window_token_probe, "Perch token probing is not supported."
    if args.encoder == "HuBERT":
        assert not args.window_concat_pool, "HuBERT concat pooling is not supported."
    if args.encoder == "BirdMAE":
        assert not args.window_concat_pool, "Bird-MAE concat pooling is not supported."

    _apply_spec_normalization_preset(args)

    out_dir.mkdir(parents=True, exist_ok=True)
    patch_width = 1 if args.encoder in {"AVES", "Perch", "HuBERT", "BirdMAE"} else _load_patch_width(run_dir)
    model_state = None
    args.songmae_input_normalization = None
    args.songmae_input_normalization_stats_dir = None
    if args.encoder == "SongMAE":
        model_state = extract_embedding.load_model_state(str(run_dir), args.checkpoint)
        args.songmae_input_normalization = "audio_params"
        args.songmae_input_normalization_stats_dir = model_state["run_dir"]
    elif args.encoder == "AVES":
        model_state = aves.load_model_state_for_inference(
            {
                "run_dir": str(run_dir),
                "checkpoint": args.checkpoint,
                "encoder_layer_idx": args.encoder_layer_idx,
                "wav_root": args.wav_root,
                "wav_manifest": args.wav_manifest,
                "wav_exts": args.wav_exts,
                "audio_sr": args.aves_audio_sr,
                "audio_context_seconds": args.audio_context_seconds,
                "aves_model_path": args.aves_model_path,
                "aves_config_path": args.aves_config_path,
            }
        )
    elif args.encoder == "Perch":
        model_state = perch.load_model_state_for_inference(
            {
                "run_dir": str(run_dir),
                "wav_root": args.wav_root,
                "wav_manifest": args.wav_manifest,
                "wav_exts": args.wav_exts,
                "perch_model_name": args.perch_model_name,
                "perch_audio_sr": args.perch_audio_sr,
                "perch_window_seconds": args.perch_window_seconds,
            }
        )
    elif args.encoder == "HuBERT":
        model_state = hubert.load_model_state_for_inference(
            {
                "run_dir": str(run_dir),
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
                "run_dir": str(run_dir),
                "wav_root": args.wav_root,
                "wav_manifest": args.wav_manifest,
                "wav_exts": args.wav_exts,
                "bird_mae_model_name": args.bird_mae_model_name,
                "bird_mae_audio_sr": args.bird_mae_audio_sr,
            }
        )

    stems_by_bird = _load_recording_stems_by_bird(annotation_json)
    train_recordings, val_recordings = _build_recording_splits(args, stems_by_bird)

    x_train, y_train_raw, train_group_ids, train_recording_counts, train_example_counts = _build_split_matrix(
        args,
        train_recordings,
        patch_width,
        model_state,
        apply_audio_speed_augmentation=True,
    )
    x_val, y_val_raw, val_group_ids, val_recording_counts, val_example_counts = _build_split_matrix(
        args,
        val_recordings,
        patch_width,
        model_state,
        apply_audio_speed_augmentation=False,
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    keep_val = np.isin(y_val_raw, label_encoder.classes_)
    x_val = x_val[keep_val]
    y_val_raw = y_val_raw[keep_val]
    val_group_ids = val_group_ids[keep_val]
    assert x_val.shape[0] > 0, "Validation split has no examples for the trained classes."
    y_val = label_encoder.transform(y_val_raw)
    x_train, x_val, feature_postprocess = _apply_train_feature_postprocess(x_train, x_val, args)
    train_pred, val_pred, val_probs = _fit_probe(
        x_train,
        y_train,
        x_val,
        args,
        len(label_encoder.classes_),
    )

    train_accuracy = float(accuracy_score(y_train, train_pred))
    probe_label = args.probe_model
    if args.window_token_probe:
        y_val_auc_true, val_auc_probs, val_auc_group_ids = _aggregate_group_probabilities(
            val_group_ids,
            y_val,
            val_probs,
        )
        y_val_mean_true, y_val_mean_pred = _aggregate_group_predictions(
            val_group_ids,
            y_val,
            val_probs,
        )
        val_accuracy = float(accuracy_score(y_val_mean_true, y_val_mean_pred))
        val_macro_f1 = float(f1_score(y_val_mean_true, y_val_mean_pred, average="macro"))
        val_roc_auc_ovr_macro = _multiclass_ovr_auc_present_classes(
            y_val_auc_true,
            val_auc_probs,
        )
        print(
            f"[{probe_label}] train_examples={x_train.shape[0]} val_examples={x_val.shape[0]} "
            f"train_acc={train_accuracy:.4f} meanprob_acc={val_accuracy:.4f} "
            f"meanprob_macro_f1={val_macro_f1:.4f} val_roc_auc_ovr_macro={val_roc_auc_ovr_macro:.4f}"
        )
        _plot_confusion(
            y_val_mean_true,
            y_val_mean_pred,
            label_encoder.classes_.tolist(),
            out_dir / "val_confusion_matrix",
        )
        save_y_true = y_val_auc_true
        save_y_pred = y_val_mean_pred
        save_val_probs = val_auc_probs
        save_group_ids = val_auc_group_ids
    else:
        val_accuracy = float(accuracy_score(y_val, val_pred))
        val_macro_f1 = float(f1_score(y_val, val_pred, average="macro"))
        val_roc_auc_ovr_macro = _multiclass_ovr_auc_present_classes(
            y_val,
            val_probs,
        )
        print(
            f"[{probe_label}] train_examples={x_train.shape[0]} val_examples={x_val.shape[0]} "
            f"train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} val_roc_auc_ovr_macro={val_roc_auc_ovr_macro:.4f}"
        )
        _plot_confusion(y_val, val_pred, label_encoder.classes_.tolist(), out_dir / "val_confusion_matrix")
        save_y_true = y_val
        save_y_pred = val_pred
        save_val_probs = val_probs.astype(np.float32, copy=False)
        save_group_ids = val_group_ids

    if feature_postprocess is not None:
        extract_embedding.save_feature_postprocess(
            out_dir / "feature_postprocess.npz",
            feature_postprocess,
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
            "window_mean_pool": bool(args.window_mean_pool),
            "window_concat_pool": bool(args.window_concat_pool),
            "window_token_probe": bool(args.window_token_probe),
            "encoder_layer_idx": args.encoder_layer_idx,
            "val_fraction": float(args.val_fraction),
            "probe_model": args.probe_model,
            "c": float(args.c),
            "max_iter": int(args.max_iter),
            "mlp_hidden_dim": int(args.mlp_hidden_dim),
            "mlp_epochs": int(args.mlp_epochs),
            "mlp_batch_size": int(args.mlp_batch_size),
            "mlp_lr": float(args.mlp_lr),
            "mlp_weight_decay": float(args.mlp_weight_decay),
            "feature_postprocess": args.feature_postprocess,
            "feature_postprocess_dim": int(args.feature_postprocess_dim),
            "normalization_preset": args.normalization_preset,
            "audio_params_stats_dir": args.audio_params_stats_dir,
            "spec_normalization": args.spec_normalization,
            "spec_normalization_stats_dir": args.spec_normalization_stats_dir,
            "songmae_embedding_variant": args.songmae_embedding_variant,
            "songmae_input_normalization": args.songmae_input_normalization,
            "songmae_input_normalization_stats_dir": args.songmae_input_normalization_stats_dir,
            "aves_model_path": args.aves_model_path,
            "aves_config_path": args.aves_config_path,
            "wav_root": args.wav_root,
            "wav_manifest": args.wav_manifest,
            "wav_exts": args.wav_exts,
            "aves_audio_sr": int(args.aves_audio_sr),
            "perch_model_name": args.perch_model_name,
            "perch_audio_sr": int(args.perch_audio_sr),
            "perch_window_seconds": float(args.perch_window_seconds),
            "hubert_model_name": args.hubert_model_name,
            "hubert_audio_sr": int(args.hubert_audio_sr),
            "bird_mae_model_name": args.bird_mae_model_name,
            "bird_mae_audio_sr": int(args.bird_mae_audio_sr),
            "audio_context_seconds": float(args.audio_context_seconds),
            "train_audio_speed_min_pct": float(args.train_audio_speed_min_pct),
            "train_audio_speed_max_pct": float(args.train_audio_speed_max_pct),
        },
    }
    metrics = {
        "birds": label_encoder.classes_.tolist(),
        "num_classes": int(len(label_encoder.classes_)),
        "train_examples": int(x_train.shape[0]),
        "val_examples": int(x_val.shape[0]),
        "feature_dim": int(x_train.shape[1]),
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "val_roc_auc_ovr_macro": val_roc_auc_ovr_macro,
        "train_recordings_per_bird": train_recording_counts,
        "val_recordings_per_bird": val_recording_counts,
        "train_examples_per_bird": train_example_counts,
        "val_examples_per_bird": val_example_counts,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez(
        out_dir / "val_predictions.npz",
        y_true=save_y_true,
        y_pred=save_y_pred,
        probs=save_val_probs,
        group_ids=save_group_ids,
        class_names=np.asarray(label_encoder.classes_.tolist(), dtype=object),
    )


if __name__ == "__main__":
    main()
