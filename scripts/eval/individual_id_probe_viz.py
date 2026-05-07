#!/usr/bin/env python3
import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "individual_id"))

import extract_embedding
from individual_identification_linear_probe import (
    _apply_train_feature_postprocess,
    _build_recording_splits,
    _build_split_matrix,
    _load_patch_width,
    _load_recording_stems_by_bird,
)


def _args_from_summary(summary_path, max_birds, songs_per_bird):
    summary = json.loads(Path(summary_path).read_text())
    model = summary["model"]
    saved = summary["args"]
    assert model["encoder"] == "SongMAE", "This visualization currently supports SongMAE probes only."
    assert saved.get("feature_postprocess", "none") == "none", "PCA/whitened probes cannot be mapped back to patches."

    defaults = {
        "encoder": "SongMAE",
        "feature_postprocess": "none",
        "feature_postprocess_dim": 256,
        "normalization_preset": None,
        "audio_params_stats_dir": None,
        "spec_normalization": "none",
        "spec_normalization_stats_dir": None,
        "wav_manifest": None,
        "wav_exts": ".wav,.flac,.ogg,.mp3",
        "audio_context_seconds": 2.0,
        "train_audio_speed_min_pct": 0.0,
        "train_audio_speed_max_pct": 0.0,
        "encoder_layer_idx": None,
        "window_mean_pool": False,
        "window_concat_pool": False,
        "window_token_probe": False,
    }
    defaults.update(saved)
    defaults.update(
        {
            "encoder": "SongMAE",
            "run_dir": model["run_dir"],
            "checkpoint": model.get("checkpoint") or None,
            "annotation_json": saved["annotation_json"],
            "spec_dir": saved["spec_dir"],
            "out_dir": str(Path(summary_path).parent),
            "max_birds": int(max_birds) if max_birds is not None else int(saved.get("max_birds", 0)),
            "songs_per_bird": int(songs_per_bird) if songs_per_bird is not None else int(saved["songs_per_bird"]),
        }
    )
    return Namespace(**defaults)


def _extract_recording(args, model_state, bird_id, recording_stem):
    return extract_embedding.extract_recording_embeddings_with_state(
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
        },
        model_state,
    )


def _class_contribution_grid(features, pipeline, class_idx, patch_rows, score_mode):
    scaler = pipeline.named_steps["standardscaler"]
    clf = pipeline.named_steps["logisticregression"]
    assert clf.coef_.shape[0] > 1, "Expected a multiclass individual-id probe."
    assert features.shape[1] == clf.coef_.shape[1]
    assert features.shape[1] % patch_rows == 0

    scaled = (features - scaler.mean_) / scaler.scale_
    hidden_dim = features.shape[1] // patch_rows
    if score_mode == "raw":
        contrib = scaled * clf.coef_[class_idx]
        grid = contrib.reshape(features.shape[0], patch_rows, hidden_dim).sum(axis=2).T
    else:
        contrib = scaled[:, None, :] * clf.coef_[None, :, :]
        grids = contrib.reshape(features.shape[0], clf.coef_.shape[0], patch_rows, hidden_dim).sum(axis=3)
        grids = grids.transpose(1, 2, 0)
        others = np.delete(grids, class_idx, axis=0)
        grid = grids[class_idx] - others.max(axis=0)

    if score_mode.endswith("_row_zscore"):
        center = np.median(grid, axis=1, keepdims=True)
        scale = np.median(np.abs(grid - center), axis=1, keepdims=True) * 1.4826
        scale = np.maximum(scale, 1e-6)
        grid = (grid - center) / scale
    return grid


def _spec_image(segment, token_count, patch_width):
    timebins = token_count * patch_width
    spec = segment["spectrograms"][:timebins]
    return spec.T


def _save_overlay(spec, grid, patch_height, patch_width, out_path, title, alpha):
    image = np.repeat(np.repeat(grid, patch_height, axis=0), patch_width, axis=1)
    spec = spec[:, : image.shape[1]]
    positive = np.maximum(image, 0.0)
    vmax = float(np.percentile(positive, 99.0))
    if vmax <= 0.0:
        vmax = 1.0
    overlay_alpha = np.clip(positive / vmax, 0.0, 1.0) * alpha

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(spec, origin="lower", aspect="auto", cmap="gray")
    im = ax.imshow(
        positive,
        origin="lower",
        aspect="auto",
        cmap="Reds",
        alpha=overlay_alpha,
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _top_patches(grid, count):
    flat = grid.reshape(-1)
    order = np.argsort(flat)[::-1][:count]
    patches = []
    for rank, index in enumerate(order, start=1):
        patch_y, patch_x = np.unravel_index(int(index), grid.shape)
        patches.append(
            {
                "rank": rank,
                "patch_y": int(patch_y),
                "patch_x": int(patch_x),
                "score": float(grid[patch_y, patch_x]),
            }
        )
    return patches


def _draw_patch(ax, patch, patch_height, patch_width, color, label=None, linewidth=2.0):
    x0 = patch["patch_x"] * patch_width
    y0 = patch["patch_y"] * patch_height
    ax.add_patch(
        Rectangle(
            (x0, y0),
            patch_width,
            patch_height,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
        )
    )
    if label is not None:
        ax.text(
            x0 + 1,
            y0 + patch_height - 2,
            label,
            color=color,
            fontsize=9,
            weight="bold",
            va="top",
        )


def _save_top_patch_plot(spec, patches, patch_height, patch_width, out_path, title):
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(spec, origin="lower", aspect="auto", cmap="gray")
    for patch in patches:
        color = "yellow" if patch["rank"] == 1 else "deeppink"
        label = str(patch["rank"]) if patch["rank"] <= 8 else None
        linewidth = 3.0 if patch["rank"] == 1 else 1.8
        _draw_patch(ax, patch, patch_height, patch_width, color, label=label, linewidth=linewidth)
    best = patches[0]
    ax.set_title(f"{title} | strongest patch={best['patch_y']},{best['patch_x']} score={best['score']:.3f}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _save_best_patch_zoom(spec, best, patch_height, patch_width, out_path, title):
    x0 = best["patch_x"] * patch_width
    y0 = best["patch_y"] * patch_height
    pad_x = 12 * patch_width
    pad_y = patch_height
    left = max(0, x0 - pad_x)
    right = min(spec.shape[1], x0 + patch_width + pad_x)
    bottom = max(0, y0 - pad_y)
    top = min(spec.shape[0], y0 + patch_height + pad_y)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(spec[bottom:top, left:right], origin="lower", aspect="auto", cmap="gray")
    ax.add_patch(
        Rectangle(
            (x0 - left, y0 - bottom),
            patch_width,
            patch_height,
            linewidth=3.0,
            edgecolor="yellow",
            facecolor="none",
        )
    )
    ax.set_title(f"{title} | strongest patch score={best['score']:.3f}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Overlay individual-id linear-probe evidence on SongMAE spectrogram patches.")
    parser.add_argument("--linear_summary", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_birds", type=int, default=None)
    parser.add_argument("--songs_per_bird", type=int, default=None)
    parser.add_argument("--max_viz_recordings", type=int, default=6)
    parser.add_argument("--viz_recordings_per_bird", type=int, default=1)
    parser.add_argument("--top_k_patches", type=int, default=12)
    parser.add_argument("--save_top_patches", action="store_true")
    parser.add_argument(
        "--score_mode",
        default="margin_row_zscore",
        choices=["raw", "raw_row_zscore", "margin", "margin_row_zscore"],
    )
    parser.add_argument("--alpha", type=float, default=0.55)
    args_cli = parser.parse_args()

    args = _args_from_summary(args_cli.linear_summary, args_cli.max_birds, args_cli.songs_per_bird)
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_state = extract_embedding.load_model_state(
        {
            "run_dir": str(args.run_dir),
            "checkpoint": args.checkpoint,
        }
    )
    args.songmae_input_normalization, args.songmae_input_normalization_stats_dir = extract_embedding.get_native_input_normalization(model_state)

    stems_by_bird = _load_recording_stems_by_bird(Path(args.annotation_json))
    train_recordings, val_recordings = _build_recording_splits(args, stems_by_bird)
    patch_width = _load_patch_width(Path(args.run_dir))

    x_train, y_train_raw, _, _, _ = _build_split_matrix(
        args,
        train_recordings,
        patch_width,
        model_state,
        apply_audio_speed_augmentation=True,
    )
    x_val, y_val_raw, _, _, _ = _build_split_matrix(
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
    y_val = label_encoder.transform(y_val_raw[keep_val])
    x_train, x_val, _ = _apply_train_feature_postprocess(x_train, x_val, args)

    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=args.c, max_iter=args.max_iter, solver="lbfgs"),
    )
    pipeline.fit(x_train, y_train)
    val_accuracy = float(accuracy_score(y_val, pipeline.predict(x_val)))

    patch_height = int(model_state["patch_height"])
    patch_rows = int(model_state["num_patches_height"])
    saved = []
    viz_count = 0
    for bird_id in label_encoder.classes_:
        bird_viz_count = 0
        for recording_stem in val_recordings.get(str(bird_id), []):
            if viz_count >= args_cli.max_viz_recordings:
                break
            if bird_viz_count >= args_cli.viz_recordings_per_bird:
                break
            extracted = _extract_recording(args, model_state, str(bird_id), recording_stem)
            class_idx = int(label_encoder.transform([bird_id])[0])
            for segment_idx, segment in enumerate(extracted["segments"][:1]):
                key = f"encoded_embeddings_{args.songmae_embedding_variant}_pos_removal"
                features = segment[key].astype(np.float32, copy=False)
                grid = _class_contribution_grid(features, pipeline, class_idx, patch_rows, args_cli.score_mode)
                spec = _spec_image(segment, features.shape[0], patch_width)
                safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in recording_stem)
                prefix = out_dir / f"{bird_id}_{safe}_seg{segment_idx:02d}"
                out_path = prefix.with_name(f"{prefix.name}_identity_evidence.png")
                _save_overlay(
                    spec,
                    grid,
                    patch_height,
                    patch_width,
                    out_path,
                    f"{bird_id} {args_cli.score_mode} identity evidence | {recording_stem}",
                    args_cli.alpha,
                )
                saved.append(str(out_path))
                if args_cli.save_top_patches:
                    patches = _top_patches(grid, int(args_cli.top_k_patches))
                    top_path = prefix.with_name(f"{prefix.name}_top_patches.png")
                    zoom_path = prefix.with_name(f"{prefix.name}_strongest_patch_zoom.png")
                    json_path = prefix.with_name(f"{prefix.name}_top_patches.json")
                    title = f"{bird_id} {args_cli.score_mode} identity evidence | {recording_stem}"
                    _save_top_patch_plot(spec, patches, patch_height, patch_width, top_path, title)
                    _save_best_patch_zoom(spec, patches[0], patch_height, patch_width, zoom_path, title)
                    json_path.write_text(json.dumps(patches, indent=2))
                    saved.extend([str(top_path), str(zoom_path), str(json_path)])
                viz_count += 1
                bird_viz_count += 1
        if viz_count >= args_cli.max_viz_recordings:
            break

    summary = {
        "linear_summary": str(Path(args_cli.linear_summary).resolve()),
        "out_dir": str(out_dir.resolve()),
        "val_accuracy": val_accuracy,
        "classes": label_encoder.classes_.tolist(),
        "max_birds": int(args.max_birds),
        "songs_per_bird": int(args.songs_per_bird),
        "score_mode": args_cli.score_mode,
        "save_top_patches": bool(args_cli.save_top_patches),
        "viz_recordings": int(viz_count),
        "saved": saved,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
