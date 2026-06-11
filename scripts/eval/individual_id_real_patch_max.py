#!/usr/bin/env python3
import argparse
import heapq
import json
import sys
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
sys.path.append(str(Path(__file__).resolve().parent))

import extract_embedding
from individual_id_probe_viz import _args_from_summary, _extract_recording
from individual_identification_linear_probe import (
    _apply_train_feature_postprocess,
    _build_recording_splits,
    _build_split_matrix,
    _load_patch_width,
    _load_recording_stems_by_bird,
)


def _train_probe(args, model_state, patch_width):
    stems_by_bird = _load_recording_stems_by_bird(Path(args.annotation_json))
    train_recordings, val_recordings = _build_recording_splits(args, stems_by_bird)
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
    return pipeline, label_encoder, train_recordings, val_recordings, val_accuracy


def _patch_scores(features, pipeline, patch_rows):
    scaler = pipeline.named_steps["standardscaler"]
    clf = pipeline.named_steps["logisticregression"]
    assert features.shape[1] == clf.coef_.shape[1]
    assert features.shape[1] % patch_rows == 0

    hidden = features.shape[1] // patch_rows
    scaled = ((features - scaler.mean_) / scaler.scale_).reshape(features.shape[0], patch_rows, hidden)
    weights = clf.coef_.reshape(clf.coef_.shape[0], patch_rows, hidden)
    return np.einsum("trh,crh->crt", scaled, weights, optimize=True)


def _margin_grid(class_scores, class_idx):
    target = class_scores[class_idx]
    others = np.delete(class_scores, class_idx, axis=0)
    return target - others.max(axis=0)


def _row_zscore(grid):
    center = np.median(grid, axis=1, keepdims=True)
    scale = np.median(np.abs(grid - center), axis=1, keepdims=True) * 1.4826
    scale = np.maximum(scale, 1e-6)
    return (grid - center) / scale


def _maybe_push(heap, item, top_k):
    packed = (float(item["score"]), id(item), item)
    if len(heap) < top_k:
        heapq.heappush(heap, packed)
        return
    if packed[0] > heap[0][0]:
        heapq.heapreplace(heap, packed)


def _scan_segment(targets, heaps, class_scores, segment, context_patches, patch_height, patch_width, top_k, source_bird, recording_stem, segment_idx, row_normalize):
    spec = segment["spectrograms"].T
    for bird_id, class_idx in targets.items():
        grid = _margin_grid(class_scores, class_idx)
        if row_normalize:
            grid = _row_zscore(grid)
        for index in np.argpartition(grid.reshape(-1), -top_k)[-top_k:]:
            patch_y, patch_x = np.unravel_index(int(index), grid.shape)
            x0 = int(patch_x * patch_width)
            y0 = int(patch_y * patch_height)
            left = max(0, x0 - context_patches * patch_width)
            right = min(spec.shape[1], x0 + (context_patches + 1) * patch_width)
            item = {
                "target_bird": bird_id,
                "source_bird": source_bird,
                "recording_stem": recording_stem,
                "segment_idx": int(segment_idx),
                "patch_y": int(patch_y),
                "patch_x": int(patch_x),
                "score": float(grid[patch_y, patch_x]),
                "crop": spec[:, left:right].astype(np.float32, copy=False),
                "crop_left": int(left),
                "patch_x0": int(x0 - left),
                "patch_y0": int(y0),
            }
            _maybe_push(heaps[bird_id], item, top_k)


def _crop_json(item, rank):
    return {
        "rank": int(rank),
        "target_bird": item["target_bird"],
        "source_bird": item["source_bird"],
        "recording_stem": item["recording_stem"],
        "segment_idx": item["segment_idx"],
        "patch_y": item["patch_y"],
        "patch_x": item["patch_x"],
        "score": item["score"],
    }


def _save_collage(items, out_path, patch_height, patch_width, title):
    cols = 4
    rows = int(np.ceil(len(items) / cols))
    fig = plt.figure(figsize=(cols * 4.0, rows * 2.6))
    for plot_idx, item in enumerate(items, start=1):
        ax = fig.add_subplot(rows, cols, plot_idx)
        ax.imshow(item["crop"], origin="lower", aspect="auto", cmap="gray")
        ax.add_patch(
            Rectangle(
                (item["patch_x0"], item["patch_y0"]),
                patch_width,
                patch_height,
                linewidth=2.0,
                edgecolor="yellow",
                facecolor="none",
            )
        )
        ax.set_title(
            f"#{plot_idx} {item['source_bird']} score={item['score']:.2f}\n"
            f"row={item['patch_y']} col={item['patch_x']}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _scan_recordings(args, model_state, recordings, targets, heaps, top_k, context_patches, patch_height, patch_width, patch_rows, row_normalize):
    for source_bird in sorted(recordings):
        for recording_stem in recordings[source_bird]:
            extracted = _extract_recording(args, model_state, source_bird, recording_stem)
            key = f"encoded_embeddings_{args.songmae_embedding_variant}_pos_removal"
            for segment_idx, segment in enumerate(extracted["segments"]):
                features = segment[key].astype(np.float32, copy=False)
                class_scores = _patch_scores(features, args.pipeline, patch_rows)
                _scan_segment(
                    targets,
                    heaps,
                    class_scores,
                    segment,
                    context_patches,
                    patch_height,
                    patch_width,
                    top_k,
                    source_bird,
                    recording_stem,
                    segment_idx,
                    row_normalize,
                )


def main():
    parser = argparse.ArgumentParser(description="Find real spectrogram patches that maximize individual-id probe margins.")
    parser.add_argument("--linear_summary", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target_birds", default="B385,B145")
    parser.add_argument("--scan_split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--context_patches", type=int, default=12)
    parser.add_argument("--row_normalize", action="store_true")
    parser.add_argument("--max_birds", type=int, default=None)
    parser.add_argument("--songs_per_bird", type=int, default=None)
    args_cli = parser.parse_args()

    args = _args_from_summary(args_cli.linear_summary, args_cli.max_birds, args_cli.songs_per_bird)
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_state = extract_embedding.load_model_state(str(args.run_dir), args.checkpoint)
    args.songmae_input_normalization = "audio_params"
    args.songmae_input_normalization_stats_dir = model_state["run_dir"]
    patch_width = _load_patch_width(Path(args.run_dir))
    patch_height = int(model_state["patch_height"])
    patch_rows = int(model_state["num_patches_height"])

    pipeline, label_encoder, train_recordings, val_recordings, val_accuracy = _train_probe(args, model_state, patch_width)
    args.pipeline = pipeline

    recordings = val_recordings
    if args_cli.scan_split == "train":
        recordings = train_recordings
    elif args_cli.scan_split == "all":
        recordings = {bird: train_recordings[bird] + val_recordings[bird] for bird in sorted(train_recordings)}

    targets = {}
    for bird_id in [x.strip() for x in args_cli.target_birds.split(",") if x.strip()]:
        assert bird_id in label_encoder.classes_, f"Unknown target bird: {bird_id}"
        targets[bird_id] = int(label_encoder.transform([bird_id])[0])

    heaps = {bird_id: [] for bird_id in targets}
    _scan_recordings(
        args,
        model_state,
        recordings,
        targets,
        heaps,
        int(args_cli.top_k),
        int(args_cli.context_patches),
        patch_height,
        patch_width,
        patch_rows,
        bool(args_cli.row_normalize),
    )

    summary = {
        "linear_summary": str(Path(args_cli.linear_summary).resolve()),
        "out_dir": str(out_dir.resolve()),
        "target_birds": sorted(targets),
        "scan_split": args_cli.scan_split,
        "top_k": int(args_cli.top_k),
        "context_patches": int(args_cli.context_patches),
        "row_normalize": bool(args_cli.row_normalize),
        "val_accuracy": val_accuracy,
        "saved": [],
    }
    for bird_id, heap in heaps.items():
        items = [packed[2] for packed in sorted(heap, reverse=True)]
        json_items = [_crop_json(item, rank) for rank, item in enumerate(items, start=1)]
        json_path = out_dir / f"{bird_id}_top_real_patches.json"
        png_path = out_dir / f"{bird_id}_top_real_patches.png"
        json_path.write_text(json.dumps(json_items, indent=2))
        _save_collage(items, png_path, patch_height, patch_width, f"{bird_id} top real probe-margin patches")
        summary["saved"].extend([str(json_path), str(png_path)])

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
