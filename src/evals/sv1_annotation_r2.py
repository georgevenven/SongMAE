#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.plotting_utils.sv1_annotation_r2 import plot_metric_bars, plot_sv1_overlay


def npz_paths(path):
    path = Path(path)
    if path.is_file():
        return [path]
    paths = [p for p in sorted(path.glob("*.npz")) if ".tmp." not in p.name]
    assert paths, f"no npz files found: {path}"
    return paths


def parse_dataset(value):
    name, sep, path = value.partition("=")
    assert sep and name and path, "--dataset must look like name=/path/to/embeddings.npz"
    return name, Path(path)


def load_one(path, feature_key):
    data = np.load(path, allow_pickle=True)
    assert feature_key in data, f"{feature_key} missing from {path}"
    assert "labels_downsampled" in data, f"labels_downsampled missing from {path}"
    x = data[feature_key].astype(np.float32, copy=False).reshape(data[feature_key].shape[0], -1)
    y = data["labels_downsampled"].astype(np.int64, copy=False)
    assert x.shape[0] == y.shape[0], f"feature/label length mismatch in {path}"
    return data, x, (y >= 0).astype(np.float32)


def token_spectrogram(data, count):
    if "spectrograms" not in data:
        return None
    spec = data["spectrograms"].astype(np.float32, copy=False)
    if spec.shape[0] == count:
        return spec.T
    patch_width = int(data["patch_width"]) if "patch_width" in data else spec.shape[0] // count
    assert patch_width > 0 and spec.shape[0] >= count * patch_width
    spec = spec[: count * patch_width]
    return spec.reshape(count, patch_width, spec.shape[1]).mean(axis=1).T


def pixel_intensity(data, count):
    spec = token_spectrogram(data, count)
    if spec is None:
        return None
    return spec.mean(axis=0).astype(np.float32, copy=False)


def overlay_items(dataset, path, data, y, offset):
    spec = token_spectrogram(data, y.size)
    if spec is None:
        return []
    stems = np.asarray(data["recording_stem"]).astype(str) if "recording_stem" in data else np.full(y.size, path.stem)
    song_ids = np.asarray(data["song_id"]).astype(str) if "song_id" in data else np.full(y.size, "0")

    items = []
    start = 0
    keys = np.char.add(np.char.add(stems, ":"), song_ids)
    for end in np.r_[np.flatnonzero(keys[1:] != keys[:-1]) + 1, y.size]:
        song = y[start:end]
        if song.min() < song.max():
            items.append(
                {
                    "dataset": dataset,
                    "name": keys[start],
                    "start": offset + start,
                    "end": offset + end,
                    "song": song,
                    "spectrogram": spec[:, start:end],
                }
            )
        start = int(end)
    return items


def load_dataset(name, path, feature_key):
    xs, ys, pixels, items = [], [], [], []
    offset = 0
    for npz_path in npz_paths(path):
        data, x, y = load_one(npz_path, feature_key)
        pixel = pixel_intensity(data, y.size)
        xs.append(x)
        ys.append(y)
        if pixel is not None:
            pixels.append(pixel)
        items.extend(overlay_items(name, npz_path, data, y, offset))
        offset += x.shape[0]
    return {
        "name": name,
        "path": str(path),
        "x": np.concatenate(xs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "pixel": np.concatenate(pixels, axis=0) if pixels else None,
        "files": len(xs),
        "items": items,
    }


def zscore(features):
    mean = features.mean(axis=0, keepdims=True)
    std = np.maximum(features.std(axis=0, keepdims=True), 1e-8)
    return ((features - mean) / std).astype(np.float32, copy=False)


def top_singular_vector(x):
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    values, vectors = np.linalg.eigh(centered.T @ centered)
    index = int(np.argmax(values))
    return mean, vectors[:, index].astype(np.float32, copy=False), float(np.sqrt(values[index]))


def pearson_r2(score, y):
    score = score.astype(np.float64, copy=False) - score.mean()
    y = y.astype(np.float64, copy=False) - y.mean()
    denom = np.sqrt((score @ score) * (y @ y))
    assert denom > 0.0
    r = float((score @ y) / denom)
    return r, r * r


def summarize(name, score, y, files):
    assert y.min() < y.max(), f"need both song and silence labels for {name}"
    r, r2 = pearson_r2(score, y)
    return {
        "dataset": name,
        "files": int(files),
        "tokens": int(y.size),
        "song_fraction": float(y.mean()),
        "pearson_r": r,
        "r2": r2,
        "song_score_mean": float(score[y == 1].mean()),
        "silence_score_mean": float(score[y == 0].mean()),
    }


def concat_x(datasets):
    return np.concatenate([item["x"] for item in datasets], axis=0).astype(np.float32, copy=False)


def concat_y(datasets):
    return np.concatenate([item["y"] for item in datasets], axis=0).astype(np.float32, copy=False)


def concat_pixel(datasets):
    if any(item["pixel"] is None for item in datasets):
        return None
    return np.concatenate([item["pixel"] for item in datasets], axis=0).astype(np.float32, copy=False)


def check_dims(datasets):
    dims = {item["x"].shape[1] for item in datasets}
    assert len(dims) == 1, f"feature dims must match: {sorted(dims)}"


def add_pixel_summary(row, pixel, y, files):
    if pixel is not None:
        row["pixel_intensity"] = summarize("pixel_intensity", pixel, y, files)
    return row


def summarize_datasets(datasets, scores, summary):
    start = 0
    for item in datasets:
        end = start + item["x"].shape[0]
        row = summarize(item["name"], scores[start:end], item["y"], item["files"])
        summary["by_dataset"].append(add_pixel_summary(row, item["pixel"], item["y"], item["files"]))
        for plot_item in item["items"]:
            plot_item["score"] = scores[start + plot_item["start"] : start + plot_item["end"]]
        start = end


def main():
    parser = argparse.ArgumentParser(description="Measure how much global SV1 tracks annotation song state.")
    parser.add_argument("--model", required=True, choices=["songmae", "hubert", "birdaves", "aves", "songmae_random"])
    parser.add_argument("--dataset", action="append", required=True, help="Repeat as name=/path/to/embeddings.npz-or-dir")
    parser.add_argument("--feature_key", default="encoded_embeddings")
    parser.add_argument("--zscore", dest="zscore", action="store_true", default=True)
    parser.add_argument("--no_zscore", dest="zscore", action="store_false")
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--plot_dir", default=None)
    args = parser.parse_args()

    datasets = [load_dataset(*parse_dataset(item), args.feature_key) for item in args.dataset]
    check_dims(datasets)
    x = concat_x(datasets)
    y = concat_y(datasets)
    pixel = concat_pixel(datasets)
    assert y.min() < y.max(), "need both song and silence labels"
    if args.zscore:
        x = zscore(x)

    mean, sv1, singular_value = top_singular_vector(x)
    scores = (x - mean) @ sv1
    if scores[y == 1].mean() < scores[y == 0].mean():
        sv1 *= -1.0
        scores *= -1.0
    global_row = summarize("global", scores, y, sum(item["files"] for item in datasets))
    summary = {
        "model": "birdaves" if args.model == "aves" else args.model,
        "feature_key": args.feature_key,
        "zscore": bool(args.zscore),
        "feature_dim": int(x.shape[1]),
        "sv1_singular_value": singular_value,
        "datasets": [{"dataset": item["name"], "path": item["path"], "tokens": int(item["x"].shape[0])} for item in datasets],
        "target": "labels_downsampled >= 0",
        "global": add_pixel_summary(global_row, pixel, y, sum(item["files"] for item in datasets)),
        "by_dataset": [],
    }

    summarize_datasets(datasets, scores, summary)

    plot_dir = Path(args.plot_dir) if args.plot_dir else (Path(args.out_json).parent if args.out_json else None)
    if plot_dir:
        plot_metric_bars(summary, plot_dir / "sv1_annotation_r2.png")
        items = [plot_item for item in datasets for plot_item in item["items"]]
        if items:
            plot_sv1_overlay(items, plot_dir / "sv1_projection_overlay.png")

    text = json.dumps(summary, indent=2) + "\n"
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
