#!/usr/bin/env python3
"""Measure how much a model's top singular vector (SV1) tracks song state.

One SV1 is fit (PCA, top component) on every token of the full recordings of all
--dataset arguments pooled together, then scored per dataset, at 1 ms resolution over
the whole recording, against unit-coverage song state from each dataset's annotations.
Run with a single --dataset for a per-species SV1, or many for a shared one.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.embedding_store import EmbeddingStore
from src.evals.syllable_classification import load_units
from src.plotting_utils.spectrogram_prediction_vs_groundtruth import ground_truth


def embedding_paths(path):
    path = Path(path)
    if (path / "encoded_embeddings.npy").exists():
        return [path]
    paths = [p for p in sorted(path.iterdir()) if p.is_dir() and (p / "encoded_embeddings.npy").exists()]
    assert paths, f"no embedding folders found: {path}"
    return paths


def parse_dataset(value):
    # name=/path/to/embeddings  or  name=/path/to/embeddings=/path/to/annotations.json
    # (the per-dataset annotations let one pooled SV1 be scored against each dataset's units).
    parts = value.split("=", 2)
    assert len(parts) >= 2 and parts[0] and parts[1], "--dataset must look like name=/path/to/embeddings[=annotations.json]"
    annotations = Path(parts[2]) if len(parts) == 3 and parts[2] else None
    return parts[0], Path(parts[1]), annotations


def load_one(path, feature_key):
    data = EmbeddingStore(path)
    assert feature_key in data, f"{feature_key} missing from {path}"
    assert "labels_downsampled" in data, f"labels_downsampled missing from {path}"
    x = data[feature_key].astype(np.float32, copy=False).reshape(data[feature_key].shape[0], -1)
    y = data["labels_downsampled"].astype(np.int64, copy=False)
    assert x.shape[0] == y.shape[0], f"feature/label length mismatch in {path}"
    return data, x, (y >= 0).astype(np.float32)


def token_spans(data, count):
    # One (recording_stem, start_ms, end_ms) per feature row, for per-ms rasterization.
    if any(key not in data for key in ("recording_stem", "token_start_ms", "token_end_ms")):
        return None
    stems = np.asarray(data["recording_stem"]).astype(str)
    starts = np.rint(data["token_start_ms"]).astype(np.int64)
    ends = np.rint(data["token_end_ms"]).astype(np.int64)
    assert stems.shape[0] == starts.shape[0] == ends.shape[0] == count
    return list(zip(stems.tolist(), starts.tolist(), ends.tolist()))


def token_spectrogram(data, count):
    # Best-effort per-token spectrogram for the overlay panels. Returns None (skip panels)
    # when stored timebins do not tile into count*patch_width tokens, e.g. patch_width>1 with
    # segment padding (p32x4); the R^2 metric does not depend on this.
    if "spectrograms" not in data:
        return None
    spec = data["spectrograms"].astype(np.float32, copy=False)
    if spec.shape[0] == count:
        return spec.T
    patch_width = int(data.metadata.get("patch_width", spec.shape[0] // count))
    if patch_width <= 0 or spec.shape[0] < count * patch_width:
        return None
    spec = spec[: count * patch_width]
    return spec.reshape(count, patch_width, spec.shape[1]).mean(axis=1).T


def pixel_intensity(data, count):
    spec = token_spectrogram(data, count)
    if spec is None:
        return None
    return spec.mean(axis=0).astype(np.float32, copy=False)


def overlay_window(data, g_start, g_end, song, window_ms=5000.0):
    # ~window_ms token slice of a recording, anchored so annotated song is visible.
    if any(key not in data for key in ("token_start_ms", "token_end_ms")):
        return g_start, g_end
    starts = np.asarray(data["token_start_ms"], dtype=np.float64)
    ends = np.asarray(data["token_end_ms"], dtype=np.float64)
    per_token = max(1e-6, float(ends[g_start] - starts[g_start]))
    span = max(1, int(round(window_ms / per_token)))
    if span >= (g_end - g_start):
        return g_start, g_end
    song_idx = np.flatnonzero(song > 0)
    anchor = g_start + int(song_idx[0]) if song_idx.size else g_start
    lo = min(max(g_start, anchor - span // 4), g_end - span)
    return lo, lo + span


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
            lo, hi = overlay_window(data, start, end, song)
            items.append(
                {
                    "dataset": dataset,
                    "name": keys[start],
                    "start": offset + lo,
                    "end": offset + hi,
                    "song": y[lo:hi],
                    "spectrogram": spec[:, lo:hi],
                }
            )
        start = int(end)
    return items


def load_dataset(name, path, feature_key):
    xs, ys, pixels, items, spans = [], [], [], [], []
    offset = 0
    have_spans = True
    for embedding_path in embedding_paths(path):
        data, x, y = load_one(embedding_path, feature_key)
        pixel = pixel_intensity(data, y.size)
        xs.append(x)
        ys.append(y)
        if pixel is not None:
            pixels.append(pixel)
        span = token_spans(data, y.size)
        if span is None:
            have_spans = False
        else:
            spans.extend(span)
        items.extend(overlay_items(name, embedding_path, data, y, offset))
        offset += x.shape[0]
    return {
        "name": name,
        "path": str(path),
        "x": np.concatenate(xs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "pixel": np.concatenate(pixels, axis=0) if pixels else None,
        "files": len(xs),
        "items": items,
        "spans": spans if have_spans else None,
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


def minmax01(values):
    # Min-max normalize to [0, 1]; constant input maps to all-zeros.
    values = np.asarray(values, dtype=np.float64)
    lo, hi = values.min(), values.max()
    return (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)


def rasterize(spans, scores, units):
    # Expand each token's SV1 score over its [start_ms, end_ms) at 1 ms resolution over the whole
    # recording and compare against unit-coverage song state. Scores are min-max normalized to
    # [0, 1] once, globally (a single affine that Pearson R^2 is invariant to).
    by_stem = {}
    for index, (stem, start, end) in enumerate(spans):
        by_stem.setdefault(stem, []).append(index)

    score_parts, song_parts = [], []
    for stem, indices in by_stem.items():
        max_end = max(int(spans[i][2]) for i in indices)
        if max_end <= 0:
            continue
        score_grid = np.full(max_end, np.nan, dtype=np.float64)
        for i in indices:
            lo = max(0, int(spans[i][1]))
            hi = min(max_end, int(spans[i][2]))
            if hi > lo:
                score_grid[lo:hi] = scores[i]
        song_grid = (ground_truth(units, stem, 0, max_end) > 0).astype(np.float64)
        mask = ~np.isnan(score_grid)
        if mask.any():
            score_parts.append(score_grid[mask])
            song_parts.append(song_grid[mask])

    assert score_parts, "no token-covered frames to rasterize"
    return minmax01(np.concatenate(score_parts)), np.concatenate(song_parts)


def raster_summary(spans, scores, units):
    if not spans or units is None:
        return {}
    score, song = rasterize(spans, scores, units)
    assert song.min() < song.max(), "need both song and silence frames after rasterizing"
    r, r2 = pearson_r2(score, song)
    return {
        "r2_raster": r2,
        "pearson_r_raster": r,
        "frames_raster": int(song.size),
        "song_fraction_raster": float(song.mean()),
    }


def summarize(name, score, y, files, spans=None, units=None):
    scores = minmax01(score)
    assert y.min() < y.max(), f"need both song and silence labels for {name}"
    r, r2 = pearson_r2(scores, y)
    row = {
        "dataset": name,
        "files": int(files),
        "tokens": int(y.size),
        "song_fraction": float(y.mean()),
        "pearson_r": r,
        "r2": r2,
        "song_score_mean": float(scores[y == 1].mean()),
        "silence_score_mean": float(scores[y == 0].mean()),
    }
    row.update(raster_summary(spans, score, units))
    return row


def concat_x(datasets):
    return np.concatenate([item["x"] for item in datasets], axis=0).astype(np.float32, copy=False)


def concat_y(datasets):
    return np.concatenate([item["y"] for item in datasets], axis=0).astype(np.float32, copy=False)


def concat_pixel(datasets):
    if any(item["pixel"] is None for item in datasets):
        return None
    return np.concatenate([item["pixel"] for item in datasets], axis=0).astype(np.float32, copy=False)


def concat_spans(datasets):
    if any(item["spans"] is None for item in datasets):
        return None
    spans = []
    for item in datasets:
        spans.extend(item["spans"])
    return spans


def save_overlay_per_dataset(datasets, out_dir, limit=8):
    # One <name>.npz per dataset so each species gets its own spec panels.
    out_dir = Path(out_dir)
    for item in datasets:
        items = item["items"][:limit]
        if not items:
            continue
        payload = {}
        for index, plot_item in enumerate(items):
            payload[f"spectrogram_{index}"] = plot_item["spectrogram"].astype(np.float32, copy=False)
            payload[f"score_{index}"] = np.asarray(plot_item["score"], dtype=np.float32)
            payload[f"song_{index}"] = np.asarray(plot_item["song"], dtype=np.float32)
        payload["names"] = np.array([plot_item["name"] for plot_item in items])
        payload["datasets"] = np.array([plot_item["dataset"] for plot_item in items])
        payload["count"] = np.array(len(items))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{item['name']}.npz"
        np.savez_compressed(out_path, **payload)
        print(f"overlay saved to {out_path}")


def check_dims(datasets):
    dims = {item["x"].shape[1] for item in datasets}
    assert len(dims) == 1, f"feature dims must match: {sorted(dims)}"


def add_pixel_summary(row, pixel, y, files, spans=None, units=None):
    if pixel is not None:
        row["pixel_intensity"] = summarize("pixel_intensity", pixel, y, files, spans, units)
    return row


def summarize_datasets(datasets, scores, summary):
    # Each dataset is rasterized against its own units (so one pooled SV1 is scored per dataset);
    # plot scores are sliced from the shared projection.
    start = 0
    for item in datasets:
        end = start + item["x"].shape[0]
        units = item["units"]
        row = summarize(item["name"], scores[start:end], item["y"], item["files"], item["spans"], units)
        pixel_row = add_pixel_summary(row, item["pixel"], item["y"], item["files"], item["spans"], units)
        summary["by_dataset"].append(pixel_row)
        for plot_item in item["items"]:
            plot_item["score"] = scores[start + plot_item["start"] : start + plot_item["end"]]
        start = end


def merge_units(datasets):
    merged = {}
    for item in datasets:
        if item["units"]:
            merged.update(item["units"])
    return merged or None


def main():
    parser = argparse.ArgumentParser(description="Measure how much SV1 tracks unit-coverage song state over full recordings.")
    parser.add_argument("--model", required=True, help="Model label used as the heatmap column (free-form).")
    parser.add_argument("--dataset", action="append", required=True,
                        help="Repeat as name=/path/to/embeddings[=annotations.json]; many datasets share one SV1.")
    parser.add_argument("--annotations", default=None, help="Default annotation JSON (used when a --dataset omits its own).")
    parser.add_argument("--feature_key", default="encoded_embeddings")
    parser.add_argument("--zscore", dest="zscore", action="store_true", default=True)
    parser.add_argument("--no_zscore", dest="zscore", action="store_false")
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--overlay_dir", default=None, help="Save one <name>.npz per dataset here for spec panels.")
    args = parser.parse_args()

    datasets = []
    for item in args.dataset:
        name, path, ann = parse_dataset(item)
        ann = str(ann) if ann is not None else args.annotations
        dataset = load_dataset(name, path, args.feature_key)
        dataset["units"] = load_units(ann) if ann else None
        datasets.append(dataset)
    check_dims(datasets)
    x = concat_x(datasets)
    y = concat_y(datasets)
    pixel = concat_pixel(datasets)
    assert y.min() < y.max(), "need both song and silence labels"
    if args.zscore:
        x = zscore(x)

    # One SV1 fit on every token of every dataset's full recordings (song + silence), pooled.
    mean, sv1, singular_value = top_singular_vector(x)
    scores = (x - mean) @ sv1
    if scores[y == 1].mean() < scores[y == 0].mean():
        sv1 *= -1.0
        scores *= -1.0
    global_files = sum(item["files"] for item in datasets)
    global_spans = concat_spans(datasets)
    global_units = merge_units(datasets)
    global_row = summarize("global", scores, y, global_files, global_spans, global_units)
    summary = {
        "model": args.model,
        "feature_key": args.feature_key,
        "zscore": bool(args.zscore),
        "feature_dim": int(x.shape[1]),
        "sv1_singular_value": singular_value,
        "datasets": [{"dataset": item["name"], "path": item["path"], "tokens": int(item["x"].shape[0])} for item in datasets],
        "fit": f"SV1 fit on full recordings, pooled across {len(datasets)} dataset(s)",
        "target": "unit coverage over full recording (per-ms)" if global_units is not None else "labels_downsampled >= 0",
        "global": add_pixel_summary(global_row, pixel, y, global_files, global_spans, global_units),
        "by_dataset": [],
    }

    summarize_datasets(datasets, scores, summary)

    if args.overlay_dir:
        save_overlay_per_dataset(datasets, args.overlay_dir)

    text = json.dumps(summary, indent=2) + "\n"
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
