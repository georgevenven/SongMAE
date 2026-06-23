#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.embedding_store import EmbeddingStore
from src.plotting_utils.spectrogram_prediction_vs_groundtruth import (
    load_plot_specs,
    save_spectrogram_prediction_vs_groundtruth,
)


def class_labels(labels):
    labels = np.asarray(labels, dtype=np.int64)
    return np.where(labels < 0, 0, labels + 1)


def token_spans(data, count):
    stems = np.asarray(data["recording_stem"]).astype(str)
    starts = np.rint(data["token_start_ms"]).astype(np.int64)
    ends = np.rint(data["token_end_ms"]).astype(np.int64)
    assert stems.shape[0] == starts.shape[0] == ends.shape[0] == count
    return stems, starts, ends


def token_groups(data, stems, count):
    assert "song_id" in data, "embeddings must include song_id"
    song_ids = np.asarray(data["song_id"])
    assert song_ids.shape[0] == count
    return [f"{stem}:{song}" for stem, song in zip(stems, song_ids.tolist())]


def load_embeddings(path, feature_key):
    data = EmbeddingStore(path)
    x = data[feature_key].astype(np.float32, copy=False)
    y = class_labels(data["labels_downsampled"])
    if x.shape[0] != y.shape[0]:
        assert feature_key == "spectrograms", f"{feature_key} rows do not match labels"
        x = np.asarray([chunk.mean(axis=0) for chunk in np.array_split(x, y.shape[0])], dtype=np.float32)
    x = x.reshape(x.shape[0], -1)
    assert x.shape[0] == y.shape[0], path
    stems, starts, ends = token_spans(data, x.shape[0])
    spans = list(zip(stems.tolist(), starts.tolist(), ends.tolist()))
    groups = token_groups(data, stems, x.shape[0])
    return x, y, spans, groups


def load_units(path):
    units = {}
    data = json.loads(Path(path).read_text())
    for recording in data["recordings"]:
        stem = Path(recording["recording"]["filename"]).stem
        items = []
        for event in recording.get("detected_events", []):
            for unit in event.get("units", []):
                items.append((round(unit["onset_ms"]), round(unit["offset_ms"]), int(unit["id"]) + 1))
        units[stem] = items
    return units


def ground_truth(units, stem, start, end):
    y = np.zeros(max(0, end - start), dtype=np.int64)
    for onset, offset, label in units.get(stem, []):
        lo = max(start, onset)
        hi = min(end, offset)
        if lo < hi:
            y[lo - start : hi - start] = label
    return y


def raster_metrics(predictions, spans, units):
    y_true, y_pred = [], []
    for pred, (stem, start, end) in zip(predictions, spans):
        if end <= start:
            continue
        y_true.append(ground_truth(units, stem, start, end))
        y_pred.append(np.full(end - start, pred, dtype=np.int64))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "fer": float(np.mean(y_true != y_pred)),
        "frames": int(y_true.size),
        "classes": len(labels),
    }


def standardize(train_x, val_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = np.maximum(train_x.std(axis=0, keepdims=True), 1e-6)
    return (train_x - mean) / std, (val_x - mean) / std


def group_seconds(spans, groups):
    seconds = {}
    for (_, start, end), group in zip(spans, groups):
        lo, hi = seconds.get(group, (start, end))
        seconds[group] = (min(lo, start), max(hi, end))
    return {group: (end - start) / 1000.0 for group, (start, end) in seconds.items()}


def group_classes(y, groups):
    classes = {}
    for label, group in zip(y, groups):
        classes.setdefault(group, set())
        if label > 0:
            classes[group].add(int(label))
    return classes


def select_val_keys(keys, group_to_classes, val_fraction, seed):
    counts = {}
    for classes in group_to_classes.values():
        for label in classes:
            counts[label] = counts.get(label, 0) + 1

    target_classes = set(counts)
    assert all(count > 1 for count in counts.values()), "Every syllable class needs train and validation examples."

    keys = keys[np.random.default_rng(seed).permutation(keys.size)]
    val_count = max(1, int(round(keys.size * val_fraction)))
    val_count = min(val_count, keys.size - 1)
    val_keys = []
    missing = set(target_classes)
    while missing:
        candidates = [
            key
            for key in keys
            if key not in val_keys
            and group_to_classes[key] & missing
            and all(counts[label] > 1 for label in group_to_classes[key])
        ]
        assert candidates, "Cannot make validation cover every syllable while keeping train coverage."
        key = max(candidates, key=lambda k: (len(group_to_classes[k] & missing), -len(group_to_classes[k]), k))
        val_keys.append(key)
        classes = group_to_classes[key]
        for label in classes:
            counts[label] -= 1
        missing -= classes

    for key in keys.tolist():
        if len(val_keys) >= val_count:
            break
        classes = group_to_classes[key]
        if key in val_keys or any(counts[label] <= 1 for label in classes):
            continue
        val_keys.append(key)
        for label in classes:
            counts[label] -= 1

    return set(val_keys)


def select_train_keys(keys, group_to_seconds, group_to_classes, max_seconds):
    if max_seconds is None:
        return set(keys.tolist())

    selected = set()
    used = 0.0
    missing = set().union(*(group_to_classes[key] for key in keys))
    while missing:
        candidates = [
            key
            for key in keys
            if key not in selected
            and group_to_seconds[key] + used <= max_seconds + 1e-9
            and group_to_classes[key] & missing
        ]
        assert candidates, f"Cannot cover syllables within --max_train_seconds={max_seconds:g}"
        key = min(
            candidates,
            key=lambda k: (-len(group_to_classes[k] & missing) / group_to_seconds[k], group_to_seconds[k], k),
        )
        selected.add(key)
        used += group_to_seconds[key]
        missing -= group_to_classes[key]

    while True:
        candidates = [
            key
            for key in keys
            if key not in selected and group_to_seconds[key] + used <= max_seconds + 1e-9
        ]
        if not candidates:
            return selected
        key = max(candidates, key=lambda k: (group_to_seconds[k], k))
        selected.add(key)
        used += group_to_seconds[key]


def split_by_group(x, y, spans, groups, val_fraction, seed, max_train_seconds):
    groups = np.asarray(groups)
    keys = np.array(sorted(set(groups.tolist())))
    assert keys.size >= 2, "Need at least two songs for a train/val split."

    group_to_seconds = group_seconds(spans, groups)
    group_to_classes = group_classes(y, groups)
    val_keys = select_val_keys(keys, group_to_classes, val_fraction, seed)
    train_keys = select_train_keys(
        np.array([key for key in keys if key not in val_keys]),
        group_to_seconds,
        group_to_classes,
        max_train_seconds,
    )

    val = np.array([group in val_keys for group in groups])
    train = np.array([group in train_keys for group in groups])
    assert len(set(y[train].tolist())) >= 2, "Train split has fewer than two classes."
    train_seconds = sum(group_to_seconds[key] for key in train_keys)
    target_classes = set().union(*group_to_classes.values())
    val_classes = set().union(*(group_to_classes[key] for key in val_keys))
    train_classes = set().union(*(group_to_classes[key] for key in train_keys))
    assert val_classes == target_classes, "Validation split does not cover every syllable class."
    assert train_classes == target_classes, "Train split does not cover every syllable class."
    return (
        x[train],
        y[train],
        x[val],
        [spans[index] for index in np.flatnonzero(val)],
        [groups[index] for index in np.flatnonzero(val)],
        int(train.sum()),
        int(val.sum()),
        int(len(train_keys)),
        int(len(val_keys)),
        float(train_seconds),
        int(len(train_classes)),
    )


def fit_logreg(train_x, train_y):
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    return model.fit(train_x, train_y)


class MLP(torch.nn.Module):
    def __init__(self, in_dim, classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, classes),
        )

    def forward(self, x):
        return self.net(x)


def fit_mlp(train_x, train_y, args):
    classes = np.array(sorted(set(train_y.tolist())), dtype=np.int64)
    targets = np.searchsorted(classes, train_y)
    model = MLP(train_x.shape[1], len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    x = torch.from_numpy(train_x.astype(np.float32, copy=False))
    y = torch.from_numpy(targets)
    for _ in range(args.epochs):
        order = torch.randperm(x.shape[0])
        for start in range(0, x.shape[0], args.batch_size):
            idx = order[start : start + args.batch_size]
            loss = torch.nn.functional.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, classes


def predict_mlp(model, classes, val_x):
    with torch.no_grad():
        logits = model(torch.from_numpy(val_x.astype(np.float32, copy=False)))
    return classes[logits.argmax(dim=1).numpy()]


def parse_args():
    parser = argparse.ArgumentParser(description="Fit a simple syllable classifier on extracted embeddings.")
    parser.add_argument("--embeddings", required=True, help="Embedding folder; split by song_id.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--model", choices=["logreg", "mlp"], default="logreg")
    parser.add_argument("--feature_key", default="encoded_embeddings")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--plot_dir")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_train_seconds", default="MAX", help="Whole-song train budget in seconds, or MAX.")
    return parser.parse_args()


def max_train_seconds(value):
    value = str(value).strip()
    if value.upper() == "MAX":
        return None
    seconds = float(value)
    assert seconds > 0, "--max_train_seconds must be positive or MAX"
    return seconds


def main():
    args = parse_args()

    # spans is one (recording_stem, token_start_ms, token_end_ms) tuple per feature row.
    # feature row is a singular example, aka (11289, 1536) a detected event with latents 
    x, y, spans, groups = load_embeddings(args.embeddings, args.feature_key)
    budget = max_train_seconds(args.max_train_seconds)

    (
        train_x,
        train_y,
        val_x,
        val_spans,
        val_groups,
        train_tokens,
        val_tokens,
        train_songs,
        val_songs,
        train_seconds,
        train_classes,
    ) = split_by_group(x, y, spans, groups, args.val_fraction, args.seed, budget)
    train_x, val_x = standardize(train_x, val_x)

    if args.model == "logreg":
        pred = fit_logreg(train_x, train_y).predict(val_x)
    else:
        model, classes = fit_mlp(train_x, train_y, args)
        pred = predict_mlp(model, classes, val_x)

    units = load_units(args.annotations)
    metrics = raster_metrics(pred, val_spans, units)
    if args.save_plots:
        assert args.plot_dir, "--save_plots requires --plot_dir"
        save_spectrogram_prediction_vs_groundtruth(
            pred, val_spans, val_groups, units, args.plot_dir, load_plot_specs(args.embeddings)
        )
    metrics.update(
        {
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "train_songs": train_songs,
            "val_songs": val_songs,
            "target_train_seconds": None if budget is None else float(budget),
            "actual_train_seconds": train_seconds,
            "train_classes": train_classes,
        }
    )
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
