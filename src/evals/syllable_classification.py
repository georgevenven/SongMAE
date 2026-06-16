#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


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
    data = np.load(path, allow_pickle=True)
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


def save_prediction_plots(predictions, spans, groups, units, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_group = {}
    for pred, span, group in zip(predictions, spans, groups):
        by_group.setdefault(group, []).append((pred, span))

    for index, (group, rows) in enumerate(sorted(by_group.items())[:50]):
        start = min(span[1] for _, span in rows)
        end = max(span[2] for _, span in rows)
        true = ground_truth(units, rows[0][1][0], start, end)
        pred = np.zeros(end - start, dtype=np.int64)
        for label, (_, lo, hi) in rows:
            pred[lo - start : hi - start] = label

        vmax = max(1, int(max(true.max(initial=0), pred.max(initial=0))))
        fig, axes = plt.subplots(2, 1, figsize=(8, 1.6), dpi=160, sharex=True)
        for ax, row, title in zip(axes, [true, pred], ["truth", "prediction"]):
            ax.imshow(row[None, :], aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
            ax.set_yticks([])
            ax.set_ylabel(title)
        axes[-1].set_xlabel("ms")
        fig.tight_layout()
        fig.savefig(out_dir / f"{index:04d}_{group.replace(':', '_')}.png")
        plt.close(fig)


def standardize(train_x, val_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = np.maximum(train_x.std(axis=0, keepdims=True), 1e-6)
    return (train_x - mean) / std, (val_x - mean) / std


def split_by_group(x, y, spans, groups, val_fraction, seed):
    groups = np.asarray(groups)
    keys = np.array(sorted(set(groups.tolist())))
    assert keys.size >= 2, "Need at least two songs for a train/val split."
    keys = keys[np.random.default_rng(seed).permutation(keys.size)]
    val_count = max(1, int(round(keys.size * val_fraction)))
    val_count = min(val_count, keys.size - 1)
    val_keys = set(keys[:val_count].tolist())
    val = np.array([group in val_keys for group in groups])
    train = ~val
    assert len(set(y[train].tolist())) >= 2, "Train split has fewer than two classes."
    return (
        x[train],
        y[train],
        x[val],
        [spans[index] for index in np.flatnonzero(val)],
        [groups[index] for index in np.flatnonzero(val)],
        int(train.sum()),
        int(val.sum()),
        int(keys.size - val_count),
        int(val_count),
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
    parser.add_argument("--embeddings", required=True, help="Single concatenated embeddings .npz; split by song_id.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--model", choices=["logreg", "mlp"], default="logreg")
    parser.add_argument("--feature_key", default="encoded_embeddings")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot_dir")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    x, y, spans, groups = load_embeddings(args.embeddings, args.feature_key)
    train_x, train_y, val_x, val_spans, val_groups, train_tokens, val_tokens, train_songs, val_songs = split_by_group(
        x, y, spans, groups, args.val_fraction, args.seed
    )
    train_x, val_x = standardize(train_x, val_x)

    if args.model == "logreg":
        pred = fit_logreg(train_x, train_y).predict(val_x)
    else:
        model, classes = fit_mlp(train_x, train_y, args)
        pred = predict_mlp(model, classes, val_x)

    units = load_units(args.annotations)
    metrics = raster_metrics(pred, val_spans, units)
    if args.plot_dir:
        save_prediction_plots(pred, val_spans, val_groups, units, args.plot_dir)
    metrics.update(
        {
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "train_songs": train_songs,
            "val_songs": val_songs,
        }
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
