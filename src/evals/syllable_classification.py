#!/usr/bin/env python3
import argparse
import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.embedding_store import EmbeddingStore
from src.evals.syllable_metrics import macro_fer_breakdown
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


def load_embeddings(path, all_layers=False):
    data = EmbeddingStore(path)
    x = data["encoded_embeddings"]
    if all_layers:
        assert x.ndim == 3, "encoded_embeddings must have shape [tokens, layers, dimensions]"
    else:
        if x.ndim == 3:
            x = x[:, -1]
        x = x.astype(np.float32, copy=False)
    y = class_labels(data["labels_downsampled"])
    if not all_layers:
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
    index = {label: i for i, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true, pred in zip(y_true.tolist(), y_pred.tolist()):
        confusion[index[true], index[pred]] += 1
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class = per_class_metrics(labels, f1, y_true, y_pred)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "fer": float(np.mean(y_true != y_pred)),
        **macro_fer_breakdown(labels, confusion),
        "frames": int(y_true.size),
        "classes": len(labels),
        "class_labels": [int(label) for label in labels],
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def per_class_metrics(labels, f1, y_true, y_pred):
    rows = {}
    for i, label in enumerate(labels):
        mask = y_true == label
        frames = int(mask.sum())
        rows[str(label)] = {
            "f1": float(f1[i]),
            "fer": None if frames == 0 else float(np.mean(y_pred[mask] != label)),
            "frames": frames,
        }
    return rows


def write_prediction_runs(path, predictions, spans, units):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for pred, (stem, start, end) in zip(predictions, spans):
            if end <= start:
                continue
            y = ground_truth(units, stem, start, end)
            run_start = 0
            for i in range(1, y.size + 1):
                if i < y.size and y[i] == y[run_start]:
                    continue
                row = {
                    "recording_stem": stem,
                    "start_ms": int(start + run_start),
                    "end_ms": int(start + i),
                    "true_label": int(y[run_start]),
                    "pred_label": int(pred),
                }
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
                run_start = i


def standardize(train_x, val_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = np.maximum(train_x.std(axis=0, keepdims=True), 1e-6)
    return (train_x - mean) / std, (val_x - mean) / std


@dataclass(frozen=True)
class SplitPolicy:
    val_fraction: float
    seed: int
    max_train_seconds: float | None
    require_all_classes: bool


@dataclass(frozen=True)
class SyllableSplit:
    train_groups: list[str]
    val_groups: list[str]
    train_order: list[str]
    train_classes: list[int]
    val_classes: list[int]
    base_val_classes: list[int]
    missing_train_classes: list[int]
    missing_val_classes: list[int]
    unmatched_val_groups: list[str]
    unseen_val_classes: list[int]
    train_seconds: float


def group_bounds(spans, groups):
    bounds = {}
    for (stem, start, end), group in zip(spans, groups):
        if group not in bounds:
            bounds[group] = (stem, start, end)
            continue
        old_stem, lo, hi = bounds[group]
        assert stem == old_stem
        bounds[group] = (stem, min(lo, start), max(hi, end))
    return bounds


def group_seconds(bounds):
    return {group: (end - start) / 1000.0 for group, (_, start, end) in bounds.items()}


def group_classes(bounds, units):
    classes = {group: set() for group in bounds}
    for group, (stem, start, end) in bounds.items():
        for onset, offset, label in units.get(stem, []):
            if max(start, onset) < min(end, offset):
                classes[group].add(label)
    return classes


def class_counts(group_to_classes):
    counts = {}
    for classes in group_to_classes.values():
        for label in classes:
            counts[label] = counts.get(label, 0) + 1
    return counts


def can_hold_out(group, group_to_classes, train_class_counts):
    return all(train_class_counts[label] > 1 for label in group_to_classes[group])


def hold_out(group, val_groups, train_class_counts, group_to_classes):
    val_groups.append(group)
    for label in group_to_classes[group]:
        train_class_counts[label] -= 1


def target_val_count(keys, val_fraction):
    count = max(1, int(round(keys.size * val_fraction)))
    return min(count, keys.size - 1)


def select_val_groups(keys, group_to_classes, policy):
    train_class_counts = class_counts(group_to_classes)
    target_count = target_val_count(keys, policy.val_fraction)
    shuffled = keys[np.random.default_rng(policy.seed).permutation(keys.size)].tolist()
    val_groups = []

    if policy.require_all_classes:
        assert all(
            count > 1 for count in train_class_counts.values()
        ), "Every syllable class needs train and validation examples."
        missing = set(train_class_counts)
        while missing:
            candidates = [
                group
                for group in shuffled
                if group not in val_groups
                and group_to_classes[group] & missing
                and can_hold_out(group, group_to_classes, train_class_counts)
            ]
            assert candidates, "Cannot make validation cover every syllable while keeping train coverage."
            group = max(
                candidates,
                key=lambda item: (
                    len(group_to_classes[item] & missing),
                    -len(group_to_classes[item]),
                    item,
                ),
            )
            hold_out(group, val_groups, train_class_counts, group_to_classes)
            missing -= group_to_classes[group]

    for group in shuffled:
        if len(val_groups) >= target_count:
            break
        if group in val_groups or not can_hold_out(group, group_to_classes, train_class_counts):
            continue
        hold_out(group, val_groups, train_class_counts, group_to_classes)

    assert val_groups, "Validation split has no songs with train-seen syllable classes."
    return set(val_groups)


def train_group_order(keys, group_to_seconds, group_to_classes, seed):
    shuffled = keys[np.random.default_rng(seed).permutation(keys.size)].tolist()
    selected = []
    missing = classes_for(shuffled, group_to_classes)
    while missing:
        candidates = [
            key
            for key in shuffled
            if key not in selected and group_to_classes[key] & missing
        ]
        assert candidates, "Cannot cover syllables from non-validation songs."
        key = max(
            candidates,
            key=lambda item: (
                len(group_to_classes[item] & missing) / max(group_to_seconds[item], 1e-9),
                -group_to_seconds[item],
                item,
            ),
        )
        selected.append(key)
        missing -= group_to_classes[key]
    selected += sorted(
        [key for key in shuffled if key not in selected],
        key=lambda item: (group_to_seconds[item], item),
    )
    return selected


def budget_train_groups(train_order, group_to_seconds, max_seconds):
    if max_seconds is None:
        return set(train_order)

    selected = set()
    used = 0.0
    for group in train_order:
        seconds = group_to_seconds[group]
        if used + seconds > max_seconds + 1e-9:
            break
        selected.add(group)
        used += seconds
    assert selected, f"No train songs fit within --max_train_seconds={max_seconds:g}"
    return selected


def classes_for(groups, group_to_classes):
    if not groups:
        return set()
    return set().union(*(group_to_classes[group] for group in groups))


def build_syllable_split(spans, groups, units, policy):
    groups = np.asarray(groups)
    keys = np.array(sorted(set(groups.tolist())))
    assert keys.size >= 2, "Need at least two songs for a train/val split."

    bounds = group_bounds(spans, groups)
    group_to_seconds = group_seconds(bounds)
    group_to_classes = group_classes(bounds, units)
    target_classes = classes_for(keys.tolist(), group_to_classes)
    val_groups = select_val_groups(keys, group_to_classes, policy)
    order = train_group_order(
        np.array([key for key in keys if key not in val_groups]),
        group_to_seconds,
        group_to_classes,
        policy.seed,
    )
    train_groups = budget_train_groups(order, group_to_seconds, policy.max_train_seconds)

    train_classes = classes_for(train_groups, group_to_classes)
    base_val_classes = classes_for(val_groups, group_to_classes)
    unseen_val_classes = base_val_classes - train_classes
    val_classes = base_val_classes if policy.require_all_classes else base_val_classes & train_classes
    unmatched_val_groups = sorted(group for group in val_groups if group_to_classes[group] - train_classes)
    if policy.require_all_classes:
        assert base_val_classes == target_classes, "Validation split does not cover every syllable class."
        assert train_classes == target_classes, "Train split does not cover every syllable class."

    return SyllableSplit(
        train_groups=sorted(train_groups),
        val_groups=sorted(val_groups),
        train_order=order,
        train_classes=sorted(int(label) for label in train_classes),
        val_classes=sorted(int(label) for label in val_classes),
        base_val_classes=sorted(int(label) for label in base_val_classes),
        missing_train_classes=sorted(int(label) for label in target_classes - train_classes),
        missing_val_classes=sorted(int(label) for label in target_classes - base_val_classes),
        unmatched_val_groups=unmatched_val_groups,
        unseen_val_classes=sorted(int(label) for label in unseen_val_classes),
        train_seconds=float(sum(group_to_seconds[group] for group in train_groups)),
    )


def split_masks(groups, y, split):
    train_groups = set(split.train_groups)
    val_groups = set(split.val_groups)
    groups = np.asarray(groups)
    train = np.array([group in train_groups for group in groups])
    train_labels = set(y[train].tolist())
    # Relaxed splits keep the same validation songs, but score only train-seen labels.
    val = np.array([group in val_groups and int(label) in train_labels for group, label in zip(groups, y)])
    return train, val


def fit_logreg(train_x, train_y):
    model = LogisticRegression(max_iter=5000)
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


class ScalarMixMLP(torch.nn.Module):
    def __init__(self, layers, dimensions, classes):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(layers))
        self.gamma = torch.nn.Parameter(torch.ones(()))
        self.head = MLP(dimensions, classes)

    def forward(self, x):
        mixed = self.gamma * torch.einsum("l,bld->bd", self.weights.softmax(dim=0), x)
        return self.head(mixed)


def feature_stats(x, indices, batch_size):
    total = 0
    sums = np.zeros(x.shape[1:], dtype=np.float64)
    squares = np.zeros(x.shape[1:], dtype=np.float64)
    for start in range(0, indices.size, batch_size):
        batch = np.asarray(x[indices[start : start + batch_size]], dtype=np.float32)
        sums += batch.sum(axis=0, dtype=np.float64)
        squares += np.square(batch, dtype=np.float64).sum(axis=0)
        total += batch.shape[0]
    mean = sums / total
    std = np.sqrt(np.maximum(squares / total - mean**2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def feature_batch(x, indices, mean, std, device):
    batch = np.asarray(x[indices], dtype=np.float32)
    batch -= mean
    batch /= std
    return torch.from_numpy(batch).to(device)


def fit_torch_classifier(x, y, indices, args):
    assert args.model in {"mlp", "scalar_mix"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = np.array(sorted(set(y[indices].tolist())), dtype=np.int64)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    if args.model == "mlp":
        model = MLP(x.shape[1], len(classes)).to(device)
    else:
        model = ScalarMixMLP(x.shape[1], x.shape[2], len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    mean, std = feature_stats(x, indices, args.batch_size)
    model.train()
    for _ in range(args.epochs):
        order = torch.randperm(indices.size, generator=generator).numpy()
        for start in range(0, indices.size, args.batch_size):
            batch_indices = indices[order[start : start + args.batch_size]]
            batch_targets = torch.from_numpy(np.searchsorted(classes, y[batch_indices])).to(device)
            loss = torch.nn.functional.cross_entropy(
                model(feature_batch(x, batch_indices, mean, std, device)), batch_targets
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model, classes, mean, std


@torch.no_grad()
def predict_torch_classifier(model, classes, x, indices, batch_size, mean, std):
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    for start in range(0, indices.size, batch_size):
        batch = feature_batch(x, indices[start : start + batch_size], mean, std, device)
        predictions.append(model(batch).argmax(dim=1).cpu().numpy())
    return classes[np.concatenate(predictions)]


def parse_args():
    parser = argparse.ArgumentParser(description="Fit a simple syllable classifier on extracted embeddings.")
    parser.add_argument("--embeddings", required=True, help="Embedding folder; split by song_id.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--model", choices=["logreg", "mlp", "scalar_mix"], default="logreg")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--plot_dir")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_train_seconds", default="MAX", help="Whole-song train budget in seconds, or MAX.")
    parser.add_argument("--allow_unmatched_val_classes", action="store_true")
    parser.add_argument("--split_json")
    parser.add_argument("--predictions_jsonl")
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
    x, y, spans, groups = load_embeddings(args.embeddings, args.model == "scalar_mix")
    units = load_units(args.annotations)
    budget = max_train_seconds(args.max_train_seconds)

    split = build_syllable_split(
        spans,
        groups,
        units,
        SplitPolicy(
            val_fraction=args.val_fraction,
            seed=args.seed,
            max_train_seconds=budget,
            require_all_classes=not args.allow_unmatched_val_classes,
        ),
    )

    train, val = split_masks(groups, y, split)
    assert len(set(y[train].tolist())) >= 2, "Train split has fewer than two classes."
    assert int(val.sum()) > 0, "Validation split has no embedding tokens."
    train_indices = np.flatnonzero(train)
    val_indices = np.flatnonzero(val)
    train_y = y[train_indices]
    val_spans = [spans[index] for index in val_indices]
    val_groups = [groups[index] for index in val_indices]
    scored_val_groups = sorted(set(val_groups))
    train_tokens = int(train.sum())
    val_tokens = int(val.sum())

    if args.model == "logreg":
        train_x, val_x = standardize(x[train_indices], x[val_indices])
        pred = fit_logreg(train_x, train_y).predict(val_x)
        adaptation = {}
    else:
        model, classes, mean, std = fit_torch_classifier(x, y, train_indices, args)
        pred = predict_torch_classifier(model, classes, x, val_indices, args.batch_size, mean, std)
        adaptation = {
            "encoder_scope": "frozen_final_layer" if args.model == "mlp" else "frozen_scalar_mix",
            "classifier": "mlp_1024_256",
            "standardized": True,
        }
        if args.model == "scalar_mix":
            adaptation["layer_weights"] = model.weights.softmax(dim=0).detach().cpu().tolist()
            adaptation["layer_gamma"] = float(model.gamma.detach().cpu())

    metrics = raster_metrics(pred, val_spans, units)
    metrics.update(adaptation)
    if args.predictions_jsonl:
        write_prediction_runs(args.predictions_jsonl, pred, val_spans, units)
    if args.save_plots:
        assert args.plot_dir, "--save_plots requires --plot_dir"
        save_spectrogram_prediction_vs_groundtruth(
            pred, val_spans, val_groups, units, args.plot_dir, load_plot_specs(args.embeddings)
        )
    metrics.update(
        {
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "train_songs": len(split.train_groups),
            "val_songs": len(split.val_groups),
            "scored_val_songs": len(scored_val_groups),
            "target_train_seconds": None if budget is None else float(budget),
            "actual_train_seconds": split.train_seconds,
            "train_classes": len(split.train_classes),
            "val_classes": len(split.val_classes),
            "base_val_classes": len(split.base_val_classes),
            "missing_train_classes": len(split.missing_train_classes),
            "missing_val_classes": len(split.missing_val_classes),
            "unmatched_val_songs": len(split.unmatched_val_groups),
            "unseen_val_classes": len(split.unseen_val_classes),
            "dropped_val_songs": len(set(split.val_groups) - set(scored_val_groups)),
        }
    )
    if args.split_json:
        split_payload = {
            "seed": int(args.seed),
            "val_fraction": float(args.val_fraction),
            "max_train_seconds": None if budget is None else float(budget),
            "allow_unmatched_val_classes": bool(args.allow_unmatched_val_classes),
            "train_groups": split.train_groups,
            "val_groups": split.val_groups,
            "scored_val_groups": scored_val_groups,
            "train_order": split.train_order,
            "train_classes": split.train_classes,
            "val_classes": split.val_classes,
            "base_val_classes": split.base_val_classes,
            "missing_train_classes": split.missing_train_classes,
            "missing_val_classes": split.missing_val_classes,
            "unmatched_val_groups": split.unmatched_val_groups,
            "unseen_val_classes": split.unseen_val_classes,
        }
        Path(args.split_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.split_json).write_text(json.dumps(split_payload, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
