#!/usr/bin/env python3
"""Capped-label linear probing with shared song-level folds."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.embedding_store import EmbeddingStore
from src.evals.syllable_metrics import macro_fer_breakdown


def load_embeddings(path):
    data = EmbeddingStore(path)
    x = data["encoded_embeddings"]
    if x.ndim == 3:
        x = x[:, -1]
    x = x.astype(np.float32, copy=False).reshape(x.shape[0], -1)
    labels = np.asarray(data["labels_downsampled"], dtype=np.int64)
    y = np.where(labels < 0, 0, labels + 1)
    stems = np.asarray(data["recording_stem"]).astype(str)
    starts = np.rint(data["token_start_ms"]).astype(np.int64)
    ends = np.rint(data["token_end_ms"]).astype(np.int64)
    song_ids = np.asarray(data["song_id"])
    assert x.shape[0] == y.shape[0] == stems.shape[0] == starts.shape[0] == ends.shape[0]
    assert song_ids.shape[0] == x.shape[0]
    spans = list(zip(stems.tolist(), starts.tolist(), ends.tolist()))
    groups = [f"{stem}:{song}" for stem, song in zip(stems, song_ids.tolist())]
    return x, y, spans, groups


def load_units(path):
    units = {}
    for recording in json.loads(Path(path).read_text())["recordings"]:
        stem = Path(recording["recording"]["filename"]).stem
        units[stem] = [
            (round(unit["onset_ms"]), round(unit["offset_ms"]), int(unit["id"]) + 1)
            for event in recording.get("detected_events", [])
            for unit in event.get("units", [])
        ]
    return units


def ground_truth(units, stem, start, end):
    y = np.zeros(end - start, dtype=np.int64)
    for onset, offset, label in units.get(stem, []):
        lo = max(start, onset)
        hi = min(end, offset)
        if lo < hi:
            y[lo - start : hi - start] = label
    return y


def raster_arrays(predictions, spans, units, labels):
    y_true = []
    y_pred = []
    for pred, (stem, start, end) in zip(predictions, spans):
        truth = ground_truth(units, stem, start, end)
        keep = np.isin(truth, labels)
        y_true.append(truth[keep])
        y_pred.append(np.full(int(keep.sum()), pred, dtype=np.int64))
    return np.concatenate(y_true), np.concatenate(y_pred)


def metrics(y_true, y_pred):
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    label_index = {label: i for i, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true, pred in zip(y_true.tolist(), y_pred.tolist()):
        confusion[label_index[true], label_index[pred]] += 1
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class = {}
    for i, label in enumerate(labels):
        mask = y_true == label
        per_class[str(label)] = {
            "f1": float(f1[i]),
            "fer": float(np.mean(y_pred[mask] != label)),
            "frames": int(mask.sum()),
        }
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "fer": float(np.mean(y_true != y_pred)),
        **macro_fer_breakdown(labels, confusion),
        "frames": int(y_true.size),
        "classes": len(labels),
        "class_labels": labels,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def make_folds(y, groups, count, seed):
    keys = sorted(set(groups))
    assert 2 <= count <= len(keys)
    group_labels = {group: set() for group in keys}
    for label, group in zip(y.tolist(), groups):
        group_labels[group].add(label)
    labels = sorted(set(y.tolist()))
    available = {label: sum(label in group_labels[group] for group in keys) for label in labels}
    assert min(available.values()) >= count
    for attempt in range(100):
        rng = np.random.default_rng(seed + attempt)
        order = sorted(
            keys,
            key=lambda group: (
                min(available[label] for label in group_labels[group]),
                rng.random(),
            ),
        )
        val_groups = [[] for _ in range(count)]
        fold_counts = [{label: 0 for label in labels} for _ in range(count)]
        for group in order:
            row = group_labels[group]
            scores = [
                sum(fold_counts[fold][label] / available[label] for label in row)
                + len(val_groups[fold]) / len(keys)
                for fold in range(count)
            ]
            best = np.flatnonzero(np.asarray(scores) == min(scores))
            fold = int(rng.choice(best))
            val_groups[fold].append(group)
            for label in row:
                fold_counts[fold][label] += 1
        if all(fold_counts[fold][label] for fold in range(count) for label in labels):
            return [
                {
                    "train_groups": sorted(set(keys) - set(val)),
                    "val_groups": sorted(val),
                }
                for val in val_groups
            ]
    raise AssertionError("could not stratify every class across folds")


def group_mask(groups, selected):
    selected = set(selected)
    return np.array([group in selected for group in groups])


def validate_folds(folds, groups):
    groups = set(groups)
    validation = []
    for fold in folds:
        train = set(fold["train_groups"])
        val = set(fold["val_groups"])
        assert train.isdisjoint(val)
        assert train | val == groups
        assert all(row["group"] in train for row in fold["selected_occurrences"])
        validation.extend(val)
    assert len(validation) == len(groups) == len(set(validation))


def occurrence_runs(y, spans, groups, mask):
    runs = []
    current = []
    for index in np.flatnonzero(mask):
        if current:
            previous = current[-1]
            contiguous = (
                groups[index] == groups[previous]
                and y[index] == y[previous]
                and spans[index][0] == spans[previous][0]
                and spans[index][1] <= spans[previous][2]
            )
            if not contiguous:
                runs.append(current)
                current = []
        current.append(int(index))
    if current:
        runs.append(current)
    by_label = {}
    for run in runs:
        by_label.setdefault(int(y[run[0]]), []).append(run)
    return by_label


def occurrence(run, label, spans, groups):
    return {
        "label": label,
        "group": groups[run[0]],
        "stem": spans[run[0]][0],
        "start_ms": spans[run[0]][1],
        "end_ms": spans[run[-1]][2],
        "tokens": len(run),
    }


def select_occurrences(by_label, labels, cap, seed, fold_index, spans, groups):
    selected = []
    rows = []
    for label in labels:
        runs = by_label[label]
        rng = np.random.default_rng(np.random.SeedSequence([seed, fold_index, label]))
        for index in rng.permutation(len(runs))[:cap]:
            run = runs[index]
            selected.extend(run)
            rows.append(occurrence(run, label, spans, groups))
    return np.array(sorted(selected), dtype=np.int64), rows


def matching_indices(y, spans, pool, row):
    overlapping = [
        index
        for index in pool
        if max(spans[index][1], row["start_ms"]) < min(spans[index][2], row["end_ms"])
    ]
    matching = [index for index in overlapping if int(y[index]) == row["label"]]
    return matching or overlapping


def manifest_indices(y, spans, groups, train, rows, labels, cap, seed, fold_index):
    by_label = occurrence_runs(y, spans, groups, train)
    background = iter(())
    if 0 in labels:
        rng = np.random.default_rng(np.random.SeedSequence([seed, fold_index, 0]))
        runs = by_label[0]
        background = iter(
            occurrence(runs[index], 0, spans, groups)
            for index in rng.permutation(len(runs))[:cap]
        )
    pools = {}
    for index in np.flatnonzero(train):
        pools.setdefault(groups[index], []).append(index)
    selected = []
    assigned = {}
    for row in rows:
        matches = matching_indices(y, spans, pools.get(row["group"], []), row)
        if row["label"] == 0 and not any(int(y[index]) == 0 for index in matches):
            row = next(background)
            matches = matching_indices(y, spans, pools.get(row["group"], []), row)
        assert matches, row
        for index in matches:
            assert assigned.get(index, row["label"]) == row["label"]
            assigned[index] = row["label"]
            y[index] = row["label"]
        selected.extend(matches)
    return np.array(sorted(set(selected)), dtype=np.int64)


def build_manifest(y, spans, groups, folds, cap, seed):
    labels = sorted(set(y.tolist()))
    rows = []
    for fold_index, fold in enumerate(folds):
        train = group_mask(groups, fold["train_groups"])
        by_label = occurrence_runs(y, spans, groups, train)
        _, occurrences = select_occurrences(by_label, labels, cap, seed, fold_index, spans, groups)
        rows.append({**fold, "selected_occurrences": occurrences})
    return {
        "seed": seed,
        "fold_count": len(folds),
        "fold_strategy": "multilabel_stratified_song",
        "sampling": "nested_capped_occurrences",
        "label_cap": cap,
        "class_labels": labels,
        "folds": rows,
    }


def pca_features(x, components, seed, cache_path):
    if not components:
        return x, 0.0, False
    started = time.perf_counter()
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        transformed = np.load(cache_path, mmap_mode="r")
        assert transformed.shape == (x.shape[0], components)
        return transformed, time.perf_counter() - started, True
    assert components <= min(x.shape)
    model = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    model.fit(np.asarray(x, dtype=np.float32))
    transformed = model.transform(x).astype(np.float32, copy=False)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, transformed)
    return transformed, time.perf_counter() - started, False


def feature_stats(x, indices, batch_size):
    total = 0
    sums = np.zeros(x.shape[1], dtype=np.float64)
    squares = np.zeros(x.shape[1], dtype=np.float64)
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


def fit_linear(x, y, indices, args, fold_index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = np.array(sorted(set(y[indices].tolist())), dtype=np.int64)
    torch.manual_seed(args.seed + fold_index)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + fold_index)
    model = torch.nn.Linear(x.shape[1], len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    counts = np.array([np.sum(y[indices] == label) for label in classes])
    weights = torch.from_numpy(indices.size / (len(classes) * counts)).float().to(device)
    generator = torch.Generator().manual_seed(args.seed + fold_index)
    mean, std = feature_stats(x, indices, args.batch_size)
    batches = (indices.size + args.batch_size - 1) // args.batch_size
    model.train()
    for step in range(args.steps):
        if step % batches == 0:
            order = torch.randperm(indices.size, generator=generator).numpy()
        start = (step % batches) * args.batch_size
        batch_indices = indices[order[start : start + args.batch_size]]
        targets = torch.from_numpy(np.searchsorted(classes, y[batch_indices])).to(device)
        loss = torch.nn.functional.cross_entropy(
            model(feature_batch(x, batch_indices, mean, std, device)), targets, weight=weights
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model, classes, mean, std


@torch.no_grad()
def predict(model, classes, x, indices, batch_size, mean, std):
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    for start in range(0, indices.size, batch_size):
        batch = feature_batch(x, indices[start : start + batch_size], mean, std, device)
        predictions.append(model(batch).argmax(dim=1).cpu().numpy())
    return classes[np.concatenate(predictions)]


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-validated capped-label linear probe.")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--label_cap", type=int, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--manifest_in")
    parser.add_argument("--manifest_out")
    parser.add_argument("--pca_components", type=int, default=128)
    parser.add_argument("--pca_cache")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    total_started = time.perf_counter()
    args = parse_args()
    assert args.label_cap > 0
    assert args.steps > 0
    x, y, spans, groups = load_embeddings(args.embeddings)
    units = load_units(args.annotations)

    if args.manifest_in:
        manifest = json.loads(Path(args.manifest_in).read_text())
        assert manifest["label_cap"] == args.label_cap
    else:
        folds = make_folds(y, groups, args.folds, args.seed)
        manifest = build_manifest(y, spans, groups, folds, args.label_cap, args.seed)
    validate_folds(manifest["folds"], groups)
    if args.manifest_out:
        Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_out).write_text(json.dumps(manifest, indent=2) + "\n")

    x, pca_seconds, cache_hit = pca_features(x, args.pca_components, args.seed, args.pca_cache)
    labels = manifest["class_labels"]
    all_true = []
    all_pred = []
    fold_metrics = []
    fit_seconds = 0.0
    predict_seconds = 0.0
    for fold_index, fold in enumerate(manifest["folds"]):
        fold_y = y.copy()
        train = group_mask(groups, fold["train_groups"])
        val = group_mask(groups, fold["val_groups"]) & np.isin(y, labels)
        train_indices = manifest_indices(
            fold_y,
            spans,
            groups,
            train,
            fold["selected_occurrences"],
            labels,
            args.label_cap,
            args.seed,
            fold_index,
        )
        assert set(fold_y[train_indices].tolist()) == set(labels)
        val_indices = np.flatnonzero(val)
        assert val_indices.size

        started = time.perf_counter()
        model, classes, mean, std = fit_linear(x, fold_y, train_indices, args, fold_index)
        fit_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        pred = predict(model, classes, x, val_indices, args.batch_size, mean, std)
        predict_elapsed = time.perf_counter() - started
        truth, raster_pred = raster_arrays(
            pred, [spans[index] for index in val_indices], units, labels
        )
        fold_row = metrics(truth, raster_pred)
        for key in ("class_labels", "confusion_matrix", "per_class"):
            del fold_row[key]
        fold_row.update(
            {
                "fold": fold_index,
                "train_songs": len(fold["train_groups"]),
                "val_songs": len(fold["val_groups"]),
                "train_tokens": int(train_indices.size),
                "val_tokens": int(val_indices.size),
                "labeled_occurrences_by_class": {
                    str(label): sum(row["label"] == label for row in fold["selected_occurrences"])
                    for label in labels
                },
                "fit_seconds": fit_elapsed,
                "predict_seconds": predict_elapsed,
            }
        )
        fold_metrics.append(fold_row)
        all_true.append(truth)
        all_pred.append(raster_pred)
        fit_seconds += fit_elapsed
        predict_seconds += predict_elapsed

    result = metrics(np.concatenate(all_true), np.concatenate(all_pred))
    occurrence_counts = [
        count
        for fold in fold_metrics
        for label, count in fold["labeled_occurrences_by_class"].items()
        if label != "0"
    ]
    result.update(
        {
            "encoder_scope": "frozen_final_layer",
            "classifier": "class_balanced_linear",
            "class_weight": "balanced",
            "standardized": True,
            "label_cap": args.label_cap,
            "label_budget": "at_most",
            "folds": len(manifest["folds"]),
            "fold_strategy": manifest["fold_strategy"],
            "sampling": manifest["sampling"],
            "event_grouping": "recording_stem:song_id",
            "event_split_integrity": "disjoint",
            "class_labels": labels,
            "syllable_class_labels": [label for label in labels if label != 0],
            "background_label": 0,
            "labeled_occurrences": {
                "scope": "syllable_classes",
                "median": float(np.median(occurrence_counts)),
                "min": min(occurrence_counts),
                "max": max(occurrence_counts),
            },
            "pca_components": args.pca_components,
            "pca_fit_scope": "all_extracted_tokens",
            "pca_cache_hit": cache_hit,
            "optimizer_steps_per_fold": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "fold_metrics": fold_metrics,
            "timing_seconds": {
                "pca": pca_seconds,
                "fit": fit_seconds,
                "predict": predict_seconds,
                "total": time.perf_counter() - total_started,
            },
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
