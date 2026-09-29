#!/usr/bin/env python3
"""Recording-level cross-validated syllable linear probe; PCA and z-scoring fit on training recordings only."""
import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.embedding_store import EmbeddingStore
from src.evals.syllable_metrics import macro_fer_breakdown

DEFAULT_LOGREG_C = 1e-3


def load_embeddings(path):
    data = EmbeddingStore(path)
    x = data["encoded_embeddings"]
    if x.ndim == 3:
        x = x[:, -1]
    x = x.reshape(x.shape[0], -1).astype(np.float32, copy=False)
    raw_labels = np.asarray(data["labels_downsampled"], dtype=np.int64)
    y = np.where(raw_labels < 0, 0, raw_labels + 1)
    stems = np.asarray(data["recording_stem"]).astype(str)
    starts = np.rint(data["token_start_ms"]).astype(np.int64)
    ends = np.rint(data["token_end_ms"]).astype(np.int64)
    assert all(row.shape[0] == x.shape[0] for row in (y, stems, starts, ends))
    spans = list(zip(stems.tolist(), starts.tolist(), ends.tolist()))
    # Folds split whole recordings: events cut from one recording (often touching) never straddle train and validation.
    return x, y, spans, stems.tolist()


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


def group_truth_labels(units, spans, groups):
    # Ground-truth classes per recording: folds and scored classes are the same for every model on a bird,
    # including classes too short to win any of a coarse model's output bins.
    out = {}
    for span, group in zip(spans, groups):
        out.setdefault(group, set()).update(np.unique(ground_truth(units, *span)).tolist())
    return out


def make_folds(group_labels, count, seed):
    keys = sorted(group_labels)
    assert 2 <= count <= len(keys)
    labels = sorted(set().union(*group_labels.values()))
    availability = {
        label: sum(label in group_labels[group] for group in keys) for label in labels
    }
    assert min(availability.values()) >= count, "Every class needs one song per fold."

    for attempt in range(100):
        rng = np.random.default_rng(seed + attempt)
        order = sorted(
            keys,
            key=lambda group: (
                min(availability[label] for label in group_labels[group]),
                rng.random(),
            ),
        )
        validation = [[] for _ in range(count)]
        coverage = [{label: 0 for label in labels} for _ in range(count)]
        for group in order:
            scores = [
                sum(coverage[fold][label] / availability[label] for label in group_labels[group])
                + len(validation[fold]) / len(keys)
                for fold in range(count)
            ]
            candidates = np.flatnonzero(np.asarray(scores) == min(scores))
            fold = int(rng.choice(candidates))
            validation[fold].append(group)
            for label in group_labels[group]:
                coverage[fold][label] += 1
        if all(coverage[fold][label] for fold in range(count) for label in labels):
            return [
                {
                    "train_groups": sorted(set(keys) - set(val_groups)),
                    "val_groups": sorted(val_groups),
                }
                for val_groups in validation
            ]
    raise AssertionError("Could not stratify every class across folds.")


def load_manifest(args, group_labels):
    classes = sorted(set().union(*group_labels.values()))
    if args.manifest_in:
        manifest = json.loads(Path(args.manifest_in).read_text())
    else:
        manifest = {
            "seed": args.seed,
            "fold_strategy": "multilabel_stratified_recording",
            "class_labels": classes,
            "folds": make_folds(group_labels, args.folds, args.seed),
        }
    assert manifest["fold_strategy"] == "multilabel_stratified_recording"
    assert manifest["class_labels"] == classes
    assert len(manifest["folds"]) == args.folds
    validation = []
    all_groups = set(group_labels)
    for fold in manifest["folds"]:
        train = set(fold["train_groups"])
        val = set(fold["val_groups"])
        assert train.isdisjoint(val) and train | val == all_groups
        validation.extend(val)
    assert len(validation) == len(all_groups) == len(set(validation))
    if args.manifest_out:
        path = Path(args.manifest_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def fold_features(x, fit, train, val, components, seed):
    """PCA fit on the fold's `fit` tokens (never validation), then z-scoring on `train` tokens."""
    assert set(fit.tolist()).isdisjoint(val.tolist())
    train_x = np.asarray(x[train], dtype=np.float32)
    val_x = np.asarray(x[val], dtype=np.float32)
    if components:
        fit_x = np.asarray(x[fit], dtype=np.float32)
        assert 0 < components <= min(fit_x.shape)
        solver = "covariance_eigh" if components == x.shape[1] else "randomized"
        pca = PCA(n_components=components, svd_solver=solver, random_state=seed).fit(fit_x)
        train_x = pca.transform(train_x).astype(np.float32, copy=False)
        val_x = pca.transform(val_x).astype(np.float32, copy=False)
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    train_x -= mean
    train_x /= std
    val_x -= mean
    val_x /= std
    return train_x, val_x


def group_indices(groups, selected):
    selected = set(selected)
    return np.flatnonzero([group in selected for group in groups])


def ground_truth(units, stem, start, end):
    truth = np.zeros(max(0, end - start), dtype=np.int64)
    for onset, offset, label in units.get(stem, []):
        lo = max(start, onset)
        hi = min(end, offset)
        if lo < hi:
            truth[lo - start : hi - start] = label
    return truth


def confusion_matrix(predictions, spans, units, labels):
    label_index = {label: index for index, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for prediction, (stem, start, end) in zip(predictions, spans):
        pred_index = label_index[int(prediction)]
        truth, counts = np.unique(ground_truth(units, stem, start, end), return_counts=True)
        for label, count in zip(truth.tolist(), counts.tolist()):
            if label in label_index:
                confusion[label_index[label], pred_index] += count
    return confusion


def metrics(labels, confusion):
    true_frames = confusion.sum(axis=1)
    assert confusion.sum() and np.all(true_frames)
    predicted_frames = confusion.sum(axis=0)
    correct = np.diag(confusion)
    denominator = true_frames + predicted_frames
    f1 = np.divide(2 * correct, denominator, out=np.zeros(len(labels)), where=denominator > 0)
    per_class = {
        str(label): {
            "f1": float(f1[index]),
            "fer": float(1 - correct[index] / true_frames[index]),
            "frames": int(true_frames[index]),
        }
        for index, label in enumerate(labels)
    }
    return {
        "macro_f1": float(f1.mean()),
        "fer": float(1 - correct.sum() / confusion.sum()),
        **macro_fer_breakdown(labels, confusion),
        "frames": int(confusion.sum()),
        "classes": len(labels),
        "class_labels": labels,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def fit_fold(x, y, spans, groups, units, labels, fold, fold_index, args):
    train = group_indices(groups, fold["train_groups"])
    val = group_indices(groups, fold["val_groups"])
    assert set(y[train].tolist()) <= set(labels)  # a coarse model may never output some classes
    train_x, val_x = fold_features(x, train, train, val, args.pca_components, args.seed)

    started = time.perf_counter()
    model = LogisticRegression(
        C=args.logreg_c, class_weight="balanced", max_iter=args.max_iter
    )
    model.fit(train_x, y[train])
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions = model.predict(val_x)
    predict_seconds = time.perf_counter() - started
    confusion = confusion_matrix(
        predictions, [spans[index] for index in val], units, labels
    )
    row = metrics(labels, confusion)
    for key in ("class_labels", "confusion_matrix", "per_class"):
        del row[key]
    row.update(
        {
            "fold": fold_index,
            "train_recordings": len(fold["train_groups"]),
            "val_recordings": len(fold["val_groups"]),
            "train_tokens": int(train.size),
            "val_tokens": int(val.size),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
        }
    )
    return row, confusion


def isolated_fold(*args):
    context = multiprocessing.get_context("fork")
    queue = context.SimpleQueue()
    process = context.Process(target=lambda: queue.put(fit_fold(*args)))
    process.start()
    process.join()
    assert process.exitcode == 0, f"Fold process exited with {process.exitcode}."
    return queue.get()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--manifest_in")
    parser.add_argument("--manifest_out")
    parser.add_argument("--pca_components", type=int, default=128)
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--logreg_c", type=float, default=DEFAULT_LOGREG_C)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    started = time.perf_counter()
    args = parse_args()
    x, y, spans, groups = load_embeddings(args.embeddings)
    units = load_units(args.annotations)
    manifest = load_manifest(args, group_truth_labels(units, spans, groups))

    labels = manifest["class_labels"]
    total_confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    fold_metrics = []
    fit_seconds = 0.0
    predict_seconds = 0.0
    for fold_index, fold in enumerate(manifest["folds"]):
        fold_args = (x, y, spans, groups, units, labels, fold, fold_index, args)
        fold_row, fold_confusion = (
            isolated_fold(*fold_args)
            if args.pca_components == 0
            else fit_fold(*fold_args)
        )
        fold_metrics.append(fold_row)
        total_confusion += fold_confusion
        fit_seconds += fold_row["fit_seconds"]
        predict_seconds += fold_row["predict_seconds"]

    result = metrics(labels, total_confusion)
    result.update(
        {
            "encoder_scope": "frozen_final_layer",
            "classifier": "class_balanced_logistic_regression",
            "label_budget": "all_training_occurrences",
            "folds": args.folds,
            "fold_strategy": manifest["fold_strategy"],
            "event_grouping": "recording_stem",
            "scored_classes": "all_ground_truth_classes",
            "model_output_classes": sorted(set(y.tolist())),
            "event_split_integrity": "disjoint",
            "pca_components": args.pca_components,
            "pca_fit_scope": "disabled" if args.pca_components == 0 else "training_fold",
            "standardized": True,
            "standardization_fit_scope": (
                "training_fold_raw_features"
                if args.pca_components == 0
                else "training_fold_after_pca"
            ),
            "class_weight": "balanced",
            "logreg_c": args.logreg_c,
            "max_iter": args.max_iter,
            "fold_metrics": fold_metrics,
            "timing_seconds": {
                "fit": fit_seconds,
                "predict": predict_seconds,
                "total": time.perf_counter() - started,
            },
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
