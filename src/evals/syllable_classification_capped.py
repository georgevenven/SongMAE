#!/usr/bin/env python3
"""Capped-label, song-level cross-validated syllable linear probe."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.evals.syllable_classification import (
    DEFAULT_LOGREG_C,
    confusion_matrix,
    group_indices,
    load_embeddings,
    load_units,
    make_folds,
    metrics,
    pca_features,
    standardize,
)


def occurrence_runs(y, spans, groups, indices):
    runs = []
    current = []
    for index in indices:
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
    }


def select_occurrences(y, spans, groups, train, labels, cap, seed, fold_index):
    by_label = occurrence_runs(y, spans, groups, train)
    selected = []
    for label in labels:
        runs = by_label[label]
        rng = np.random.default_rng(np.random.SeedSequence([seed, fold_index, label]))
        selected.extend(
            occurrence(runs[index], label, spans, groups)
            for index in rng.permutation(len(runs))[:cap]
        )
    return selected


def build_manifest(y, spans, groups, count, cap, seed):
    labels = sorted(set(y.tolist()))
    folds = make_folds(y, groups, count, seed)
    for fold_index, fold in enumerate(folds):
        train = group_indices(groups, fold["train_groups"])
        fold["selected_occurrences"] = select_occurrences(
            y, spans, groups, train, labels, cap, seed, fold_index
        )
    return {
        "seed": seed,
        "fold_strategy": "multilabel_stratified_song",
        "sampling": "nested_capped_occurrences",
        "label_cap": cap,
        "class_labels": labels,
        "folds": folds,
    }


def validate_manifest(manifest, y, groups, folds, cap):
    assert manifest["label_cap"] == cap
    assert manifest["class_labels"] == sorted(set(y.tolist()))
    assert len(manifest["folds"]) == folds
    all_groups = set(groups)
    validation = []
    for fold in manifest["folds"]:
        train = set(fold["train_groups"])
        val = set(fold["val_groups"])
        assert train.isdisjoint(val) and train | val == all_groups
        assert all(row["group"] in train for row in fold["selected_occurrences"])
        validation.extend(val)
    assert len(validation) == len(all_groups) == len(set(validation))


def load_manifest(args, y, spans, groups):
    if args.manifest_in:
        manifest = json.loads(Path(args.manifest_in).read_text())
    else:
        manifest = build_manifest(y, spans, groups, args.folds, args.label_cap, args.seed)
    validate_manifest(manifest, y, groups, args.folds, args.label_cap)
    if args.manifest_out:
        path = Path(args.manifest_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def matching_indices(y, spans, pool, row, assigned):
    indices, starts, ends = pool
    first = np.searchsorted(ends, row["start_ms"], side="right")
    last = np.searchsorted(starts, row["end_ms"], side="left")
    overlapping = indices[first:last]
    matching = [
        index
        for index in overlapping
        if int(y[index]) == row["label"]
        and assigned.get(index, row["label"]) == row["label"]
    ]
    if matching:
        return matching
    available = [index for index in overlapping if index not in assigned]
    if not available:
        return []
    return [
        max(
            available,
            key=lambda index: min(spans[index][2], row["end_ms"])
            - max(spans[index][1], row["start_ms"]),
        )
    ]


def selected_indices(y, spans, groups, train, rows, labels, cap, seed, fold_index):
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
    for index in train:
        pools.setdefault(groups[index], []).append(index)
    pools = {
        group: (
            indices,
            np.array([spans[index][1] for index in indices]),
            np.array([spans[index][2] for index in indices]),
        )
        for group, indices in pools.items()
    }
    empty_pool = ([], np.array([]), np.array([]))
    selected = []
    counts = {}
    assigned = {}
    for row in rows:
        matches = matching_indices(y, spans, pools.get(row["group"], empty_pool), row, assigned)
        if row["label"] == 0 and not any(int(y[index]) == 0 for index in matches):
            row = next(background)
            matches = matching_indices(
                y, spans, pools.get(row["group"], empty_pool), row, assigned
            )
        if not matches:
            continue
        for index in matches:
            assert assigned.get(index, row["label"]) == row["label"]
            assigned[index] = row["label"]
            y[index] = row["label"]
        selected.extend(matches)
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    return np.array(sorted(set(selected)), dtype=np.int64), counts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--label_cap", type=int, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--manifest_in")
    parser.add_argument("--manifest_out")
    parser.add_argument("--pca_components", type=int, default=128)
    parser.add_argument("--pca_cache")
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--logreg_c", type=float, default=DEFAULT_LOGREG_C)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    started = time.perf_counter()
    args = parse_args()
    assert args.label_cap > 0
    x, y, spans, groups = load_embeddings(args.embeddings)
    units = load_units(args.annotations)
    manifest = load_manifest(args, y, spans, groups)
    x, pca_seconds, cache_hit = pca_features(
        x, args.pca_components, args.seed, args.pca_cache
    )

    labels = manifest["class_labels"]
    total_confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    fold_metrics = []
    fit_seconds = 0.0
    predict_seconds = 0.0
    for fold_index, fold in enumerate(manifest["folds"]):
        fold_y = y.copy()
        train = group_indices(groups, fold["train_groups"])
        val = group_indices(groups, fold["val_groups"])
        selected, counts = selected_indices(
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
        assert set(fold_y[selected].tolist()) == set(labels)
        train_x, val_x = standardize(x, selected, val)

        fit_started = time.perf_counter()
        model = LogisticRegression(
            C=args.logreg_c, class_weight="balanced", max_iter=args.max_iter
        )
        model.fit(train_x, fold_y[selected])
        fit_elapsed = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        predictions = model.predict(val_x)
        predict_elapsed = time.perf_counter() - predict_started
        fold_confusion = confusion_matrix(
            predictions, [spans[index] for index in val], units, labels
        )
        fold_row = metrics(labels, fold_confusion)
        for key in ("class_labels", "confusion_matrix", "per_class"):
            del fold_row[key]
        fold_row.update(
            {
                "fold": fold_index,
                "train_songs": len(fold["train_groups"]),
                "val_songs": len(fold["val_groups"]),
                "train_tokens": int(selected.size),
                "val_tokens": int(val.size),
                "labeled_occurrences_by_class": {
                    str(label): counts.get(label, 0) for label in labels
                },
                "fit_seconds": fit_elapsed,
                "predict_seconds": predict_elapsed,
            }
        )
        fold_metrics.append(fold_row)
        total_confusion += fold_confusion
        fit_seconds += fit_elapsed
        predict_seconds += predict_elapsed

    result = metrics(labels, total_confusion)
    occurrence_counts = [
        count
        for fold in fold_metrics
        for label, count in fold["labeled_occurrences_by_class"].items()
        if label != "0"
    ]
    result.update(
        {
            "encoder_scope": "frozen_final_layer",
            "classifier": "class_balanced_logistic_regression",
            "label_cap": args.label_cap,
            "label_budget": "at_most_occurrences_per_class",
            "folds": args.folds,
            "fold_strategy": manifest["fold_strategy"],
            "sampling": manifest["sampling"],
            "event_grouping": "recording_stem:song_id",
            "event_split_integrity": "disjoint",
            "pca_components": args.pca_components,
            "pca_fit_scope": "disabled" if args.pca_components == 0 else "all_extracted_tokens",
            "pca_cache_hit": cache_hit,
            "standardized": True,
            "standardization_fit_scope": (
                "training_fold_raw_features"
                if args.pca_components == 0
                else "training_fold_after_pca"
            ),
            "class_weight": "balanced",
            "logreg_c": args.logreg_c,
            "max_iter": args.max_iter,
            "labeled_occurrences": {
                "scope": "syllable_classes",
                "median": float(np.median(occurrence_counts)),
                "min": min(occurrence_counts),
                "max": max(occurrence_counts),
            },
            "fold_metrics": fold_metrics,
            "timing_seconds": {
                "pca": pca_seconds,
                "fit": fit_seconds,
                "predict": predict_seconds,
                "total": time.perf_counter() - started,
            },
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
