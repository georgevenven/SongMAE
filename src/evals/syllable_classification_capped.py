#!/usr/bin/env python3
"""Capped-label, recording-level cross-validated syllable linear probes for several label budgets at once.

PCA is fit once per fold on all tokens of the fold's training recordings (unlabeled audio is available to the
annotator), never on validation recordings, and shared by every budget; z-scoring and the probe use only the capped
labeled tokens. Writes <out_dir>/cap_<NNN>/metrics.json per budget.
"""
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
    fit_projection,
    drop_rare_classes,
    group_truth_labels,
    make_folds,
    metrics,
    zscore,
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


def build_manifest(y, spans, groups, group_labels, count, cap, seed):
    labels = sorted(set().union(*group_labels.values()))
    folds = make_folds(group_labels, count, seed)
    for fold_index, fold in enumerate(folds):
        train = group_indices(groups, fold["train_groups"])
        train = train[y[train] >= 0]
        fold["selected_occurrences"] = select_occurrences(
            y, spans, groups, train, sorted(set(y[train].tolist())), cap, seed, fold_index
        )
    return {
        "seed": seed,
        "fold_strategy": "multilabel_stratified_recording",
        "sampling": "nested_capped_occurrences",
        "label_cap": cap,
        "class_labels": labels,
        "folds": folds,
    }


def validate_manifest(manifest, group_labels, folds, cap):
    assert manifest["label_cap"] == cap
    assert manifest["fold_strategy"] == "multilabel_stratified_recording"
    assert manifest["class_labels"] == sorted(set().union(*group_labels.values()))
    assert len(manifest["folds"]) == folds
    all_groups = set(group_labels)
    validation = []
    for fold in manifest["folds"]:
        train = set(fold["train_groups"])
        val = set(fold["val_groups"])
        assert train.isdisjoint(val) and train | val == all_groups
        assert all(row["group"] in train for row in fold["selected_occurrences"])
        validation.extend(val)
    assert len(validation) == len(all_groups) == len(set(validation))


def load_manifest(args, y, spans, groups, group_labels, cap):
    path = Path(args.manifest_dir) / f"cap_{cap:03d}.json"
    if path.exists():
        manifest = json.loads(path.read_text())
    else:
        manifest = build_manifest(y, spans, groups, group_labels, args.folds, cap, args.seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    validate_manifest(manifest, group_labels, args.folds, cap)
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
    # Tokens of one recording come from several events extracted in balanced (not time) order; sort by start time
    # so the binary searches in matching_indices are valid.
    pools = {group: sorted(indices, key=lambda index: spans[index][1]) for group, indices in pools.items()}
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
    parser.add_argument("--label_caps", required=True, help="comma-separated occurrences per class, e.g. 1,5,10")
    parser.add_argument("--manifest_dir", required=True, help="holds cap_<NNN>.json; created if missing")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--pca_components", type=int, default=128)
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--logreg_c", type=float, default=DEFAULT_LOGREG_C)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    started = time.perf_counter()
    args = parse_args()
    caps = [int(cap) for cap in args.label_caps.split(",")]
    assert caps and min(caps) > 0
    x, y, spans, groups = load_embeddings(args.embeddings)
    units = load_units(args.annotations)
    group_labels, y, rare = drop_rare_classes(group_truth_labels(units, spans, groups), y, args.folds)
    manifests = {cap: load_manifest(args, y, spans, groups, group_labels, cap) for cap in caps}
    labels = manifests[caps[0]]["class_labels"]
    runs = {cap: {"confusion": np.zeros((len(labels), len(labels)), dtype=np.int64), "folds": [], "fit": 0.0, "predict": 0.0}
            for cap in caps}
    pca_seconds = 0.0
    for fold_index in range(args.folds):
        fold = manifests[caps[0]]["folds"][fold_index]
        assert all(manifests[cap]["folds"][fold_index]["val_groups"] == fold["val_groups"] for cap in caps)
        train = group_indices(groups, fold["train_groups"])
        val = group_indices(groups, fold["val_groups"])
        pca_started = time.perf_counter()
        project = fit_projection(x, train, args.pca_components, args.seed)
        val_projected = project(val)
        pca_seconds += time.perf_counter() - pca_started
        val_spans = [spans[index] for index in val]
        labeled = train[y[train] >= 0]  # PCA above uses all training audio; labels only from labeled tokens
        for cap in caps:
            fold_y = y.copy()
            selected, counts = selected_indices(
                fold_y, spans, groups, labeled, manifests[cap]["folds"][fold_index]["selected_occurrences"],
                labels, cap, args.seed, fold_index,
            )
            assert set(fold_y[selected].tolist()) == set(y[labeled].tolist())
            train_x, val_x = zscore(project(selected), val_projected)
            fit_started = time.perf_counter()
            model = LogisticRegression(C=args.logreg_c, class_weight="balanced", max_iter=args.max_iter)
            model.fit(train_x, fold_y[selected])
            fit_elapsed = time.perf_counter() - fit_started
            predict_started = time.perf_counter()
            predictions = model.predict(val_x)
            predict_elapsed = time.perf_counter() - predict_started
            fold_confusion = confusion_matrix(predictions, val_spans, units, labels)
            fold_row = metrics(labels, fold_confusion)
            for key in ("class_labels", "confusion_matrix", "per_class"):
                del fold_row[key]
            fold_row.update({
                "fold": fold_index,
                "train_recordings": len(fold["train_groups"]),
                "val_recordings": len(fold["val_groups"]),
                "train_tokens": int(selected.size),
                "val_tokens": int(val.size),
                "labeled_occurrences_by_class": {str(label): counts.get(label, 0) for label in labels},
                "fit_seconds": fit_elapsed,
                "predict_seconds": predict_elapsed,
            })
            run = runs[cap]
            run["folds"].append(fold_row)
            run["confusion"] += fold_confusion
            run["fit"] += fit_elapsed
            run["predict"] += predict_elapsed

    for cap in caps:
        run = runs[cap]
        result = metrics(labels, run["confusion"])
        occurrence_counts = [
            count for fold in run["folds"] for label, count in fold["labeled_occurrences_by_class"].items() if label != "0"
        ]
        result.update({
            "encoder_scope": "frozen_final_layer",
            "classifier": "class_balanced_logistic_regression",
            "label_cap": cap,
            "label_budget": "at_most_occurrences_per_class",
            "folds": args.folds,
            "fold_strategy": manifests[cap]["fold_strategy"],
            "sampling": manifests[cap]["sampling"],
            "event_grouping": "recording_stem",
            "scored_classes": "all_ground_truth_classes",
            "excluded_rare_classes": rare,
            "event_split_integrity": "disjoint",
            "pca_components": args.pca_components,
            "pca_fit_scope": "disabled" if args.pca_components == 0 else "training_fold_all_tokens",
            "standardized": True,
            "standardization_fit_scope": (
                "labeled_tokens_raw_features" if args.pca_components == 0 else "labeled_tokens_after_pca"
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
            "fold_metrics": run["folds"],
            "timing_seconds": {"pca_shared": pca_seconds, "fit": run["fit"], "predict": run["predict"],
                               "total_all_caps": time.perf_counter() - started},
        })
        out = Path(args.out_dir) / f"cap_{cap:03d}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.tmp").write_text(json.dumps(result, indent=2) + "\n")
        (out / "metrics.tmp").replace(out / "metrics.json")
        print(f"cap {cap}: macro FER {result['macro_fer']:.4f} -> {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
