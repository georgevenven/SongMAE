#!/usr/bin/env python3
"""Recording-disjoint individual-ID linear probe."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.embedding_store import EmbeddingStore
from src.evals.syllable_classification import pca_features, standardize

DEFAULT_LOGREG_C = 1e-3


def load_annotations(path):
    recordings = {}
    for row in json.loads(Path(path).read_text())["recordings"]:
        recording = row["recording"]
        stem = Path(recording["filename"]).stem
        assert stem not in recordings, f"duplicate recording stem: {stem}"
        recordings[stem] = {
            "bird_id": str(recording["bird_id"]),
            "events": [
                (float(event["onset_ms"]), float(event["offset_ms"]))
                for event in row.get("detected_events", [])
            ],
        }
    return recordings


def event_mask(stems, starts, ends, recordings):
    selected = []
    for stem, start, end in zip(stems.tolist(), starts.tolist(), ends.tolist()):
        assert stem in recordings, f"recording missing from annotations: {stem}"
        selected.append(
            any(start < offset and end > onset for onset, offset in recordings[stem]["events"])
        )
    return np.asarray(selected, dtype=bool)


def load_embeddings(path, annotations, audio_scope):
    data = EmbeddingStore(path)
    x = data["encoded_embeddings"]
    if x.ndim == 3:
        x = x[:, -1]
    x = x.reshape(x.shape[0], -1).astype(np.float32, copy=False)
    stems = np.asarray(data["recording_stem"]).astype(str)
    starts = np.asarray(data["token_start_ms"], dtype=np.float64)
    ends = np.asarray(data["token_end_ms"], dtype=np.float64)
    songs = np.asarray(data["song_id"])
    assert all(row.shape[0] == x.shape[0] for row in (stems, starts, ends, songs))

    events = event_mask(stems, starts, ends, annotations)
    keep = events if audio_scope == "events" else ~events
    assert keep.any(), f"no {audio_scope} patches found"
    x = x[keep]
    stems = stems[keep]
    songs = songs[keep]
    birds = np.asarray([annotations[stem]["bird_id"] for stem in stems], dtype=object)
    groups = np.asarray(
        [f"{stem}:{song}" for stem, song in zip(stems.tolist(), songs.tolist())],
        dtype=object,
    )
    return x, birds, stems, groups


def make_folds(recording_labels, count, seed):
    by_bird = {}
    for stem, bird in sorted(recording_labels.items()):
        by_bird.setdefault(bird, []).append(stem)
    assert len(by_bird) >= 2, "Need at least two birds."
    assert min(map(len, by_bird.values())) >= count, "Every bird needs one recording per fold."

    rng = np.random.default_rng(seed)
    validation = [[] for _ in range(count)]
    for stems in by_bird.values():
        for index, stem in enumerate(rng.permutation(stems).tolist()):
            validation[index % count].append(stem)
    all_stems = set(recording_labels)
    return [
        {
            "train_recordings": sorted(all_stems - set(val)),
            "val_recordings": sorted(val),
        }
        for val in validation
    ]


def folds_from_clip_map(path, recording_labels, count):
    rows = [
        row
        for row in json.loads(Path(path).read_text())
        if row["composite_stem"] in recording_labels
    ]
    assert {row["composite_stem"] for row in rows} == set(recording_labels)
    assert all(
        str(row["source_bird_id"]) == recording_labels[row["composite_stem"]]
        for row in rows
    )
    source_folds = {}
    for row in rows:
        source_folds.setdefault(row["source_stem"], row["fold"])
        assert source_folds[row["source_stem"]] == row["fold"]
    assert {row["fold"] for row in rows} == set(range(count))
    all_stems = set(recording_labels)
    folds = []
    for index in range(count):
        val = {row["composite_stem"] for row in rows if row["fold"] == index}
        folds.append({
            "train_recordings": sorted(all_stems - val),
            "val_recordings": sorted(val),
        })
    return folds


def load_manifest(args, recording_labels):
    class_labels = sorted(set(recording_labels.values()))
    assert not (args.manifest_in and args.clip_map)
    if args.manifest_in:
        manifest = json.loads(Path(args.manifest_in).read_text())
    elif args.clip_map:
        manifest = {
            "seed": args.seed,
            "fold_strategy": "source_recording_disjoint_clip_map",
            "split_integrity": "source_recording_disjoint",
            "class_labels": class_labels,
            "folds": folds_from_clip_map(args.clip_map, recording_labels, args.folds),
        }
    else:
        manifest = {
            "seed": args.seed,
            "fold_strategy": "stratified_recording",
            "split_integrity": "recording_disjoint",
            "class_labels": class_labels,
            "folds": make_folds(recording_labels, args.folds, args.seed),
        }
    assert manifest["class_labels"] == class_labels
    assert len(manifest["folds"]) == args.folds
    validation = []
    all_recordings = set(recording_labels)
    for fold in manifest["folds"]:
        train = set(fold["train_recordings"])
        val = set(fold["val_recordings"])
        assert train.isdisjoint(val) and train | val == all_recordings
        validation.extend(val)
    assert len(validation) == len(all_recordings) == len(set(validation))
    if args.manifest_out:
        path = Path(args.manifest_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def select_recordings(stems, selected):
    return np.flatnonzero(np.isin(stems, selected))


def aggregate_probabilities(y, probabilities, groups):
    grouped = {}
    for index, group in enumerate(groups.tolist()):
        grouped.setdefault(group, []).append(index)
    true = []
    mean_probabilities = []
    for indices in grouped.values():
        labels = y[indices]
        assert np.all(labels == labels[0])
        true.append(labels[0])
        mean_probabilities.append(probabilities[indices].mean(axis=0))
    return np.asarray(true), np.asarray(mean_probabilities, dtype=np.float32)


def metrics(y, predictions, labels):
    confusion = confusion_matrix(y, predictions, labels=np.arange(len(labels)))
    per_bird = {
        bird: {
            "accuracy": float(confusion[index, index] / confusion[index].sum()),
            "examples": int(confusion[index].sum()),
        }
        for index, bird in enumerate(labels)
    }
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro")),
        "examples": int(len(y)),
        "confusion_matrix": confusion.tolist(),
        "per_bird": per_bird,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument(
        "--audio_scope",
        choices=["events", "background"],
        required=True,
        help="Keep patches overlapping detected events or patches outside every event.",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--manifest_in")
    parser.add_argument("--manifest_out")
    parser.add_argument("--clip_map")
    parser.add_argument("--pca_components", type=int, default=128)
    parser.add_argument("--pca_cache")
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--logreg_c", type=float, default=DEFAULT_LOGREG_C)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    started = time.perf_counter()
    args = parse_args()
    annotations = load_annotations(args.annotations)
    x, bird_names, stems, groups = load_embeddings(
        args.embeddings, annotations, args.audio_scope
    )
    recording_labels = {stem: annotations[stem]["bird_id"] for stem in set(stems)}
    manifest = load_manifest(args, recording_labels)
    labels = manifest["class_labels"]
    label_index = {label: index for index, label in enumerate(labels)}
    y = np.asarray([label_index[bird] for bird in bird_names], dtype=np.int64)
    x, pca_seconds, cache_hit = pca_features(
        x, args.pca_components, args.seed, args.pca_cache
    )

    all_true = []
    all_predictions = []
    fold_metrics = []
    fit_seconds = 0.0
    predict_seconds = 0.0
    for fold_index, fold in enumerate(manifest["folds"]):
        train = select_recordings(stems, fold["train_recordings"])
        val = select_recordings(stems, fold["val_recordings"])
        assert set(y[train].tolist()) == set(range(len(labels)))
        train_x, val_x = standardize(x, train, val)

        fit_started = time.perf_counter()
        model = LogisticRegression(
            C=args.logreg_c, class_weight="balanced", max_iter=args.max_iter
        )
        model.fit(train_x, y[train])
        fit_elapsed = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        probabilities = model.predict_proba(val_x)
        predict_elapsed = time.perf_counter() - predict_started
        assert model.classes_.tolist() == list(range(len(labels)))
        true, mean_probabilities = aggregate_probabilities(y[val], probabilities, groups[val])
        predictions = mean_probabilities.argmax(axis=1)
        fold_row = metrics(true, predictions, labels)
        for key in ("confusion_matrix", "per_bird"):
            del fold_row[key]
        fold_row.update(
            {
                "fold": fold_index,
                "train_recordings": len(fold["train_recordings"]),
                "val_recordings": len(fold["val_recordings"]),
                "train_patches": int(train.size),
                "val_patches": int(val.size),
                "fit_seconds": fit_elapsed,
                "predict_seconds": predict_elapsed,
            }
        )
        fold_metrics.append(fold_row)
        all_true.append(true)
        all_predictions.append(predictions)
        fit_seconds += fit_elapsed
        predict_seconds += predict_elapsed

    result = metrics(np.concatenate(all_true), np.concatenate(all_predictions), labels)
    result.update(
        {
            "encoder_scope": "frozen_final_layer",
            "classifier": "class_balanced_logistic_regression",
            "audio_scope": args.audio_scope,
            "patch_training": True,
            "prediction_aggregation": "mean_patch_probability_per_recording_song",
            "folds": args.folds,
            "fold_strategy": manifest["fold_strategy"],
            "recording_split_integrity": manifest.get(
                "split_integrity", "recording_disjoint"
            ),
            "classes": len(labels),
            "class_labels": labels,
            "patches": int(len(y)),
            "pca_components": args.pca_components,
            "pca_fit_scope": (
                "disabled" if args.pca_components == 0 else "all_selected_patches"
            ),
            "pca_cache_hit": cache_hit,
            "standardized": True,
            "standardization_fit_scope": "training_fold_after_pca",
            "class_weight": "balanced",
            "logreg_c": args.logreg_c,
            "max_iter": args.max_iter,
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
