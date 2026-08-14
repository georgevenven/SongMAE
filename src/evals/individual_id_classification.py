#!/usr/bin/env python3
"""Recording-disjoint centroid and token individual-ID linear probes."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.embedding_store import EmbeddingStore

MAX_GROUP_MS = 5_000
DEFAULT_LOGREG_C = 1e-3
METHODS = ("centroid", "token")


def load_annotations(path):
    recordings = {}
    for row in json.loads(Path(path).read_text())["recordings"]:
        recording = row["recording"]
        stem = Path(recording["filename"]).stem
        assert stem not in recordings
        recordings[stem] = {
            "bird": str(recording["bird_id"]),
            "events": [
                (float(event["onset_ms"]), float(event["offset_ms"]))
                for event in row.get("detected_events", [])
            ],
        }
    return recordings


def validate_metadata(metadata):
    if "max_segment_timebins" in metadata:
        bins = int(metadata["max_segment_timebins"])
        duration_ms = bins * int(metadata["audio_hop_size"]) * 1_000 / int(metadata["audio_sr"])
        assert 0 < duration_ms <= MAX_GROUP_MS
        assert metadata["per_segment_normalize"], "spectrogram segments must be z-scored and projected to dataset statistics"
    if "chunk_timebins" in metadata:
        assert 0 < int(metadata["chunk_timebins"]) <= 1_000


def load_embeddings(path, annotations, audio_scope, layer=None):
    store = EmbeddingStore(path)
    validate_metadata(store.metadata)
    x = store["encoded_embeddings"]
    if x.ndim == 3:
        selected = -1 if layer is None else layer
        assert -x.shape[1] <= selected < x.shape[1], f"layer {selected} outside {x.shape[1]} stored layers"
        x = x[:, selected]
    elif layer is not None:
        assert store.metadata.get("encoder_layer_idx") == layer
    x = x.reshape(x.shape[0], -1).astype(np.float32, copy=False)
    stems = np.asarray(store["recording_stem"]).astype(str)
    starts = np.asarray(store["token_start_ms"], dtype=np.float64)
    ends = np.asarray(store["token_end_ms"], dtype=np.float64)
    songs = np.asarray(store["song_id"]).astype(str)
    assert all(len(row) == len(x) for row in (stems, starts, ends, songs))
    assert set(stems) <= set(annotations)

    centers = (starts + ends) / 2
    origins = {}
    for stem, song, center in zip(stems, songs, centers):
        origins.setdefault((stem, song), center)
        origins[stem, song] = min(origins[stem, song], center)

    keep, groups, kinds = [], [], []
    for index, (stem, song, center) in enumerate(zip(stems, songs, centers)):
        is_song = any(onset <= center < offset for onset, offset in annotations[stem]["events"])
        if audio_scope == "song" and not is_song:
            continue
        kind = "song" if is_song else "non_song"
        window = int((center - origins[stem, song]) // MAX_GROUP_MS)
        keep.append(index)
        groups.append(f"{stem}:{song}:{window}:{kind}")
        kinds.append(kind)

    assert keep, f"no {audio_scope} tokens found"
    keep = np.asarray(keep)
    stems = stems[keep]
    return {
        "x": x[keep],
        "y": np.asarray([annotations[stem]["bird"] for stem in stems]),
        "stems": stems,
        "groups": np.asarray(groups),
        "kinds": np.asarray(kinds),
        "metadata": store.metadata,
    }


def make_folds(recording_labels, count, seed):
    by_bird = {}
    for stem, bird in sorted(recording_labels.items()):
        by_bird.setdefault(bird, []).append(stem)
    assert len(by_bird) >= 2
    assert min(map(len, by_bird.values())) >= count, "every bird needs one recording per fold"
    rng = np.random.default_rng(seed)
    validation = [[] for _ in range(count)]
    for stems in by_bird.values():
        for index, stem in enumerate(rng.permutation(stems)):
            validation[index % count].append(stem)
    all_stems = set(recording_labels)
    return [
        {"train_recordings": sorted(all_stems - set(val)), "val_recordings": sorted(val)}
        for val in validation
    ]


def load_manifest(args, recording_labels, labels):
    if args.manifest_in:
        manifest = json.loads(Path(args.manifest_in).read_text())
    else:
        manifest = {
            "seed": args.seed,
            "fold_strategy": "stratified_recording",
            "class_labels": labels,
            "folds": make_folds(recording_labels, args.folds, args.seed),
        }
    assert manifest["class_labels"] == labels and len(manifest["folds"]) == args.folds
    validation = []
    all_stems = set(recording_labels)
    for fold in manifest["folds"]:
        train, val = set(fold["train_recordings"]), set(fold["val_recordings"])
        assert train.isdisjoint(val) and train | val == all_stems
        validation.extend(val)
    assert len(validation) == len(all_stems) == len(set(validation))
    if args.manifest_out:
        path = Path(args.manifest_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def group_rows(x, y, groups):
    keys = list(dict.fromkeys(groups.tolist()))
    rows, labels = [], []
    for key in keys:
        indices = np.flatnonzero(groups == key)
        assert np.all(y[indices] == y[indices[0]])
        rows.append(x[indices].mean(axis=0))
        labels.append(y[indices[0]])
    return np.asarray(rows, dtype=np.float32), np.asarray(labels), np.asarray(keys)


def prepare_features(train_x, val_x, components, seed):
    if components:
        assert components <= min(train_x.shape)
        pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
        train_x = pca.fit_transform(train_x)
        val_x = pca.transform(val_x)
    mean = train_x.mean(axis=0, dtype=np.float64)
    std = np.maximum(train_x.std(axis=0, dtype=np.float64), 1e-6)
    return (
        ((train_x - mean) / std).astype(np.float32),
        ((val_x - mean) / std).astype(np.float32),
    )


def aggregate_probabilities(y, probabilities, groups):
    averaged, true, keys = group_rows(probabilities, y, groups)
    return true, averaged, keys


def metrics(matrix, labels):
    true = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = np.diag(matrix)
    assert matrix.sum() and np.all(true)
    f1 = np.divide(2 * correct, true + predicted, out=np.zeros(len(labels)), where=true + predicted > 0)
    return {
        "accuracy": float(correct.sum() / matrix.sum()),
        "macro_f1": float(f1.mean()),
        "examples": int(matrix.sum()),
        "confusion_matrix": matrix.tolist(),
        "per_bird": {
            bird: {
                "accuracy": float(correct[index] / true[index]),
                "examples": int(true[index]),
            }
            for index, bird in enumerate(labels)
        },
    }


def score(y, probabilities, labels):
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(y, predictions, labels=np.arange(len(labels)))
    return metrics(matrix, labels), matrix


def fit_method(method, train_x, train_y, train_groups, val_x, val_y, val_groups, labels, args):
    if method == "centroid":
        train_x, train_y, _ = group_rows(train_x, train_y, train_groups)
    elif method != "token":
        raise ValueError(method)

    model = LogisticRegression(C=args.logreg_c, class_weight="balanced", max_iter=args.max_iter)
    model.fit(train_x, train_y)
    assert np.array_equal(model.classes_, np.arange(len(labels)))
    if method == "centroid":
        val_x, val_y, _ = group_rows(val_x, val_y, val_groups)
        probabilities = model.predict_proba(val_x)
    else:
        val_y, probabilities, _ = aggregate_probabilities(val_y, model.predict_proba(val_x), val_groups)
    return score(val_y, probabilities, labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--audio_scope", choices=("song", "song_and_non_song"), required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--manifest_in")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--manifest_out")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--pca_components", type=int, default=128)
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--logreg_c", type=float, default=DEFAULT_LOGREG_C)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    started = time.perf_counter()

    annotations = load_annotations(args.annotations)
    data = load_embeddings(args.embeddings, annotations, args.audio_scope, args.layer)
    labels = sorted(set(data["y"].tolist()))
    label_index = {label: index for index, label in enumerate(labels)}
    data["y"] = np.asarray([label_index[label] for label in data["y"]])
    recording_labels = {stem: annotations[stem]["bird"] for stem in sorted(set(data["stems"]))}
    manifest = load_manifest(args, recording_labels, labels)
    matrices = {args.method: np.zeros((len(labels), len(labels)), dtype=np.int64)}
    fold_metrics = {args.method: []}

    for fold_index, fold in enumerate(manifest["folds"]):
        train_indices = np.flatnonzero(np.isin(data["stems"], fold["train_recordings"]))
        val_indices = np.flatnonzero(np.isin(data["stems"], fold["val_recordings"]))
        train_x, val_x = prepare_features(
            data["x"][train_indices], data["x"][val_indices], args.pca_components, args.seed + fold_index
        )
        row, matrix = fit_method(
            args.method,
            train_x,
            data["y"][train_indices],
            data["groups"][train_indices],
            val_x,
            data["y"][val_indices],
            data["groups"][val_indices],
            labels,
            args,
        )
        fold_metrics[args.method].append({
            "fold": fold_index,
            **{key: value for key, value in row.items() if key not in ("confusion_matrix", "per_bird")},
        })
        matrices[args.method] += matrix

    results = {
        args.method: {
            **metrics(matrices[args.method], labels),
            "fold_metrics": fold_metrics[args.method],
        }
    }

    output = {
        "task": "closed_set_individual_id",
        "methods": results,
        "class_labels": labels,
        "condition": args.condition,
        "audio_scope": args.audio_scope,
        "maximum_group_seconds": MAX_GROUP_MS / 1_000,
        "method": args.method,
        "layer": args.layer,
        "folds": args.folds,
        "fold_strategy": manifest["fold_strategy"],
        "pca_components": args.pca_components,
        "standardization_fit_scope": "training_fold_tokens",
        "logreg_c": args.logreg_c,
        "seconds": time.perf_counter() - started,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
