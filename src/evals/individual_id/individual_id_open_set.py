#!/usr/bin/env python3
"""Recording-level open-set individual identification from enrollment songs."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_auc_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.embedding_store import EmbeddingStore

METHODS = ("prototype_cosine", "enrollment_knn", "knn_support")
METRICS = (
    "known_id_accuracy",
    "known_correct_accept_rate",
    "known_accept_rate",
    "unknown_rejection_rate",
    "open_set_accuracy",
    "auroc",
    "oscr",
    "threshold",
)


def evenly_spaced(values, limit):
    if limit == 0 or len(values) <= limit:
        return values
    return values[np.linspace(0, len(values) - 1, limit, dtype=int)]


def load_recordings(embedding_path, clip_map_path, condition, max_points):
    store = EmbeddingStore(embedding_path)
    features = store["encoded_embeddings"]
    if features.ndim == 3:
        features = features[:, -1]
    stems = np.asarray(store["recording_stem"]).astype(str)
    order = np.argsort(stems)
    unique, starts = np.unique(stems[order], return_index=True)
    ends = np.r_[starts[1:], len(order)]
    stem_indices = {
        stem: order[start:end]
        for stem, start, end in zip(unique.tolist(), starts.tolist(), ends.tolist())
    }

    rows = [
        row
        for row in json.loads(Path(clip_map_path).read_text())
        if row["condition"] == condition
    ]
    source_clips = defaultdict(list)
    source_birds = {}
    for row in rows:
        source = row["source_stem"]
        source_clips[source].append(row["composite_stem"])
        source_birds.setdefault(source, str(row["source_bird_id"]))
        assert source_birds[source] == str(row["source_bird_id"])
        assert row["composite_stem"] in stem_indices

    recordings = {}
    by_bird = defaultdict(list)
    excluded = 0
    for source, clips in source_clips.items():
        indices = np.concatenate([stem_indices[clip] for clip in clips])
        indices = evenly_spaced(indices, max_points)
        points = np.asarray(features[indices], dtype=np.float32)
        if len(points) < 2:
            excluded += 1
            continue
        recordings[source] = points
        by_bird[source_birds[source]].append(source)
    return recordings, {bird: sorted(sources) for bird, sources in by_bird.items()}, excluded


def split_episode(by_bird, shots, query_limit, rng):
    birds = np.asarray(sorted(by_bird))
    eligible = np.asarray([bird for bird in birds if len(by_bird[bird]) >= shots + 2])
    if len(eligible) < 2 or len(birds) < 4:
        return None
    rng.shuffle(eligible)
    known_count = min(10, len(eligible), len(birds) - 2)
    known = eligible[:known_count].tolist()
    unknown = np.asarray([bird for bird in birds if bird not in known])
    rng.shuffle(unknown)
    cut = max(1, len(unknown) // 2)
    calibration_unknown = unknown[:cut].tolist()
    test_unknown = unknown[cut:].tolist()
    if not test_unknown:
        return None

    enrollment = {}
    calibration_known = []
    test_known = []
    for bird in known:
        sources = rng.permutation(by_bird[bird]).tolist()
        enrollment[bird] = sources[:shots]
        calibration_known.append((sources[shots], bird))
        test_known.extend((source, bird) for source in sources[shots + 1:shots + 1 + query_limit])
    calibration_unknown = [
        (source, bird)
        for bird in calibration_unknown
        for source in rng.permutation(by_bird[bird])[:query_limit]
    ]
    test_unknown = [
        (source, bird)
        for bird in test_unknown
        for source in rng.permutation(by_bird[bird])[:query_limit]
    ]
    return enrollment, calibration_known, test_known, calibration_unknown, test_unknown


def normalize(values):
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


def prepare_support(recordings, split, components, fit_points, seed):
    enrollment, calibration_known, test_known, calibration_unknown, test_unknown = split
    fit_sources = [source for sources in enrollment.values() for source in sources]
    fit = np.concatenate([evenly_spaced(recordings[source], fit_points) for source in fit_sources])
    pca = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=seed,
    ).fit(fit)
    projected = {
        source: pca.transform(recordings[source]).astype(np.float32)
        for source in fit_sources
    }

    models = {}
    for bird, sources in enrollment.items():
        groups = [projected[source] for source in sources]
        points = np.concatenate(groups)
        models[bird] = {
            "prototype": normalize(np.mean([group.mean(axis=0) for group in groups], axis=0)),
            "points": points,
            "normalized_points": normalize(points),
            "radii": NearestNeighbors(n_neighbors=1).fit(points).kneighbors()[0][:, 0],
        }
    return pca, models, calibration_known, test_known, calibration_unknown, test_unknown


def project_queries(pca, recordings, queries):
    return {
        source: pca.transform(recordings[source]).astype(np.float32)
        for source, _ in queries
    }


def score_query(query, birds, models):
    normalized_query = normalize(query)
    query_prototype = normalize(query.mean(axis=0))
    scores = {method: [] for method in METHODS}
    for bird in birds:
        model = models[bird]
        scores["prototype_cosine"].append(float(query_prototype @ model["prototype"]))
        cosine = normalized_query @ model["normalized_points"].T
        scores["enrollment_knn"].append(float(cosine.max(axis=1).mean()))
        distance = cdist(query, model["points"])
        scores["knn_support"].append(float(np.any(distance <= model["radii"], axis=1).mean()))
    return {method: np.asarray(values) for method, values in scores.items()}


def score_queries(projected, models, queries):
    birds = sorted(models)
    output = {method: {"confidence": [], "correct": []} for method in METHODS}
    for source, truth in queries:
        scores = score_query(projected[source], birds, models)
        for method, values in scores.items():
            output[method]["confidence"].append(float(values.max()))
            output[method]["correct"].append(birds[int(values.argmax())] == truth)
    return output


def calibrate_threshold(known, unknown):
    candidates = np.unique(np.r_[known, unknown])
    objective = [np.mean(known >= value) + np.mean(unknown < value) for value in candidates]
    return float(candidates[int(np.argmax(objective))])


def oscr(correct, known, unknown):
    thresholds = np.r_[np.inf, np.sort(np.unique(np.r_[known, unknown]))[::-1], -np.inf]
    false_positive = np.asarray([np.mean(unknown >= value) for value in thresholds])
    correct_classification = np.asarray([
        np.mean(correct & (known >= value)) for value in thresholds
    ])
    return float(auc(false_positive, correct_classification))


def evaluate(calibration_known, calibration_unknown, test_known, test_unknown):
    calibration_known = np.asarray(calibration_known["confidence"])
    calibration_unknown = np.asarray(calibration_unknown["confidence"])
    known = np.asarray(test_known["confidence"])
    correct = np.asarray(test_known["correct"], dtype=bool)
    unknown = np.asarray(test_unknown["confidence"])
    threshold = calibrate_threshold(calibration_known, calibration_unknown)
    correct_accept = correct & (known >= threshold)
    unknown_reject = unknown < threshold
    return {
        "known_id_accuracy": float(correct.mean()),
        "known_correct_accept_rate": float(correct_accept.mean()),
        "known_accept_rate": float(np.mean(known >= threshold)),
        "unknown_rejection_rate": float(unknown_reject.mean()),
        "open_set_accuracy": float((correct_accept.sum() + unknown_reject.sum()) / (len(known) + len(unknown))),
        "auroc": float(roc_auc_score(np.r_[np.ones(len(known)), np.zeros(len(unknown))], np.r_[known, unknown])),
        "oscr": oscr(correct, known, unknown),
        "threshold": threshold,
    }


def summarize(rows, shots):
    output = []
    selected = [row for row in rows if row["shots"] == shots]
    for method in METHODS:
        method_rows = [row for row in selected if row["method"] == method]
        if not method_rows:
            continue
        output.append({
            "shots": shots,
            "method": method,
            "episodes": len(method_rows),
            "known_individuals_mean": float(np.mean([row["known_individuals"] for row in method_rows])),
            **{
                key: {
                    "mean": float(np.mean([row[key] for row in method_rows])),
                    "std": float(np.std([row[key] for row in method_rows])),
                }
                for key in METRICS
            },
        })
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--clip_map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--shots", default="1,2,4,8,16")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pca_components", type=int, default=32)
    parser.add_argument("--pca_fit_points_per_recording", type=int, default=32)
    parser.add_argument("--max_points_per_recording", type=int, default=64)
    parser.add_argument("--queries_per_identity", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    shots = [int(value) for value in args.shots.split(",")]
    recordings, by_bird, excluded = load_recordings(
        args.embeddings, args.clip_map, "clean", args.max_points_per_recording
    )
    rows = []
    unsupported = []
    for k in shots:
        completed = 0
        for repeat in range(args.repeats):
            seed = args.seed + 1000 * k + repeat
            split = split_episode(
                by_bird, k, args.queries_per_identity, np.random.default_rng(seed)
            )
            if split is None:
                break
            prepared = prepare_support(
                recordings,
                split,
                args.pca_components,
                args.pca_fit_points_per_recording,
                seed,
            )
            pca, models, calibration_known, test_known, calibration_unknown, test_unknown = prepared
            queries = calibration_known + test_known + calibration_unknown + test_unknown
            projected = project_queries(pca, recordings, queries)
            calibration_known_scores = score_queries(projected, models, calibration_known)
            calibration_unknown_scores = score_queries(projected, models, calibration_unknown)
            test_known_scores = score_queries(projected, models, test_known)
            test_unknown_scores = score_queries(projected, models, test_unknown)
            for method in METHODS:
                rows.append({
                    "shots": k,
                    "repeat": repeat,
                    "method": method,
                    "known_individuals": len(models),
                    "calibration_unknown_individuals": len(calibration_unknown),
                    "test_unknown_individuals": len(test_unknown),
                    **evaluate(
                        calibration_known_scores[method],
                        calibration_unknown_scores[method],
                        test_known_scores[method],
                        test_unknown_scores[method],
                    ),
                })
            completed += 1
        if completed == 0:
            unsupported.append(k)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "species": args.species,
        "model": args.model,
        "individuals": len(by_bird),
        "recordings": len(recordings),
        "excluded_short_recordings": excluded,
        "shots": shots,
        "unsupported_shots": unsupported,
        "repeats": args.repeats,
        "pca_fit_scope": "enrollment_recordings_only",
        "pca_components": args.pca_components,
        "support_neighbors": 1,
        "max_points_per_recording": args.max_points_per_recording,
        "queries_per_identity": args.queries_per_identity,
        "known_unknown_split": "identity_disjoint; unknown calibration and test identities disjoint",
        "summaries": [summary for k in shots for summary in summarize(rows, k)],
        "episodes": rows,
    }
    output.write_text(json.dumps(summary, indent=2) + "\n")
    with output.with_suffix(".tsv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "species": args.species,
        "model": args.model,
        "output": str(output),
        "episodes": len(rows) // len(METHODS),
        "unsupported_shots": unsupported,
    }))


if __name__ == "__main__":
    main()
