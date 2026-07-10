#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.embedding_store import EmbeddingStore
from src.evals.syllable_classification import load_units, token_groups, token_spans


@dataclass(frozen=True)
class Sequence:
    key: str
    stem: str
    features: np.ndarray
    starts: np.ndarray
    ends: np.ndarray


@dataclass(frozen=True)
class Query:
    sequence: Sequence
    label: int
    onset: int
    offset: int
    start: int
    end: int


def split_sequences(data, features):
    stems, starts, ends = token_spans(data, len(features))
    groups = np.asarray(token_groups(data, stems, len(features)))
    bounds = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(groups)]
    assert len(set(groups)) == len(bounds) - 1, "embedding groups must be contiguous"
    sequences = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        assert len(set(stems[start:end])) == 1
        sequences.append(Sequence(groups[start], stems[start], features[start:end], starts[start:end], ends[start:end]))
    return sequences


def load_sequences(path, feature_key):
    data = EmbeddingStore(path)
    features = data[feature_key].astype(np.float32, copy=False)
    features = features.reshape(features.shape[0], -1)
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    return split_sequences(data, features), data.metadata


def token_slice(sequence, onset, offset):
    center = (sequence.starts + sequence.ends) / 2
    index = np.flatnonzero((center >= onset) & (center < offset))
    if index.size:
        return int(index[0]), int(index[-1] + 1)
    nearest = int(np.argmin(abs(center - (onset + offset) / 2)))
    return nearest, nearest + 1


def select_queries(sequences, units, limit, seed):
    by_stem = {}
    for sequence in sequences:
        by_stem.setdefault(sequence.stem, []).append(sequence)

    by_label = {}
    for stem, items in units.items():
        for onset, offset, label in items:
            middle = (onset + offset) / 2
            matches = [r for r in by_stem.get(stem, []) if r.starts[0] <= middle < r.ends[-1]]
            if not matches:
                continue
            sequence = matches[0]
            start, end = token_slice(sequence, onset, offset)
            by_label.setdefault(label, []).append(Query(sequence, label, onset, offset, start, end))
    rng = np.random.default_rng(seed)
    queries = []
    for label in sorted(by_label):
        rng.shuffle(by_label[label])
        one_per_recording = {}
        for query in by_label[label]:
            one_per_recording.setdefault(query.sequence.stem, query)
        choices = list(one_per_recording.values())
        if len(choices) < 2:
            continue
        rng.shuffle(choices)
        queries.extend(choices[:limit])
    return queries


def trajectory_scores(target, query):
    length = len(query)
    count = len(target) - length + 1
    if count <= 0:
        return np.empty(0, dtype=np.float32)
    similarity = target @ query.T
    scores = np.zeros(count, dtype=np.float32)
    for offset in range(length):
        scores += similarity[offset : offset + count, offset]
    return scores / length


def nms(scores, width):
    blocked = np.zeros(len(scores), dtype=bool)
    peaks = []
    radius = width // 2
    for index in np.argsort(-scores, kind="stable"):
        if blocked[index]:
            continue
        peaks.append(int(index))
        blocked[max(0, index - radius) : index + radius + 1] = True
    return peaks


def event_metrics(matches, positives):
    hits = np.asarray([item["status"] == "tp" for item in matches])
    ranks = np.flatnonzero(hits) + 1
    ap = float((np.cumsum(hits)[ranks - 1] / ranks).sum() / positives)
    r_precision = float(hits[:positives].sum() / positives)
    return ap, r_precision


def ranked_matches(candidates, truth):
    candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
    matched = {key: np.zeros(len(items), dtype=bool) for key, items in truth.items()}
    matches = []
    for _, key, start, end in candidates:
        intervals = truth[key]
        center = (start + end) / 2
        inside = [i for i, (onset, offset) in enumerate(intervals) if onset <= center < offset]
        available = [i for i in inside if not matched[key][i]]
        if available:
            hit = available[0]
            matched[key][hit] = True
            status, truth_index = "tp", hit
        elif inside:
            status, truth_index = "duplicate", inside[0]
        else:
            status, truth_index = "fp", None
        truth_start, truth_end = intervals[truth_index] if truth_index is not None else (None, None)
        matches.append({
            "status": status,
            "truth_start_ms": truth_start,
            "truth_end_ms": truth_end,
        })
    return candidates, matches


def target_units(sequences, units, label):
    by_stem = {}
    for sequence in sequences:
        by_stem.setdefault(sequence.stem, []).append(sequence)
    truth = {sequence.key: [] for sequence in sequences}
    for stem, items in units.items():
        for onset, offset, item_label in items:
            if item_label != label:
                continue
            middle = (onset + offset) / 2
            sequence = next(
                (item for item in by_stem.get(stem, []) if item.starts[0] <= middle < item.ends[-1]),
                None,
            )
            if sequence:
                truth[sequence.key].append((max(sequence.starts[0], onset), min(sequence.ends[-1], offset)))
    return truth


def query_identifier(query):
    return f"{query.label}:{query.sequence.stem}:{query.onset}:{query.offset}"


def evaluate_query(query, sequences, units, top_k):
    template = query.sequence.features[query.start : query.end]
    targets = [sequence for sequence in sequences if sequence.stem != query.sequence.stem]
    truth = target_units(targets, units, query.label)
    query_id = query_identifier(query)
    query_row = {
        "query_id": query_id,
        "query_sequence": query.sequence.key,
        "label": query.label,
        "query_recording": query.sequence.stem,
        "query_onset_ms": query.onset,
        "query_offset_ms": query.offset,
        "query_start_ms": int(query.sequence.starts[query.start]),
        "query_end_ms": int(query.sequence.ends[query.end - 1]),
        "query_tokens": query.end - query.start,
    }
    by_stem = {}
    for sequence in targets:
        by_stem.setdefault(sequence.stem, []).append(sequence)

    song_rows, detections = [], []
    for stem, song_sequences in by_stem.items():
        song_truth = {sequence.key: truth[sequence.key] for sequence in song_sequences}
        positives = sum(len(items) for items in song_truth.values())
        if not positives:
            continue

        candidates = []
        for sequence in song_sequences:
            scores = trajectory_scores(sequence.features, template)
            length = len(template)
            starts, ends = sequence.starts[: len(scores)], sequence.ends[length - 1 :]
            for index in nms(scores, length):
                candidates.append((float(scores[index]), sequence.key, int(starts[index]), int(ends[index])))
        ranked, matches = ranked_matches(candidates, song_truth)
        event_ap, r_precision = event_metrics(matches, positives)
        song_rows.append({
            "query_id": query_id,
            "label": query.label,
            "recording": stem,
            "start_ms": int(min(sequence.starts[0] for sequence in song_sequences)),
            "end_ms": int(max(sequence.ends[-1] for sequence in song_sequences)),
            "event_ap": event_ap,
            "r_precision": r_precision,
            "target_events": positives,
            "candidates": len(candidates),
        })
        detections.extend({
            "query_id": query_id,
            "rank": rank,
            "score": score,
            "sequence": key,
            "recording": stem,
            "start_ms": start,
            "end_ms": end,
            **match,
        } for rank, ((score, key, start, end), match) in enumerate(zip(ranked[:top_k], matches[:top_k]), 1))
    return (query_row, song_rows, detections) if song_rows else None


def summarize(queries, songs, args, sequences, metadata):
    class_metrics = []
    for label in sorted({row["label"] for row in queries}):
        items = [row for row in queries if row["label"] == label]
        query_metrics = [
            {
                key: mean(song[key] for song in songs if song["query_id"] == row["query_id"])
                for key in ("event_ap", "r_precision")
            }
            for row in items
        ]
        class_metrics.append({
            "label": label,
            "queries": len(items),
            "positive_song_pairs": sum(song["label"] == label for song in songs),
            **{key: mean(row[key] for row in query_metrics) for key in ("event_ap", "r_precision")},
        })
    query_ids = sorted(
        (row["label"], row["query_recording"], row["query_onset_ms"], row["query_offset_ms"])
        for row in queries
    )
    gallery = sorted((sequence.key, sequence.stem, int(sequence.starts[0]), int(sequence.ends[-1])) for sequence in sequences)
    return {
        "model": args.model,
        "species": args.species,
        "bird": args.bird,
        "recordings": len({sequence.stem for sequence in sequences}),
        "sequences": len(sequences),
        "classes": len(class_metrics),
        "queries": len(queries),
        "positive_song_pairs": len(songs),
        "ranking": "per_song_peaks",
        "candidates": "trajectory_peaks",
        "similarity": "mean_aligned_cosine",
        "deduplication": "greedy_query_width",
        "matching": "peak_center_in_event",
        "query_hash": hashlib.sha256(json.dumps(query_ids).encode()).hexdigest(),
        "gallery_hash": hashlib.sha256(json.dumps(gallery).encode()).hexdigest(),
        "event_map": mean(row["event_ap"] for row in class_metrics),
        "r_precision": mean(row["r_precision"] for row in class_metrics),
        "queries_per_class": args.queries_per_class,
        "top_k": args.top_k,
        "seed": args.seed,
        "feature_key": args.feature_key,
        "embeddings": str(Path(args.embeddings).resolve()),
        "annotations": str(Path(args.annotations).resolve()),
        "embedding_metadata": metadata,
        "class_metrics": class_metrics,
    }


def write_results(out_dir, queries, songs, detections, summary):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("queries", queries), ("songs", songs), ("detections", detections)):
        with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Slide annotated query trajectories across each song.")
    for name in ("embeddings", "annotations", "out_dir", "model", "species", "bird"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--feature_key", default="encoded_embeddings")
    parser.add_argument("--queries_per_class", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=10, help="Peaks saved per query-song; metrics use all peaks.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert args.queries_per_class > 0
    assert args.top_k > 0

    sequences, metadata = load_sequences(args.embeddings, args.feature_key)
    units = load_units(args.annotations)
    queries = select_queries(sequences, units, args.queries_per_class, args.seed)
    evaluated = [
        result
        for query in queries
        if (result := evaluate_query(query, sequences, units, args.top_k))
    ]
    queries = [row for row, _, _ in evaluated]
    songs = [item for _, items, _ in evaluated for item in items]
    detections = [item for _, _, items in evaluated for item in items]
    assert queries and songs, "no per-song queries could be evaluated"
    summary = summarize(queries, songs, args, sequences, metadata)
    write_results(args.out_dir, queries, songs, detections, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
