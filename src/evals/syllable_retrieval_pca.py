#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.embedding_store import EmbeddingStore
from src.core.utils import load_spec, resolve_single_spec_path
from src.evals.syllable_classification import load_units
from src.evals.syllable_retrieval import (
    Sequence,
    event_metrics,
    nms,
    query_identifier,
    ranked_matches,
    select_queries,
    split_sequences,
    summarize,
    target_units,
    write_results,
)


MODEL = "spectrogram_pca_euclidean"
PAPER_WINDOW_MS = 68
PCA_COMPONENTS = 100
MAX_PCA_WINDOWS = 10_000


def windows(spec, width):
    view = np.lib.stride_tricks.sliding_window_view(spec, width, axis=1)
    return view.transpose(1, 0, 2).reshape(view.shape[1], -1)


def reference_sequences(path):
    data = EmbeddingStore(path)
    count = len(data["token_start_ms"])
    return split_sequences(data, np.empty((count, 0), dtype=np.float32))


def coverage(sequence, spec, ms_per_bin):
    start = max(0, int(np.floor(sequence.starts[0] / ms_per_bin)))
    end = min(spec.shape[1], int(np.ceil(sequence.ends[-1] / ms_per_bin)))
    return spec[:, start:end], start


def fit_pca(sequences, specs, width, ms_per_bin, seed):
    rows = np.concatenate([
        windows(coverage(sequence, specs[sequence.stem], ms_per_bin)[0], width)[::width]
        for sequence in sequences
    ])
    if len(rows) > MAX_PCA_WINDOWS:
        rows = rows[np.linspace(0, len(rows) - 1, MAX_PCA_WINDOWS, dtype=int)]
    components = min(PCA_COMPONENTS, min(rows.shape) - 1)
    model = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    model.fit(rows)
    return model, len(rows)


def project_sequences(reference, specs, model, width, ms_per_bin):
    sequences = []
    for item in reference:
        spec, start = coverage(item, specs[item.stem], ms_per_bin)
        features = model.transform(windows(spec, width)).astype(np.float32, copy=False)
        index = np.arange(len(features)) + start
        starts = np.rint(index * ms_per_bin).astype(np.int64)
        ends = np.rint((index + width) * ms_per_bin).astype(np.int64)
        sequences.append(Sequence(item.key, item.stem, features, starts, ends))
    return sequences


def evaluate_query(query, sequences, units, top_k, width):
    source = [sequence for sequence in sequences if sequence.stem == query.sequence.stem]
    assert len(source) == 1
    source = source[0]
    query_center = (query.onset + query.offset) / 2
    query_index = int(np.argmin(abs((source.starts + source.ends) / 2 - query_center)))
    query_vector = source.features[query_index]
    targets = [sequence for sequence in sequences if sequence.stem != source.stem]
    truth = target_units(targets, units, query.label)
    query_id = query_identifier(query)
    query_row = {
        "query_id": query_id,
        "query_sequence": source.key,
        "label": query.label,
        "query_recording": source.stem,
        "query_onset_ms": query.onset,
        "query_offset_ms": query.offset,
        "query_start_ms": int(source.starts[query_index]),
        "query_end_ms": int(source.ends[query_index]),
        "query_tokens": width,
    }

    songs, detections = [], []
    for sequence in targets:
        positives = len(truth[sequence.key])
        if not positives:
            continue
        scores = -np.linalg.norm(sequence.features - query_vector, axis=1)
        candidates = [
            (float(scores[index]), sequence.key, int(sequence.starts[index]), int(sequence.ends[index]))
            for index in nms(scores, width)
        ]
        ranked, matches = ranked_matches(candidates, {sequence.key: truth[sequence.key]})
        event_ap, r_precision = event_metrics(matches, positives)
        songs.append({
            "query_id": query_id,
            "label": query.label,
            "recording": sequence.stem,
            "start_ms": int(sequence.starts[0]),
            "end_ms": int(sequence.ends[-1]),
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
            "recording": sequence.stem,
            "start_ms": start,
            "end_ms": end,
            **match,
        } for rank, ((score, key, start, end), match) in enumerate(zip(ranked[:top_k], matches[:top_k]), 1))
    return query_row, songs, detections


def main():
    parser = argparse.ArgumentParser(description="Kollmorgen-style spectrogram PCA retrieval baseline.")
    for name in ("embeddings", "spec_dir", "annotations", "out_dir", "species", "bird"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--queries_per_class", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert args.queries_per_class > 0 and args.top_k > 0
    args.model = MODEL
    args.feature_key = "spectrogram_pca"

    reference = reference_sequences(args.embeddings)
    assert len(reference) == len({sequence.stem for sequence in reference}), "baseline requires full recordings"
    units = load_units(args.annotations)
    queries = select_queries(reference, units, args.queries_per_class, args.seed)
    spec_dir = Path(args.spec_dir)
    params = json.loads((spec_dir / "audio_params.json").read_text())
    ms_per_bin = 1000 * params["hop_size"] / params["sr"]
    width = round(PAPER_WINDOW_MS / ms_per_bin)
    specs = {
        sequence.stem: load_spec(resolve_single_spec_path(spec_dir, sequence.stem))
        for sequence in reference
    }
    model, fit_windows = fit_pca(reference, specs, width, ms_per_bin, args.seed)
    sequences = project_sequences(reference, specs, model, width, ms_per_bin)
    evaluated = [evaluate_query(query, sequences, units, args.top_k, width) for query in queries]
    query_rows = [row for row, _, _ in evaluated]
    songs = [row for _, rows, _ in evaluated for row in rows]
    detections = [row for _, _, rows in evaluated for row in rows]
    assert query_rows and songs

    metadata = {
        "citation": "Kollmorgen et al., Nature 2020",
        "doi": "10.1038/s41586-019-1892-x",
        "paper_window_ms": PAPER_WINDOW_MS,
        "window_ms": width * ms_per_bin,
        "window_bins": width,
        "alignment": "event_center",
        "pca_components": model.n_components_,
        "pca_fit_windows": fit_windows,
        "pca_fit": "nonoverlapping_windows_same_bird",
        "pca_whiten": False,
        "pca_explained_variance": float(model.explained_variance_ratio_.sum()),
        "spectrogram_frontend": params,
    }
    summary = summarize(query_rows, songs, args, reference, metadata)
    summary["reference_embeddings"] = summary.pop("embeddings")
    summary.update({
        "candidates": "spectrogram_pca_peaks",
        "similarity": "negative_euclidean",
        "representation": "fixed_log_mel_spectrogram_pca",
        "spectrograms": str(spec_dir.resolve()),
    })
    write_results(args.out_dir, query_rows, songs, detections, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
