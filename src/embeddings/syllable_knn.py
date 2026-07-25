#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == HERE:
    sys.path.pop(0)
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.embedding_store import EmbeddingStore  # noqa: E402
from src.core.model import TARGET_FEATURE_TYPES  # noqa: E402
from src.embeddings.extract import extract  # noqa: E402


CONTEXT_TIMEBINS = 1000
SILENCE = 0


def ints(text):
    return sorted({int(x) for x in text.split(",") if x.strip()})


def prepare(reference, query, components, seed):
    mean = reference.mean(axis=0, keepdims=True)
    std = np.maximum(reference.std(axis=0, keepdims=True), 1e-6)
    reference = (reference - mean) / std
    query = (query - mean) / std
    if components:
        from sklearn.decomposition import PCA

        assert components <= min(reference.shape), reference.shape
        pca = PCA(components, svd_solver="randomized", random_state=seed, whiten=False)
        reference = pca.fit_transform(reference)
        query = pca.transform(query)
    return unit(reference), unit(query)


def unit(features):
    features = np.ascontiguousarray(features, dtype=np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    return features


def recording_for_stem(recordings, stem):
    if stem in recordings:
        return recordings[stem]
    matches = [value for key, value in recordings.items() if stem.startswith(f"{key}_")]
    assert len(matches) == 1, stem
    return matches[0]


def annotation_recordings(path, bird):
    data = json.loads(Path(path).read_text())
    return {
        Path(row["recording"]["filename"]).stem: row
        for row in data["recordings"]
        if row["recording"].get("bird_id") == bird
    }


def add_occurrence(out, label, event, onset, offset, token_indices, starts, ends):
    tokens = token_indices[(ends[token_indices] > onset) & (starts[token_indices] < offset)]
    if tokens.size:
        out.append({"label": label, "event": event, "tokens": tokens})


def occurrences(store, annotation_file, bird):
    starts = np.asarray(store["token_start_ms"])
    ends = np.asarray(store["token_end_ms"])
    stems = np.asarray(store["recording_stem"]).astype(str)
    events = np.asarray(store["song_id"]).astype(np.int64)
    recordings = annotation_recordings(annotation_file, bird)
    out = []
    for event in np.unique(events):
        token_indices = np.flatnonzero(events == event)
        start, end = float(starts[token_indices].min()), float(ends[token_indices].max())
        recording = recording_for_stem(recordings, stems[token_indices[0]])
        units = [
            (max(start, float(unit["onset_ms"])), min(end, float(unit["offset_ms"])), int(unit["id"]) + 1)
            for detected in recording.get("detected_events", [])
            for unit in detected.get("units", [])
            if unit["offset_ms"] > start and unit["onset_ms"] < end
        ]
        units.sort()
        cursor = start
        for onset, offset, label in units:
            if cursor < onset:
                add_occurrence(out, SILENCE, int(event), cursor, onset, token_indices, starts, ends)
            add_occurrence(out, label, int(event), onset, offset, token_indices, starts, ends)
            cursor = max(cursor, offset)
        if cursor < end:
            add_occurrence(out, SILENCE, int(event), cursor, end, token_indices, starts, ends)
    assert out
    return out


def sample_occurrences(rows, per_class, seed):
    rng = np.random.default_rng(seed)
    selected = []
    labels = sorted({row["label"] for row in rows})
    for label in labels:
        indices = np.flatnonzero([row["label"] == label for row in rows])
        count = indices.size if per_class <= 0 else min(indices.size, per_class)
        selected.extend(rng.choice(indices, size=count, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def memberships(rows, selected):
    tokens, occurrence_ids = [], []
    for occurrence_id in selected:
        indices = rows[occurrence_id]["tokens"]
        tokens.append(indices)
        occurrence_ids.append(np.full(indices.size, occurrence_id, dtype=np.int64))
    return np.concatenate(tokens), np.concatenate(occurrence_ids)


def topk(query, reference, k, chunk_size, cpu):
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    reference = torch.from_numpy(reference).to(device)
    out = np.empty((query.shape[0], k), dtype=np.int64)
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        similarities = torch.from_numpy(query[start:end]).to(device) @ reference.T
        out[start:end] = torch.topk(similarities, k=k, dim=1).indices.cpu().numpy()
    return out, str(device)


def occurrence_neighbors(candidates, query_occurrences, reference_occurrences, rows, k):
    out = np.empty((candidates.shape[0], k), dtype=np.int64)
    for i, candidates_i in enumerate(candidates):
        query_event = rows[query_occurrences[i]]["event"]
        seen, neighbors = set(), []
        for candidate in candidates_i:
            occurrence = int(reference_occurrences[candidate])
            if rows[occurrence]["event"] == query_event or occurrence in seen:
                continue
            seen.add(occurrence)
            neighbors.append(occurrence)
            if len(neighbors) == k:
                break
        assert len(neighbors) == k
        out[i] = neighbors
    return out


def add_args(parser):
    for name in "spec_dir annotation_file out_dir bird".split():
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--model", required=True, choices=["songmae", "songmae_random", "aves", "hubert"])
    for name in "name wav_dir recording_stem songmae_run_dir checkpoint embedding_dir".split():
        parser.add_argument(f"--{name}")
    for name, default in [
        ("wav_exts", ".wav,.flac,.ogg,.mp3"),
        ("k_values", "1,5,10"),
        ("aves_model_path", str(ROOT / "files" / "birdaves-biox-base.torchaudio.pt")),
        ("aves_config_path", str(ROOT / "files" / "birdaves-biox-base.torchaudio.model_config.json")),
        ("hubert_model_name", "facebook/hubert-base-ls960"),
    ]:
        parser.add_argument(f"--{name}", default=default)
    parser.add_argument("--target_feature_type", default="end_of_block", choices=TARGET_FEATURE_TYPES)
    for name, default in [
        ("num_timebins", 0), ("reference_occurrences_per_class", 100),
        ("query_occurrences_per_class", 20), ("search_k", 1000),
        ("seed", 42), ("knn_chunk_size", 512), ("encoder_layer_idx", -1),
        ("pca_components", 0),
    ]:
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--cpu", action="store_true")


def validate_protocol(store, args):
    metadata = store.metadata
    assert metadata["encoder_layer_idx"] == args.encoder_layer_idx
    assert not metadata.get("all_layers", False)
    if args.model.startswith("songmae"):
        assert metadata["target_feature_type"] == args.target_feature_type
        assert metadata["model_num_timebins"] == CONTEXT_TIMEBINS
        return
    assert metadata["chunk_timebins"] == CONTEXT_TIMEBINS
    assert metadata["feature_center_timebins"] == 2.5
    assert metadata["feature_stride_timebins"] == 4.0
    expected = "birdaves_biox_base" if args.model == "aves" else args.hubert_model_name
    assert metadata["model_name"] == expected


def main():
    parser = argparse.ArgumentParser(description="Cross-event occurrence-level trajectory kNN purity.")
    add_args(parser)
    args = parser.parse_args()
    assert args.pca_components >= 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.recording_mode = "events"
    args.minimal = True
    args.max_points = 0
    args.reuse = False
    args.chunk_timebins = CONTEXT_TIMEBINS
    source = Path(args.embedding_dir) if args.embedding_dir else extract(args.model, args, out_dir)
    store = EmbeddingStore(source)
    validate_protocol(store, args)

    rows = occurrences(store, args.annotation_file, args.bird)
    reference_ids = sample_occurrences(rows, args.reference_occurrences_per_class, args.seed + 1)
    query_ids = sample_occurrences(rows, args.query_occurrences_per_class, args.seed)
    reference_tokens, reference_occurrences = memberships(rows, reference_ids)
    query_tokens, query_occurrences = memberships(rows, query_ids)

    features = np.asarray(store["encoded_embeddings"], dtype=np.float32)
    reference, query = prepare(
        features[reference_tokens],
        features[query_tokens],
        args.pca_components,
        args.seed,
    )
    max_k = max(ints(args.k_values))
    search_k = min(reference.shape[0], max(args.search_k, max_k))
    while True:
        candidates, device = topk(query, reference, search_k, args.knn_chunk_size, args.cpu)
        try:
            neighbors = occurrence_neighbors(
                candidates, query_occurrences, reference_occurrences, rows, max_k
            )
            break
        except AssertionError:
            assert search_k < reference.shape[0], "not enough cross-event reference occurrences"
            search_k = min(reference.shape[0], search_k * 2)

    query_labels = np.asarray([rows[index]["label"] for index in query_occurrences])
    neighbor_labels = np.asarray([[rows[index]["label"] for index in row] for row in neighbors])
    output_rows = []
    for k in ints(args.k_values):
        token_purity = (neighbor_labels[:, :k] == query_labels[:, None]).mean(axis=1)
        occurrence_purity = {
            int(index): float(token_purity[query_occurrences == index].mean())
            for index in query_ids
        }
        per_class = {
            int(label): float(np.mean([
                occurrence_purity[index] for index in query_ids if rows[index]["label"] == label
            ]))
            for label in sorted(set(query_labels.tolist()))
        }
        macro = float(np.mean(list(per_class.values())))
        vocal = float(np.mean([value for label, value in per_class.items() if label != SILENCE]))
        output_rows.append({
            "k": k,
            "macro_same_purity": macro,
            "macro_different_purity": 1.0 - macro,
            "vocal_macro_same_purity": vocal,
            "vocal_macro_different_purity": 1.0 - vocal,
            "silence_same_purity": per_class[SILENCE],
            "query_occurrences": int(query_ids.size),
            "reference_occurrences": int(reference_ids.size),
            "query_tokens": int(query_tokens.size),
            "reference_tokens": int(reference_tokens.size),
            "classes": len(per_class),
            "events": len({row["event"] for row in rows}),
            "per_class_same_purity": per_class,
        })

    with (out_dir / "knn_purity.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = vars(args) | {
        "analysis_unit": "annotated_occurrence_trajectory",
        "neighbor_unit": "reference_occurrence",
        "token_membership": "any_temporal_overlap",
        "silence_label": SILENCE,
        "standardization": "reference_tokens_feature_zscore",
        "standardization_fit_scope": "reference_tokens",
        "pca_fit_scope": "reference_tokens" if args.pca_components else "disabled",
        "pca_whiten": False,
        "row_normalization": "l2",
        "distance": "cosine",
        "device": device,
        "search_k_used": int(search_k),
        "rows": output_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
