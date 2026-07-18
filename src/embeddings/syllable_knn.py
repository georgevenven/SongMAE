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


def ints(text):
    return sorted({int(x) for x in text.split(",") if x.strip()})


def stratified_queries(labels, max_queries, per_class, seed):
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    cap = per_class if max_queries <= 0 else max(1, max_queries // classes.size)
    picks = []
    for label in classes:
        idx = np.flatnonzero(labels == label)
        picks.append(rng.choice(idx, size=min(idx.size, per_class, cap), replace=False))
    return np.sort(np.concatenate(picks))


def stratified_reference(labels, max_points, min_per_class, seed):
    if max_points <= 0 or labels.size <= max_points:
        return np.arange(labels.size)
    rng = np.random.default_rng(seed + 1)
    classes = np.unique(labels)
    assert max_points >= classes.size
    floor = min(min_per_class, max(1, max_points // classes.size))
    keep, rest = [], []
    for label in classes:
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        keep.append(idx[: min(idx.size, floor)])
        rest.append(idx[min(idx.size, floor) :])
    keep = np.concatenate(keep)
    rest = np.concatenate([x for x in rest if x.size]) if any(x.size for x in rest) else np.array([], dtype=np.int64)
    if keep.size < max_points and rest.size:
        keep = np.concatenate([keep, rng.choice(rest, size=min(max_points - keep.size, rest.size), replace=False)])
    return np.sort(keep)


def unit(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x


def standardize(reference, query):
    mean = reference.mean(axis=0, keepdims=True)
    std = np.maximum(reference.std(axis=0, keepdims=True), 1e-6)
    return (reference - mean) / std, (query - mean) / std


def topk(query, ref, k, chunk_size, cpu):
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    ref = torch.from_numpy(ref).to(device)
    out = np.empty((query.shape[0], k), dtype=np.int64)
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        sims = torch.from_numpy(query[start:end]).to(device) @ ref.T
        out[start:end] = torch.topk(sims, k=k, dim=1).indices.cpu().numpy()
    return out, str(device)


def drop_same_event(candidates, query_events, ref_events, k):
    out = np.empty((candidates.shape[0], k), dtype=np.int64)
    for i, row in enumerate(candidates):
        row = row[ref_events[row] != query_events[i]]
        assert row.size >= k, "increase --search_k; not enough cross-event neighbors"
        out[i] = row[:k]
    return out


def add_args(parser):
    for name in "spec_dir annotation_file out_dir bird".split():
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--model", required=True, choices=["songmae", "aves", "hubert"])
    for name in "name wav_dir recording_stem songmae_run_dir checkpoint embedding_dir".split():
        parser.add_argument(f"--{name}")
    for name, default in [
        ("wav_exts", ".wav,.flac,.ogg,.mp3"),
        ("k_values", "1,5,10,20,50,100"),
        ("aves_model_path", str(ROOT / "files" / "birdaves-biox-base.torchaudio.pt")),
        ("aves_config_path", str(ROOT / "files" / "birdaves-biox-base.torchaudio.model_config.json")),
        ("hubert_model_name", "facebook/hubert-base-ls960"),
    ]:
        parser.add_argument(f"--{name}", default=default)
    parser.add_argument("--target_feature_type", default="end_of_block", choices=TARGET_FEATURE_TYPES)
    for name, default in [
        ("num_timebins", 0), ("max_ref_points", 200000),
        ("ref_min_per_class", 1000), ("max_queries", 5000), ("query_per_class", 200),
        ("search_k", 1000), ("seed", 42), ("knn_chunk_size", 512),
        ("encoder_layer_idx", -1), ("pca_components", 128),
    ]:
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--cpu", action="store_true")


def validate_protocol(store, args):
    metadata = store.metadata
    assert metadata["encoder_layer_idx"] == args.encoder_layer_idx
    assert not metadata.get("all_layers", False)
    if args.model == "songmae":
        assert metadata["target_feature_type"] == args.target_feature_type
        assert metadata["model_num_timebins"] == CONTEXT_TIMEBINS
        return
    assert metadata["chunk_timebins"] == CONTEXT_TIMEBINS
    assert metadata["feature_center_timebins"] == 2.5
    assert metadata["feature_stride_timebins"] == 4.0
    expected = "birdaves_biox_base" if args.model == "aves" else args.hubert_model_name
    assert metadata["model_name"] == expected


def main():
    parser = argparse.ArgumentParser(description="Cross-event syllable kNN purity.")
    add_args(parser)
    args = parser.parse_args()
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
    labels0 = store["labels_downsampled"].astype(np.int64, copy=False)
    events0 = store["song_id"].astype(np.int64, copy=False)
    good = np.flatnonzero(labels0 >= 0)
    labels, events = np.asarray(labels0[good]), np.asarray(events0[good])

    ref_idx = stratified_reference(labels, args.max_ref_points, args.ref_min_per_class, args.seed)
    query_idx = stratified_queries(labels, args.max_queries, args.query_per_class, args.seed)
    used = np.unique(np.concatenate([ref_idx, query_idx]))
    features = np.asarray(store["encoded_embeddings"][good[used]], dtype=np.float32)

    ref_pos, query_pos = np.searchsorted(used, ref_idx), np.searchsorted(used, query_idx)
    ref, query = standardize(features[ref_pos], features[query_pos])
    if args.pca_components > 0:
        from sklearn.decomposition import PCA

        assert args.pca_components <= min(ref.shape), ref.shape
        pca = PCA(args.pca_components, svd_solver="randomized", random_state=args.seed)
        ref, query = pca.fit_transform(ref), pca.transform(query)
    ref, query = unit(ref), unit(query)
    max_k, search_k = max(ints(args.k_values)), min(ref.shape[0], max(args.search_k, max(ints(args.k_values))))
    while True:
        candidates, device = topk(query, ref, search_k, args.knn_chunk_size, args.cpu)
        try:
            neighbors = drop_same_event(candidates, events[query_idx], events[ref_idx], max_k)
            break
        except AssertionError:
            assert search_k < ref.shape[0], "not enough cross-event references"
            search_k = min(ref.shape[0], search_k * 2)

    query_labels = labels[query_idx]
    same = labels[ref_idx][neighbors] == query_labels[:, None]
    rows = []
    for k in ints(args.k_values):
        vals = same[:, :k].mean(axis=1)
        macro = float(np.mean([vals[query_labels == label].mean() for label in np.unique(query_labels)]))
        rows.append({
            "k": k,
            "micro_same_purity": float(vals.mean()),
            "micro_different_purity": float(1.0 - vals.mean()),
            "macro_same_purity": macro,
            "macro_different_purity": float(1.0 - macro),
            "queries": int(query_idx.size),
            "references": int(ref_idx.size),
            "classes": int(np.unique(labels).size),
            "events": int(np.unique(events).size),
        })

    with (out_dir / "knn_purity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = vars(args) | {
        "standardization": "reference_feature_zscore",
        "pca_fit": "reference_only",
        "device": device,
        "search_k_used": int(search_k),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
