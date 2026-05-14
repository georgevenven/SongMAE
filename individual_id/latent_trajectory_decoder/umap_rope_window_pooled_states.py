import argparse
import json
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from individual_id.latent_trajectory_decoder.train_recurrent_sequence import RecurrentSequenceDecoder
from individual_id.latent_trajectory_decoder.train_rope_sequence import RopeSequenceDecoder, SinusoidalSequenceDecoder
from individual_id.run_individual_id_umap import (
    _fit_umap,
    _hdbscan_umap_analysis,
    _load_recording_features,
    _pool_embeddings,
    _pool_labels,
    _scatter_umap,
    _scatter_umap_syllables,
    _stable_seed,
    _umap_silhouette_scores,
    _window_starts_for_length,
)
from src import extract_embedding


def load_rows(path):
    data = np.load(path, allow_pickle=True)
    features = data["features"].astype(np.float32)
    birds = data["bird_labels"].astype(str)
    syllables = data["syllable_labels"].astype(np.int64)
    recordings = data["recording_labels"].astype(str)
    by_recording = defaultdict(list)
    for i, recording in enumerate(recordings):
        by_recording[recording].append(i)

    rows = []
    for recording in sorted(by_recording):
        indices = by_recording[recording]
        bird_labels = np.unique(birds[indices])
        assert len(bird_labels) == 1, recording
        if len(indices) > 2:
            rows.append((recording, bird_labels[0], features[indices], syllables[indices]))
    return rows, features.shape[1]


def load_model(path, input_dim, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_args = checkpoint["args"]
    metrics = checkpoint.get("metrics", {})
    if metrics.get("model_type") == "recurrent":
        model = RecurrentSequenceDecoder(
            input_dim,
            int(model_args["hidden_dim"]),
            int(model_args["layers"]),
            str(model_args["cell"]),
        ).to(device)
        model_args = {"d_model": int(model_args["hidden_dim"]), "model_type": str(model_args["cell"])}
    else:
        model_class = RopeSequenceDecoder
        if model_args.get("position_encoding") == "sinusoidal":
            model_class = SinusoidalSequenceDecoder
        model = model_class(
            input_dim,
            int(model_args["d_model"]),
            int(model_args["heads"]),
            int(model_args["layers"]),
        ).to(device)
        model_args["position_encoding"] = model_args.get("position_encoding", "rope")
    model.load_state_dict(checkpoint["decoder"])
    model.eval()
    return model, model_args


@torch.no_grad()
def decoder_states(model, seq, device, max_seq_len, target_mode):
    offset = 0 if target_mode == "self" else 1
    if max_seq_len <= 0:
        x = torch.from_numpy(seq[: seq.shape[0] - offset]).to(device=device, dtype=torch.float32)[None]
        valid = torch.ones((1, x.shape[1]), device=device, dtype=torch.bool)
        _, h = model(x, valid)
        return h.squeeze(0).cpu().numpy().astype(np.float32)

    parts = []
    for start in range(0, seq.shape[0] - offset, max_seq_len):
        end = min(seq.shape[0], start + max_seq_len + offset)
        x = torch.from_numpy(seq[start : end - offset]).to(device=device, dtype=torch.float32)[None]
        valid = torch.ones((1, x.shape[1]), device=device, dtype=torch.bool)
        _, h = model(x, valid)
        parts.append(h.squeeze(0).cpu().numpy().astype(np.float32))
    return np.vstack(parts).astype(np.float32, copy=False)


def pooled_decoder_windows(rows, model, args, device, recording_features, target_mode):
    x_parts = []
    bird_parts = []
    syllable_parts = []
    recording_parts = []
    for index, (recording, bird, seq, syllables) in enumerate(rows):
        h = decoder_states(model, seq, device, args.max_seq_len, target_mode)
        labels = syllables[: h.shape[0]]
        seed = _stable_seed(args.seed, bird, index)
        starts, short_segment = _window_starts_for_length(h.shape[0], args.pool_window, args.pool_hop, "sliding", seed)
        pooled = _pool_embeddings(
            h,
            args.pool_window,
            args.pool_mode,
            args.pool_hop,
            starts=starts,
            short_segment=short_segment,
        )
        pooled_labels = _pool_labels(labels, args.pool_window, args.pool_hop, starts=starts, short_segment=short_segment)
        count = min(pooled.shape[0], pooled_labels.shape[0])
        if recording_features is not None:
            assert recording in recording_features, recording
            rec_features = np.repeat(recording_features[recording][None], count, axis=0)
            pooled = np.hstack([pooled[:count], rec_features]).astype(np.float32, copy=False)
        x_parts.append(pooled[:count])
        syllable_parts.append(pooled_labels[:count])
        bird_parts.extend([bird] * count)
        recording_parts.extend([recording] * count)

    return (
        np.vstack(x_parts).astype(np.float32, copy=False),
        np.asarray(bird_parts, dtype=object),
        np.concatenate(syllable_parts).astype(np.int64, copy=False),
        np.asarray(recording_parts, dtype=object),
    )


def sample_points(features, birds, syllables, recordings, max_points, seed):
    if max_points <= 0 or features.shape[0] <= max_points:
        return features, birds, syllables, recordings
    rng = np.random.default_rng(seed)
    keep = rng.choice(features.shape[0], size=max_points, replace=False)
    keep.sort()
    return features[keep], birds[keep], syllables[keep], recordings[keep]


def parse_args():
    parser = argparse.ArgumentParser(description="UMAP window-pooled RoPE decoder hidden states.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--model_pt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--species_display_name", required=True)
    parser.add_argument("--pool_window", type=int, default=20)
    parser.add_argument("--pool_hop", type=int, default=5)
    parser.add_argument("--pool_mode", choices=["mean", "stats"], default="mean")
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=0)
    parser.add_argument("--feature_postprocess", choices=["none", "pca_whiten_l2"], default="pca_whiten_l2")
    parser.add_argument("--feature_postprocess_dim", type=int, default=512)
    parser.add_argument("--recording_svd_npz", default=None)
    parser.add_argument("--recording_feature_mode", default="svd", choices=["svd", "svd_u", "svd_us", "normalized_svd", "norm_adj_eig", "norm_adj_eig_skip1", "norm_adj_eig_kmeans", "affinity_row"])
    parser.add_argument("--recording_svd_dim", type=int, default=20)
    parser.add_argument("--recording_svd_alpha", type=float, default=1.0)
    parser.add_argument("--umap_neighbors", type=int, default=50)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_metric", default="cosine")
    parser.add_argument("--umap_random_state", type=int, default=None)
    parser.add_argument("--umap_negative_sample_rate", type=int, default=5)
    parser.add_argument("--silhouette_sample_size", type=int, default=10000)
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=0)
    parser.add_argument("--hdbscan_min_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, input_dim = load_rows(args.features_npz)
    model, model_args = load_model(args.model_pt, input_dim, device)
    recording_features = _load_recording_features(
        args.recording_svd_npz,
        args.recording_feature_mode,
        args.recording_svd_dim,
        args.recording_svd_alpha,
        include_stems=[row[0] for row in rows],
    )
    target_mode = model_args.get("target_mode", "next")
    raw_features, bird_labels, syllable_labels, recording_labels = pooled_decoder_windows(
        rows,
        model,
        args,
        device,
        recording_features,
        target_mode,
    )
    raw_points = int(raw_features.shape[0])
    raw_features, bird_labels, syllable_labels, recording_labels = sample_points(
        raw_features,
        bird_labels,
        syllable_labels,
        recording_labels,
        args.max_points,
        args.seed,
    )
    features, transform = extract_embedding.maybe_apply_feature_postprocess(
        raw_features,
        mode=args.feature_postprocess,
        dim=args.feature_postprocess_dim,
    )
    suffix = "raw" if transform is None else f"pca_whiten_l2{transform['dim']}"
    rep_name = f"{model_args.get('position_encoding', 'rope')}_decoder_state_pool_{args.pool_mode}_sliding_w{args.pool_window}_h{args.pool_hop}_{suffix}"
    if recording_features is not None:
        dim_suffix = args.recording_svd_dim if args.recording_feature_mode != "affinity_row" else "full"
        rep_name = f"{rep_name}_rec{args.recording_feature_mode}{dim_suffix}_a{args.recording_svd_alpha:g}"
    print(f"[umap] {rep_name}: points={features.shape[0]} dim={features.shape[1]}")

    xy = _fit_umap(
        features,
        neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.umap_random_state,
        negative_sample_rate=args.umap_negative_sample_rate,
    )
    plot_args = Namespace(
        species_display_name=args.species_display_name,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
    )
    silhouette_scores = _umap_silhouette_scores(xy, bird_labels, syllable_labels, args.silhouette_sample_size, args.seed)
    hdbscan_summary = _hdbscan_umap_analysis(
        xy,
        bird_labels,
        syllable_labels,
        recording_labels,
        out_dir,
        rep_name,
        plot_args,
    )
    _scatter_umap(xy, bird_labels, args.species_display_name, out_dir / rep_name)
    _scatter_umap_syllables(
        xy,
        syllable_labels,
        bird_labels,
        f"{args.species_display_name}: syllables",
        out_dir / f"{rep_name}_syllable",
    )
    np.savez_compressed(
        out_dir / f"{rep_name}_features.npz",
        features=features.astype(np.float32, copy=False),
        raw_decoder_features=raw_features.astype(np.float32, copy=False),
        xy=xy.astype(np.float32, copy=False),
        bird_labels=bird_labels,
        syllable_labels=syllable_labels,
        recording_labels=recording_labels,
    )
    summary = {
        "features_npz": args.features_npz,
        "model_pt": args.model_pt,
        "device": str(device),
        "input_dim": int(input_dim),
        "decoder_state_dim": int(model_args["d_model"]),
        "position_encoding": model_args.get("position_encoding", "rope"),
        "target_mode": target_mode,
        "recordings": len(rows),
        "representation": rep_name,
        "pool_mode": args.pool_mode,
        "max_seq_len": int(args.max_seq_len),
        "raw_points_before_sampling": raw_points,
        "points": int(features.shape[0]),
        "raw_feature_dim": int(raw_features.shape[1]),
        "feature_postprocess": args.feature_postprocess,
        "feature_postprocess_requested_dim": int(args.feature_postprocess_dim),
        "feature_postprocess_dim": 0 if transform is None else int(transform["dim"]),
        "recording_svd_npz": args.recording_svd_npz,
        "recording_feature_mode": args.recording_feature_mode,
        "recording_svd_dim": int(args.recording_svd_dim),
        "recording_svd_alpha": float(args.recording_svd_alpha),
        "args": vars(args),
        "silhouette_scores": silhouette_scores,
        "hdbscan_summary": hdbscan_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
