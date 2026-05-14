import argparse
import json
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from individual_id.latent_trajectory_decoder.train_t2vec_gru_latents import T2VecLatentGRU
from individual_id.run_individual_id_umap import (
    _fit_umap,
    _hdbscan_umap_analysis,
    _pool_embeddings,
    _pool_labels,
    _scatter_umap,
    _scatter_umap_syllables,
    _stable_seed,
    _umap_silhouette_scores,
    _window_starts_for_length,
)


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
        labels = np.unique(birds[indices])
        assert len(labels) == 1, recording
        rows.append((recording, labels[0], features[indices], syllables[indices]))
    return rows, features.shape[1]


def load_model(path, input_dim, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    model = T2VecLatentGRU(
        input_dim,
        int(args["hidden_dim"]),
        int(args["layers"]),
        bool(args["bidirectional"]),
        args.get("cell", "gru"),
        bool(args.get("vae", False)),
        int(args.get("latent_dim", 0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, args


@torch.no_grad()
def model_states(model, seq, device):
    clean = torch.from_numpy(seq).to(device=device, dtype=torch.float32)[None]
    lengths = torch.tensor([seq.shape[0]], device=device, dtype=torch.long)
    _, h, state, _ = model(clean, lengths, clean)
    return h.squeeze(0).cpu().numpy().astype(np.float32), state.squeeze(0).cpu().numpy().astype(np.float32)


def pooled_decoder_states(rows, model, args, device):
    x_parts = []
    bird_parts = []
    syllable_parts = []
    recording_parts = []
    state_parts = []
    state_birds = []
    state_recordings = []
    for index, (recording, bird, seq, syllables) in enumerate(rows):
        h, state = model_states(model, seq, device)
        seed = _stable_seed(args.seed, bird, index)
        starts, short_segment = _window_starts_for_length(h.shape[0], args.pool_window, args.pool_hop, "sliding", seed)
        pooled = _pool_embeddings(h, args.pool_window, args.pool_mode, args.pool_hop, starts=starts, short_segment=short_segment)
        pooled_labels = _pool_labels(syllables, args.pool_window, args.pool_hop, starts=starts, short_segment=short_segment)
        count = min(pooled.shape[0], pooled_labels.shape[0])
        x_parts.append(pooled[:count])
        syllable_parts.append(pooled_labels[:count])
        bird_parts.extend([bird] * count)
        recording_parts.extend([recording] * count)
        state_parts.append(state)
        state_birds.append(bird)
        state_recordings.append(recording)
    return (
        np.vstack(x_parts).astype(np.float32, copy=False),
        np.asarray(bird_parts, dtype=object),
        np.concatenate(syllable_parts).astype(np.int64, copy=False),
        np.asarray(recording_parts, dtype=object),
        np.vstack(state_parts).astype(np.float32, copy=False),
        np.asarray(state_birds, dtype=object),
        np.asarray(state_recordings, dtype=object),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="UMAP t2vec-style GRU latent states.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--model_pt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--species_display_name", required=True)
    parser.add_argument("--pool_window", type=int, default=20)
    parser.add_argument("--pool_hop", type=int, default=5)
    parser.add_argument("--pool_mode", choices=["mean", "stats"], default="stats")
    parser.add_argument("--umap_neighbors", type=int, default=50)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_metric", default="cosine")
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
    features, bird_labels, syllable_labels, recording_labels, states, state_birds, state_recordings = pooled_decoder_states(rows, model, args, device)

    rep_name = f"t2vec_gru_decoder_state_pool_{args.pool_mode}_sliding_w{args.pool_window}_h{args.pool_hop}_raw"
    print(f"[umap] {rep_name}: points={features.shape[0]} dim={features.shape[1]}")
    xy = _fit_umap(
        features,
        neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
        negative_sample_rate=args.umap_negative_sample_rate,
    )
    plot_args = Namespace(
        species_display_name=args.species_display_name,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
    )
    silhouette_scores = _umap_silhouette_scores(xy, bird_labels, syllable_labels, args.silhouette_sample_size, args.seed)
    hdbscan_summary = _hdbscan_umap_analysis(xy, bird_labels, syllable_labels, recording_labels, out_dir, rep_name, plot_args)
    _scatter_umap(xy, bird_labels, args.species_display_name, out_dir / rep_name)
    _scatter_umap_syllables(xy, syllable_labels, bird_labels, f"{args.species_display_name}: syllables", out_dir / f"{rep_name}_syllable")

    state_xy = _fit_umap(
        states,
        neighbors=min(15, max(2, states.shape[0] - 1)),
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
        negative_sample_rate=args.umap_negative_sample_rate,
    )
    _scatter_umap(state_xy, state_birds, f"{args.species_display_name}: t2vec encoder states", out_dir / "t2vec_encoder_recording_state_raw")

    np.savez_compressed(
        out_dir / f"{rep_name}_features.npz",
        features=features,
        xy=xy.astype(np.float32, copy=False),
        bird_labels=bird_labels,
        syllable_labels=syllable_labels,
        recording_labels=recording_labels,
        encoder_states=states,
        encoder_state_xy=state_xy.astype(np.float32, copy=False),
        encoder_state_birds=state_birds,
        encoder_state_recordings=state_recordings,
    )
    summary = {
        "features_npz": args.features_npz,
        "model_pt": args.model_pt,
        "device": str(device),
        "model_args": model_args,
        "representation": rep_name,
        "recordings": len(rows),
        "points": int(features.shape[0]),
        "raw_feature_dim": int(features.shape[1]),
        "encoder_state_dim": int(states.shape[1]),
        "silhouette_scores": silhouette_scores,
        "hdbscan_summary": hdbscan_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
