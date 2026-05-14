import argparse
import json
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from individual_id.latent_trajectory_decoder.train_t2vec_gru_latents import T2VecLatentGRU
from individual_id.run_individual_id_umap import _fit_umap, _hdbscan_umap_analysis, _scatter_umap, _umap_silhouette_scores


def load_rows(path):
    data = np.load(path, allow_pickle=True)
    features = data["features"].astype(np.float32)
    birds = data["bird_labels"].astype(str)
    recordings = data["recording_labels"].astype(str)
    syllables = data["syllable_labels"].astype(np.int64) if "syllable_labels" in data.files else np.zeros(len(features), dtype=np.int64)
    by_recording = defaultdict(list)
    for i, recording in enumerate(recordings):
        by_recording[recording].append(i)
    rows = []
    for recording in sorted(by_recording):
        indices = np.asarray(by_recording[recording], dtype=np.int64)
        labels = np.unique(birds[indices])
        assert len(labels) == 1, recording
        rows.append((recording, labels[0], features[indices], syllables[indices]))
    return rows, features.shape[1]


def sample_whole_recordings(rows, max_points, seed):
    if max_points <= 0 or sum(row[2].shape[0] for row in rows) <= max_points:
        return rows
    rng = np.random.default_rng(seed)
    by_bird = defaultdict(list)
    for row in rows:
        by_bird[row[1]].append(row)
    target = max(1, max_points // len(by_bird))
    sampled = []
    for bird in sorted(by_bird):
        bird_rows = by_bird[bird]
        order = rng.permutation(len(bird_rows))
        total = 0
        chosen = []
        for index in order:
            row = bird_rows[int(index)]
            length = row[2].shape[0]
            if chosen and total + length > target:
                continue
            chosen.append(row)
            total += length
            if total >= target:
                break
        if not chosen:
            chosen = [min(bird_rows, key=lambda row: row[2].shape[0])]
        sampled.extend(chosen)
    return sorted(sampled, key=lambda row: row[0])


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
    assert model.cell == "gru"
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, args


def one_layer_gru(model, layer_index):
    input_dim = model.encoder.input_size if layer_index == 0 else model.hidden_dim * model.directions
    layer = nn.GRU(input_dim, model.hidden_dim, num_layers=1, batch_first=True, bidirectional=model.directions == 2)
    state = {}
    for key, value in model.encoder.state_dict().items():
        marker = f"_l{layer_index}"
        if marker in key:
            state[key.replace(marker, "_l0")] = value.detach().cpu()
    layer.load_state_dict(state)
    return layer.to(next(model.parameters()).device).eval()


@torch.no_grad()
def collect_layer_states(rows, model, layer_number, device):
    assert 1 <= layer_number <= model.layers
    layers = [one_layer_gru(model, i) for i in range(layer_number)]
    x_parts = []
    bird_parts = []
    recording_parts = []
    syllable_parts = []
    for recording, bird, seq, syllables in rows:
        x = torch.from_numpy(seq)[None].to(device=device, dtype=torch.float32)
        for layer in layers:
            x, _ = layer(x)
        h = x.squeeze(0).cpu().numpy().astype(np.float32)
        x_parts.append(h)
        bird_parts.extend([bird] * h.shape[0])
        recording_parts.extend([recording] * h.shape[0])
        syllable_parts.append(syllables[: h.shape[0]])
    return (
        np.vstack(x_parts).astype(np.float32, copy=False),
        np.asarray(bird_parts, dtype=object),
        np.asarray(recording_parts, dtype=object),
        np.concatenate(syllable_parts).astype(np.int64, copy=False),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="UMAP unpooled t2vec GRU encoder layer states.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--model_pt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--species_display_name", required=True)
    parser.add_argument("--encoder_layer", type=int, default=1)
    parser.add_argument("--max_umap_points", type=int, default=25000)
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
    sampled_rows = sample_whole_recordings(rows, args.max_umap_points, args.seed)
    model, model_args = load_model(args.model_pt, input_dim, device)
    features, bird_labels, recording_labels, syllable_labels = collect_layer_states(sampled_rows, model, args.encoder_layer, device)

    rep_name = f"t2vec_gru_encoder_layer{args.encoder_layer}_unpooled_raw_samplewhole{args.max_umap_points}"
    print(f"[umap] {rep_name}: recordings={len(sampled_rows)}/{len(rows)} points={features.shape[0]} dim={features.shape[1]}")
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
    np.savez_compressed(
        out_dir / f"{rep_name}_features.npz",
        features=features,
        xy=xy.astype(np.float32, copy=False),
        bird_labels=bird_labels,
        syllable_labels=syllable_labels,
        recording_labels=recording_labels,
    )
    summary = {
        "features_npz": args.features_npz,
        "model_pt": args.model_pt,
        "device": str(device),
        "model_args": model_args,
        "representation": rep_name,
        "encoder_layer": int(args.encoder_layer),
        "source_recordings": int(len(rows)),
        "sampled_recordings": int(len(sampled_rows)),
        "points": int(features.shape[0]),
        "raw_feature_dim": int(features.shape[1]),
        "max_umap_points": int(args.max_umap_points),
        "whole_recording_sample": True,
        "umap": {
            "neighbors": int(args.umap_neighbors),
            "min_dist": float(args.umap_min_dist),
            "metric": args.umap_metric,
            "negative_sample_rate": int(args.umap_negative_sample_rate),
            "supervised": False,
        },
        "silhouette_scores": silhouette_scores,
        "hdbscan_summary": hdbscan_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
