import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from individual_id.latent_trajectory_decoder.train_rope_sequence import load_rows
from individual_id.latent_trajectory_decoder.train_t2vec_gru_latents import T2VecLatentGRU
from individual_id.latent_trajectory_decoder.train_t2vec_gru_contrastive_windows import crop_window, pad_sequences
from individual_id.run_individual_id_umap import _fit_umap, _hdbscan_umap_analysis, _scatter_umap, _umap_silhouette_scores


def load_model(path, input_dim, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    model = T2VecLatentGRU(input_dim, int(args["hidden_dim"]), int(args["layers"]), bool(args["bidirectional"]), "gru").to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, args


@torch.no_grad()
def collect_bottlenecks(rows, model, model_args, args, input_dim, device):
    rng = np.random.default_rng(args.seed)
    features = []
    bird_labels = []
    recording_labels = []
    for recording, bird, seq in rows:
        for _ in range(args.windows_per_recording):
            window = crop_window(seq, int(model_args["window_size"]), rng)
            enc, lengths, _ = pad_sequences([window], input_dim)
            clean, _, _ = pad_sequences([window], input_dim)
            _, bottleneck, _ = model.encode(enc.to(device), lengths.to(device))
            features.append(bottleneck.squeeze(0).cpu().numpy().astype(np.float32))
            bird_labels.append(bird)
            recording_labels.append(recording)
    return np.vstack(features), np.asarray(bird_labels, dtype=object), np.asarray(recording_labels, dtype=object)


def parse_args():
    parser = argparse.ArgumentParser(description="UMAP random-window t2vec GRU bottlenecks.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--model_pt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--species_display_name", required=True)
    parser.add_argument("--windows_per_recording", type=int, default=4)
    parser.add_argument("--umap_neighbors", type=int, default=20)
    parser.add_argument("--umap_min_dist", type=float, default=0.025)
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
    features, bird_labels, recording_labels = collect_bottlenecks(rows, model, model_args, args, input_dim, device)
    syllable_labels = np.zeros(len(features), dtype=np.int64)
    size = int(model_args["window_size"])
    source = "sequence" if size <= 0 else f"window_w{size}"
    rep_name = f"t2vec_gru_{source}_bottleneck_n{args.umap_neighbors}_md{args.umap_min_dist:g}"
    print(f"[umap] {rep_name}: recordings={len(rows)} windows={features.shape[0]} dim={features.shape[1]}")
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
        features=features.astype(np.float32, copy=False),
        xy=xy.astype(np.float32, copy=False),
        bird_labels=bird_labels,
        recording_labels=recording_labels,
        syllable_labels=syllable_labels,
    )
    summary = {
        "features_npz": args.features_npz,
        "model_pt": args.model_pt,
        "device": str(device),
        "model_args": model_args,
        "representation": rep_name,
        "recordings": int(len(rows)),
        "points": int(features.shape[0]),
        "raw_feature_dim": int(features.shape[1]),
        "windows_per_recording": int(args.windows_per_recording),
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
