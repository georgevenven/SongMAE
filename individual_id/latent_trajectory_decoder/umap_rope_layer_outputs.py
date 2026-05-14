import argparse
import csv
import json
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from individual_id.latent_trajectory_decoder.train_rope_sequence import RopeSequenceDecoder, SinusoidalSequenceDecoder
from individual_id.run_individual_id_umap import (
    _fit_umap,
    _hdbscan_umap_analysis,
    _scatter_umap,
    _scatter_umap_syllables,
    _umap_silhouette_scores,
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
    model_class = RopeSequenceDecoder
    if model_args.get("position_encoding") == "sinusoidal":
        model_class = SinusoidalSequenceDecoder
    model = model_class(
        input_dim,
        int(model_args["d_model"]),
        int(model_args["heads"]),
        int(model_args["layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["decoder"])
    model.eval()
    return model, model_args


@torch.no_grad()
def collect_layer_outputs(rows, model, layer_count, device):
    reps = {}
    for layer in range(layer_count):
        reps[f"attn_l{layer + 1}"] = []
        reps[f"ffn_l{layer + 1}"] = []

    bird_parts = []
    syllable_parts = []
    recording_parts = []
    for recording, bird, seq, syllables in rows:
        x = torch.from_numpy(seq[:-1]).to(device=device, dtype=torch.float32)[None]
        valid = torch.ones((1, x.shape[1]), device=device, dtype=torch.bool)
        _, _, parts = model.forward_layers(x, valid)
        for layer in range(layer_count):
            reps[f"attn_l{layer + 1}"].append(parts["attn"][layer].squeeze(0).cpu().numpy())
            reps[f"ffn_l{layer + 1}"].append(parts["ffn"][layer].squeeze(0).cpu().numpy())
        count = int(x.shape[1])
        bird_parts.extend([bird] * count)
        syllable_parts.append(syllables[:count])
        recording_parts.extend([recording] * count)

    features = {name: np.vstack(values).astype(np.float32, copy=False) for name, values in reps.items()}
    return (
        features,
        np.asarray(bird_parts, dtype=object),
        np.concatenate(syllable_parts).astype(np.int64, copy=False),
        np.asarray(recording_parts, dtype=object),
    )


def sample_indices(count, max_points, seed):
    if max_points <= 0 or count <= max_points:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    keep = rng.choice(count, size=max_points, replace=False)
    keep.sort()
    return keep.astype(np.int64, copy=False)


def write_csv(path, rows):
    fieldnames = [
        "representation",
        "points",
        "raw_dim",
        "feature_dim",
        "bird_silhouette",
        "syllable_silhouette",
        "hdbscan_clusters",
        "hdbscan_noise_fraction",
        "hdbscan_bird_v_measure",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_collage(out_dir, names, suffix):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=180)
    for ax, name in zip(axes.ravel(), names):
        image = mpimg.imread(out_dir / f"{name}.png")
        ax.imshow(image)
        ax.set_title(name, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"layer_output_umap_collage_{suffix}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="UMAP per-layer RoPE attention and FFN residual outputs.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--model_pt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--species_display_name", required=True)
    parser.add_argument("--max_points", type=int, default=8000)
    parser.add_argument("--feature_postprocess", choices=["none", "pca_whiten_l2"], default="none")
    parser.add_argument("--feature_postprocess_dim", type=int, default=72)
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
    layer_count = int(model_args["layers"])
    raw_by_name, birds, syllables, recordings = collect_layer_outputs(rows, model, layer_count, device)
    keep = sample_indices(birds.shape[0], args.max_points, args.seed)
    birds = birds[keep]
    syllables = syllables[keep]
    recordings = recordings[keep]

    plot_args = Namespace(
        species_display_name=args.species_display_name,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
    )
    summaries = []
    names = []
    for name, raw in raw_by_name.items():
        raw = raw[keep]
        features, transform = extract_embedding.maybe_apply_feature_postprocess(
            raw,
            mode=args.feature_postprocess,
            dim=args.feature_postprocess_dim,
        )
        suffix = "raw" if transform is None else f"pca_whiten_l2{transform['dim']}"
        rep_name = f"{name}_{suffix}"
        names.append(rep_name)
        print(f"[umap] {rep_name}: points={features.shape[0]} dim={features.shape[1]}", flush=True)

        xy = _fit_umap(
            features,
            neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=args.umap_random_state,
            negative_sample_rate=args.umap_negative_sample_rate,
        )
        silhouette = _umap_silhouette_scores(xy, birds, syllables, args.silhouette_sample_size, args.seed)
        hdbscan_summary = _hdbscan_umap_analysis(xy, birds, syllables, recordings, out_dir, rep_name, plot_args)
        _scatter_umap(xy, birds, f"{args.species_display_name}: {rep_name}", out_dir / rep_name)
        _scatter_umap_syllables(
            xy,
            syllables,
            birds,
            f"{args.species_display_name}: {rep_name} syllables",
            out_dir / f"{rep_name}_syllable",
        )
        np.savez_compressed(
            out_dir / f"{rep_name}_features.npz",
            features=features.astype(np.float32, copy=False),
            raw_features=raw.astype(np.float32, copy=False),
            xy=xy.astype(np.float32, copy=False),
            bird_labels=birds,
            syllable_labels=syllables,
            recording_labels=recordings,
        )
        summaries.append(
            {
                "representation": rep_name,
                "points": int(features.shape[0]),
                "raw_dim": int(raw.shape[1]),
                "feature_dim": int(features.shape[1]),
                "bird_silhouette": silhouette["bird"]["score"],
                "syllable_silhouette": silhouette["syllable"]["score"],
                "hdbscan_clusters": hdbscan_summary["clusters"],
                "hdbscan_noise_fraction": hdbscan_summary["noise_fraction"],
                "hdbscan_bird_v_measure": hdbscan_summary["bird_v_measure"],
            }
        )

    write_csv(out_dir / "layer_output_umap_summary.csv", summaries)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "features_npz": args.features_npz,
                "model_pt": args.model_pt,
                "device": str(device),
                "input_dim": int(input_dim),
                "d_model": int(model_args["d_model"]),
                "heads": int(model_args["heads"]),
                "layers": layer_count,
                "position_encoding": model_args.get("position_encoding", "rope"),
                "sampled_points": int(keep.shape[0]),
                "args": vars(args),
                "representations": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    make_collage(out_dir, names, "raw" if args.feature_postprocess == "none" else f"pca_whiten_l2{summaries[0]['feature_dim']}")


if __name__ == "__main__":
    main()
