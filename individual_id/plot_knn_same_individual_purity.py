#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "individual_id"))

import plot_recording_embedding_similarity as recording_similarity  # noqa: E402


def _parse_k_values(text):
    values = sorted({int(value) for value in text.split(",") if value.strip()})
    assert values and values[0] > 0
    return values


def _sample_points(args, rows):
    bird_ids = sorted({row["bird_id"] for row in rows})
    bird_to_code = {bird_id: index for index, bird_id in enumerate(bird_ids)}

    features = []
    point_birds = []
    point_recordings = []
    recording_birds = []
    sampled_counts = []
    for recording_index, row in enumerate(rows):
        x = row["features"]
        if args.max_points_per_recording > 0 and x.shape[0] > args.max_points_per_recording:
            seed = recording_similarity.hashlib.sha1(
                f"{args.seed}|{row['bird_id']}|{row['recording_stem']}|knn".encode("utf-8")
            ).hexdigest()
            rng = np.random.default_rng(int(seed[:8], 16))
            indices = rng.choice(x.shape[0], size=args.max_points_per_recording, replace=False)
            indices.sort()
            x = x[indices]
        features.append(x.astype(np.float32, copy=False))
        bird_code = bird_to_code[row["bird_id"]]
        point_birds.extend([bird_code] * int(x.shape[0]))
        point_recordings.extend([recording_index] * int(x.shape[0]))
        recording_birds.append(bird_code)
        sampled_counts.append(int(x.shape[0]))

    return {
        "features": np.vstack(features).astype(np.float32, copy=False),
        "point_birds": np.asarray(point_birds, dtype=np.int64),
        "point_recordings": np.asarray(point_recordings, dtype=np.int64),
        "recording_birds": np.asarray(recording_birds, dtype=np.int64),
        "bird_ids": np.asarray(bird_ids, dtype=object),
        "sampled_counts": np.asarray(sampled_counts, dtype=np.int64),
    }


def _chance_curve(point_birds, point_recordings, recording_birds, exclude_same_recording):
    if not exclude_same_recording:
        bird_counts = np.bincount(point_birds)
        same = bird_counts[point_birds] - 1
        total = point_birds.size - 1
        return float(np.mean(same / max(total, 1)))

    bird_counts = np.bincount(point_birds)
    recording_counts = np.bincount(point_recordings)
    same = bird_counts[point_birds] - recording_counts[point_recordings]
    total = point_birds.size - recording_counts[point_recordings]
    return float(np.mean(same / np.maximum(total, 1)))


def _knn_purity(args, sampled, k_values):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x = sampled["features"]
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    point_birds = torch.from_numpy(sampled["point_birds"]).to(device=device, dtype=torch.long)
    point_recordings = torch.from_numpy(sampled["point_recordings"]).to(device=device, dtype=torch.long)
    features = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    total_points = int(features.shape[0])

    recording_counts = torch.bincount(point_recordings)
    if args.exclude_same_recording:
        max_k = min(max(k_values), total_points - int(recording_counts.max().item()))
    else:
        max_k = min(max(k_values), total_points - 1)
    assert max_k > 0
    k_values = [k for k in k_values if k <= max_k]
    heatmap_k = min(int(args.heatmap_k), max_k)

    sums = torch.zeros(len(k_values), device=device, dtype=torch.float64)
    recording_neighbors = torch.zeros(
        (sampled["sampled_counts"].size, sampled["sampled_counts"].size),
        device=device,
        dtype=torch.float32,
    )
    bird_neighbors = torch.zeros(
        (sampled["bird_ids"].size, sampled["bird_ids"].size),
        device=device,
        dtype=torch.float32,
    )
    arange = torch.arange(total_points, device=device)
    for start in range(0, total_points, int(args.knn_chunk_size)):
        end = min(start + int(args.knn_chunk_size), total_points)
        sims = features[start:end] @ features.T
        if args.exclude_same_recording:
            sims[point_recordings[start:end, None] == point_recordings[None, :]] = -float("inf")
        else:
            sims[torch.arange(end - start, device=device), arange[start:end]] = -float("inf")
        neighbors = torch.topk(sims, k=max_k, dim=1).indices
        same = point_birds[neighbors] == point_birds[start:end, None]
        cumulative = torch.cumsum(same.to(torch.float32), dim=1)
        for index, k in enumerate(k_values):
            sums[index] += (cumulative[:, k - 1] / float(k)).sum().to(torch.float64)

        source_recordings = point_recordings[start:end, None].expand(-1, heatmap_k).reshape(-1)
        target_recordings = point_recordings[neighbors[:, :heatmap_k]].reshape(-1)
        source_birds = point_birds[start:end, None].expand(-1, heatmap_k).reshape(-1)
        target_birds = point_birds[neighbors[:, :heatmap_k]].reshape(-1)
        ones = torch.ones_like(source_recordings, dtype=torch.float32)
        recording_neighbors.index_put_((source_recordings, target_recordings), ones, accumulate=True)
        bird_neighbors.index_put_((source_birds, target_birds), ones, accumulate=True)

    purity = (sums / float(total_points)).detach().cpu().numpy().astype(np.float32, copy=False)
    recording_denominator = torch.bincount(point_recordings, minlength=sampled["sampled_counts"].size).to(torch.float32)
    bird_denominator = torch.bincount(point_birds, minlength=sampled["bird_ids"].size).to(torch.float32)
    recording_neighbors = recording_neighbors / torch.clamp(recording_denominator[:, None] * float(heatmap_k), min=1.0)
    bird_neighbors = bird_neighbors / torch.clamp(bird_denominator[:, None] * float(heatmap_k), min=1.0)
    return (
        k_values,
        purity,
        str(device),
        heatmap_k,
        recording_neighbors.detach().cpu().numpy().astype(np.float32, copy=False),
        bird_neighbors.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _bird_spans(recording_birds):
    spans = []
    start = 0
    while start < len(recording_birds):
        end = start + 1
        while end < len(recording_birds) and recording_birds[end] == recording_birds[start]:
            end += 1
        spans.append((recording_birds[start], start, end))
        start = end
    return spans


def _save_recording_heatmap(out_dir, rows, sampled, recording_neighbors, heatmap_k):
    bird_ids = sampled["bird_ids"]
    spans = _bird_spans(sampled["recording_birds"])
    fig_size = max(7.0, min(18.0, 0.22 * recording_neighbors.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=300)
    vmax = max(float(np.percentile(recording_neighbors, 99.5)), 1e-6)
    image = ax.imshow(recording_neighbors, cmap="viridis", vmin=0.0, vmax=vmax)

    for _, start, end in spans:
        ax.axhline(start - 0.5, color="black", linewidth=0.6)
        ax.axvline(start - 0.5, color="black", linewidth=0.6)
        ax.axhline(end - 0.5, color="black", linewidth=0.6)
        ax.axvline(end - 0.5, color="black", linewidth=0.6)

    centers = [(start + end - 1) / 2 for _, start, end in spans]
    labels = [str(bird_ids[bird_code]) for bird_code, _, _ in spans]
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Recording kNN neighbor attribution (k={heatmap_k})")
    ax.set_xlabel("Neighbor recording grouped by individual")
    ax.set_ylabel("Query recording grouped by individual")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Fraction of neighbor slots")
    fig.tight_layout()
    fig.savefig(out_dir / "recording_knn_neighbor_heatmap.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "recording_knn_neighbor_heatmap.pdf", bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _save_individual_heatmap(out_dir, sampled, bird_neighbors, heatmap_k):
    labels = [str(x) for x in sampled["bird_ids"]]
    fig_size = max(5.0, min(12.0, 0.35 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=300)
    image = ax.imshow(bird_neighbors, cmap="viridis", vmin=0.0, vmax=max(float(bird_neighbors.max()), 1e-6))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Individual kNN neighbor attribution (k={heatmap_k})")
    ax.set_xlabel("Neighbor individual")
    ax.set_ylabel("Query individual")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Fraction of neighbor slots")
    fig.tight_layout()
    fig.savefig(out_dir / "individual_knn_neighbor_heatmap.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "individual_knn_neighbor_heatmap.pdf", bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _save_recording_distribution(out_dir, sampled, recording_neighbors, heatmap_k):
    same = sampled["recording_birds"][:, None] == sampled["recording_birds"][None, :]
    off_diagonal = ~np.eye(recording_neighbors.shape[0], dtype=bool)
    within = recording_neighbors[same & off_diagonal]
    between = recording_neighbors[~same & off_diagonal]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    xmax = max(float(np.percentile(recording_neighbors[off_diagonal], 99.5)), 1e-6)
    bins = np.linspace(0.0, xmax, 80)
    ax.hist(between, bins=bins, alpha=0.65, density=True, label="Between individuals")
    ax.axvline(float(between.mean()), color="tab:blue", linestyle="--", linewidth=1.0)
    ax.hist(within, bins=bins, alpha=0.65, density=True, label="Within individuals")
    ax.axvline(float(within.mean()), color="tab:orange", linestyle="--", linewidth=1.0)
    ax.set_title(f"Recording kNN neighbor attribution (k={heatmap_k})")
    ax.set_xlabel("Fraction of neighbor slots")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, xmax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "recording_knn_neighbor_distributions.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "recording_knn_neighbor_distributions.pdf", bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _save_per_recording_same_individual_plot(out_dir, sampled, recording_neighbors, heatmap_k):
    same = sampled["recording_birds"][:, None] == sampled["recording_birds"][None, :]
    np.fill_diagonal(same, False)
    different = sampled["recording_birds"][:, None] != sampled["recording_birds"][None, :]
    within = (recording_neighbors * same).sum(axis=1)
    between = (recording_neighbors * different).sum(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    bins = np.linspace(0.0, 1.0, 60)
    ax.hist(between, bins=bins, color="tab:blue", alpha=0.6, density=False, label="Between birds")
    ax.axvline(float(between.mean()), color="tab:blue", linestyle="--", linewidth=1.0)
    ax.hist(within, bins=bins, color="tab:orange", alpha=0.6, density=False, label="Within bird")
    ax.axvline(float(within.mean()), color="tab:orange", linestyle="--", linewidth=1.0)
    ax.set_title(f"Per-recording kNN neighbor destination (k={heatmap_k})")
    ax.set_xlabel("Fraction of neighbor slots")
    ax.set_ylabel("Recordings")
    ax.set_xlim(0.0, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "per_recording_same_individual_purity.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "per_recording_same_individual_purity.pdf", bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)
    return within.astype(np.float32, copy=False), between.astype(np.float32, copy=False)


def _write_outputs(
    args,
    rows,
    sampled,
    k_values,
    purity,
    chance,
    device,
    heatmap_k,
    recording_neighbors,
    bird_neighbors,
    feature_postprocess,
):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    ax.plot(k_values, purity, marker="o", linewidth=1.6, label="kNN same-individual purity")
    ax.axhline(chance, color="0.4", linestyle="--", linewidth=1.0, label="Chance")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("k nearest neighbors")
    ax.set_ylabel("Same-individual fraction")
    ax.set_title(f"{args.species} | frame kNN purity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "knn_same_individual_purity.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "knn_same_individual_purity.pdf", bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)

    _save_recording_heatmap(out_dir, rows, sampled, recording_neighbors, heatmap_k)
    _save_individual_heatmap(out_dir, sampled, bird_neighbors, heatmap_k)
    _save_recording_distribution(out_dir, sampled, recording_neighbors, heatmap_k)
    per_recording_same, per_recording_between = _save_per_recording_same_individual_plot(
        out_dir, sampled, recording_neighbors, heatmap_k
    )

    np.savez(
        out_dir / "knn_same_individual_purity.npz",
        k_values=np.asarray(k_values, dtype=np.int64),
        purity=purity,
        chance=np.asarray(chance, dtype=np.float32),
        bird_ids=sampled["bird_ids"],
        sampled_point_counts=sampled["sampled_counts"],
        recording_neighbor_fraction=recording_neighbors,
        individual_neighbor_fraction=bird_neighbors,
        per_recording_same_individual_fraction=per_recording_same,
        per_recording_between_individual_fraction=per_recording_between,
    )

    summary = {
        "species": args.species,
        "run_dir": str(args.run_dir),
        "checkpoint": args.checkpoint,
        "recording_mode": args.recording_mode,
        "embedding_variant": args.embedding_variant,
        "feature_postprocess": recording_similarity._feature_postprocess_summary(feature_postprocess),
        "device": device,
        "exclude_same_recording": bool(args.exclude_same_recording),
        "recordings": int(len(rows)),
        "individuals": int(sampled["bird_ids"].size),
        "points": int(sampled["features"].shape[0]),
        "max_points_per_recording": int(args.max_points_per_recording),
        "chance": chance,
        "heatmap_k": int(heatmap_k),
        "k_values": [int(k) for k in k_values],
        "purity": [float(x) for x in purity],
        "purity_minus_chance": [float(x - chance) for x in purity],
        "per_recording_same_individual_fraction": recording_similarity._summarize(per_recording_same),
        "per_recording_between_individual_fraction": recording_similarity._summarize(per_recording_between),
        "per_recording_sampled_point_count": recording_similarity._summarize(sampled["sampled_counts"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Plot frame-level kNN same-individual purity.")
    parser.add_argument("--species", required=True)
    parser.add_argument("--annotation_json", required=True)
    parser.add_argument("--spec_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out_dir", default=str(ROOT / "results" / "individual_id_knn_purity"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--recording_mode", default="full_recordings", choices=["events", "full_recordings"])
    parser.add_argument("--songs_per_bird", type=int, default=30)
    parser.add_argument("--max_birds", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_points_per_recording", type=int, default=200)
    parser.add_argument("--k_values", default="1,2,5,10,20,50,100")
    parser.add_argument("--heatmap_k", type=int, default=10)
    parser.add_argument("--knn_chunk_size", type=int, default=512)
    parser.add_argument("--exclude_same_recording", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--embedding_variant", default="before", choices=["before", "after"])
    parser.add_argument("--feature_postprocess", default="pca_whiten_l2", choices=["none", "pca_whiten_l2", "whiten_l2"])
    parser.add_argument("--feature_postprocess_dim", type=int, default=64)
    parser.add_argument("--feature_postprocess_load", default=None)
    parser.add_argument("--feature_postprocess_save", default=None)
    parser.add_argument("--encoder_layer_idx", type=int, default=None)
    parser.add_argument("--drop_silence", action="store_true")
    parser.add_argument(
        "--spec_normalization",
        default="auto",
        choices=[
            "auto",
            "none",
            "audio_params",
            "per_recording_cmvn",
            "per_recording_cmvn_rescaled_to_target_stats",
            "per_model_input_zscore",
        ],
    )
    parser.add_argument("--normalization_stats_dir", default=None)
    args = parser.parse_args()

    args.annotation_json = str(Path(args.annotation_json).resolve())
    args.spec_dir = str(Path(args.spec_dir).resolve())
    args.run_dir = str(recording_similarity._resolve_run_dir(args.run_dir))
    args.out_dir = str(Path(args.out_dir).resolve())
    assert args.max_points_per_recording > 0
    assert args.heatmap_k > 0
    assert args.knn_chunk_size > 0

    if args.feature_postprocess_load is not None:
        args.feature_postprocess_load = str(Path(args.feature_postprocess_load).resolve())
    if args.feature_postprocess_save is not None:
        save_path = Path(args.feature_postprocess_save).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        args.feature_postprocess_save = str(save_path)

    model_state = recording_similarity.extract_embedding.load_model_state(
        {"run_dir": args.run_dir, "checkpoint": args.checkpoint}
    )
    if args.spec_normalization == "auto":
        args.spec_normalization, args.normalization_stats_dir = (
            recording_similarity.extract_embedding.get_native_input_normalization(model_state)
        )

    rows, feature_postprocess = recording_similarity._build_recording_table(args, model_state)
    sampled = _sample_points(args, rows)
    k_values = _parse_k_values(args.k_values)
    k_values, purity, device, heatmap_k, recording_neighbors, bird_neighbors = _knn_purity(args, sampled, k_values)
    chance = _chance_curve(
        sampled["point_birds"],
        sampled["point_recordings"],
        sampled["recording_birds"],
        args.exclude_same_recording,
    )
    _write_outputs(
        args,
        rows,
        sampled,
        k_values,
        purity,
        chance,
        device,
        heatmap_k,
        recording_neighbors,
        bird_neighbors,
        feature_postprocess,
    )

    print(
        "[knn-purity] "
        f"species={args.species} recordings={len(rows)} individuals={sampled['bird_ids'].size} "
        f"points={sampled['features'].shape[0]} k={k_values} purity={purity.tolist()} "
        f"chance={chance} out_dir={args.out_dir}"
    )


if __name__ == "__main__":
    main()
