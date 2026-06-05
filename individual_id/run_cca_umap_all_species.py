#!/usr/bin/env python3

import csv
import argparse
import colorsys
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/george-vengrovski/anaconda3/envs/mae/bin/python")
STABLE_RANK_CSV = ROOT / "results" / "individual_id_knn_graph_metrics" / "default_config_affinity_rank_spectra" / "affinity_rank_summary.csv"
DEFAULT_RECORDING_SVD_ROOT = "results/individual_id_knn_graph_metrics/affinity_matrix_k_sweep_songmae_unbinned_usable_allrecordings_nocap_fullgpu/k64"
DEFAULT_SPECIES = None


SPECIES = {
    "zf": {
        "display": "Zebra Finch",
        "annotation": "files/zf_annotations.json",
        "spec_dir": "/media/george-vengrovski/disk2/specs/zf_64hop_32khz",
        "svd": "results/individual_id_knn_graph_metrics/bird_knn_matrix_maxdata_uncapped/zf/knn_attribution_matrices.npz",
        "songs": 0,
        "window": 10,
        "hop": 2,
    },
    "bf": {
        "display": "Bengalese Finch",
        "annotation": "files/bf_annotations.json",
        "spec_dir": "/media/george-vengrovski/disk2/specs/bf_64hop_32khz",
        "svd": "results/individual_id_knn_graph_metrics/bird_knn_matrix_perbird200_uncapped/bf/knn_attribution_matrices.npz",
        "songs": 200,
        "window": 10,
        "hop": 2,
    },
    "canary": {
        "display": "Canary",
        "annotation": "files/canary_annotations_for_individual_id.json",
        "spec_dir": "/media/george-vengrovski/disk2/specs/canary_individual_identification_64hop_32khz",
        "svd": "results/individual_id_knn_graph_metrics/bird_knn_matrix_maxdata_uncapped/canary/knn_attribution_matrices.npz",
        "songs": 0,
        "window": 30,
        "hop": 5,
    },
    "chiffchaff": {
        "display": "Chiffchaff",
        "annotation": "files/chiffchaff_annotations.json",
        "spec_dir": "/media/george-vengrovski/disk2/specs/chiffchaff_64hop_32khz",
        "svd": "results/individual_id_knn_graph_metrics/bird_knn_matrix_recordings2000_uncapped/chiffchaff/knn_attribution_matrices.npz",
        "songs": 88,
        "window": 30,
        "hop": 5,
    },
    "european_starling": {
        "display": "European Starling",
        "annotation": "files/european_starling_annotations_unprefixed.json",
        "spec_dir": "/media/george-vengrovski/disk2/specs/european_starling_64hop_32khz",
        "affinity_sweep_annotation": "files/european_starling_annotations_fixed.json",
        "affinity_sweep_spec_dir": "/media/george-vengrovski/disk2/specs/european_starling_64hop_32khz_prefixed",
        "svd": "results/individual_id_knn_graph_metrics/bird_knn_matrix_laplacian_cap30/european_starling/knn_attribution_matrices.npz",
        "songs": 30,
        "window": 30,
        "hop": 5,
        "features_25000": "results/individual_id_umap/european_starling_metric_learning_source_stats_maxpts25000/songmae_pool_stats_sliding_w30_h5_maxpts25000_recsvd15_a1_recviewunion_pca_whiten_l21024_features.npz",
        "features_100000": "results/individual_id_umap/european_starling_metric_learning_source_stats_maxpts100000/songmae_pool_stats_sliding_w30_h5_maxpts100000_recsvd15_a1_recviewunion_pca_whiten_l21024_features.npz",
    },
    "little_owl": {
        "display": "Little Owl",
        "annotation": "files/little_owl_annotations.json",
        "spec_dir": "/media/george-vengrovski/disk2/specs/little_owl_64hop_32khz",
        "svd": "results/individual_id_knn_graph_metrics/bird_knn_matrix_maxdata_uncapped/little_owl/knn_attribution_matrices.npz",
        "songs": 0,
        "window": 5,
        "hop": 2,
        "features_25000": "results/individual_id_umap/little_owl_metric_learning_source_stats_w5_h2_maxpts100000/songmae_pool_stats_sliding_w5_h2_maxpts100000_recsvd15_a1_recviewunion_pca_whiten_l21024_features.npz",
        "features_100000": "results/individual_id_umap/little_owl_metric_learning_source_stats_w5_h2_maxpts100000/songmae_pool_stats_sliding_w5_h2_maxpts100000_recsvd15_a1_recviewunion_pca_whiten_l21024_features.npz",
    },
    "ovenbird": {
        "display": "Ovenbird",
        "annotation": "files/lapp_ovenbird.json",
        "spec_dir": "/media/george-vengrovski/disk2/specs/ovenbird_lapp_sample_64hop_32khz",
        "svd": "results/individual_id_knn_graph_metrics/bird_knn_matrix_maxdata_uncapped/ovenbird/knn_attribution_matrices.npz",
        "songs": 0,
        "window": 5,
        "hop": 2,
        "features_25000": "results/individual_id_umap/ovenbird_metric_learning_source_stats_default_maxpts100000/songmae_pool_stats_sliding_w5_h2_maxpts100000_recsvd15_a1_recviewunion_pca_whiten_l21024_features.npz",
        "features_100000": "results/individual_id_umap/ovenbird_metric_learning_source_stats_default_maxpts100000/songmae_pool_stats_sliding_w5_h2_maxpts100000_recsvd15_a1_recviewunion_pca_whiten_l21024_features.npz",
    },
    "tree_pipit": {
        "display": "Tree Pipit",
        "annotation": "files/tree_pipit_annotations.json",
        "spec_dir": "/media/george-vengrovski/disk2/specs/tree_pipit_64hop_32khz",
        "svd": "results/individual_id_knn_graph_metrics/bird_knn_matrix_perbird400_uncapped/tree_pipit/knn_attribution_matrices.npz",
        "songs": 0,
        "window": 10,
        "hop": 2,
    },
}


def _summary_path(out_dir):
    return out_dir / "summary.json"


def _hdbscan_points(summary):
    return Path(summary["feature_path"]).with_name(f"{summary['representation']}_hdbscan_points.npz")


def _stable_rank_from_npz(path):
    data = np.load(path, allow_pickle=True)
    matrix = data["recording_matrix"].astype(np.float64, copy=False)
    matrix = (matrix + matrix.T) * 0.5
    np.fill_diagonal(matrix, 0.0)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    energy = singular_values * singular_values
    return max(2, round(float(energy.sum() / max(float(energy[0]), 1e-12))))


def _stable_rank_dims(args, species_keys):
    if args.recording_svd_root is not None:
        return {
            species: _stable_rank_from_npz(_recording_svd_path(args, species, config))
            for species, config in SPECIES.items()
            if species in species_keys
        }
    with STABLE_RANK_CSV.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {row["species"]: max(2, round(float(row["stable_rank"]))) for row in rows}


def _ensure_ari(summary_path):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if "hdbscan_adjusted_rand_index" in data:
        return data
    points = np.load(_hdbscan_points(data), allow_pickle=True)
    labels = points["bird_labels"]
    clusters = points["hdbscan_clusters"]
    non_noise = clusters >= 0
    data["hdbscan_adjusted_rand_index"] = float(adjusted_rand_score(labels, clusters))
    data["hdbscan_adjusted_rand_index_non_noise"] = (
        float(adjusted_rand_score(labels[non_noise], clusters[non_noise])) if non_noise.any() else None
    )
    summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def _recording_svd_path(args, species, config):
    if args.recording_svd_root is None:
        return config["svd"]
    return str(Path(args.recording_svd_root) / species / "knn_attribution_matrices.npz")


def _run_config(args, config):
    if args.recording_svd_root is None:
        return config
    config = dict(config)
    if "affinity_sweep_annotation" in config:
        config["annotation"] = config["affinity_sweep_annotation"]
        config["spec_dir"] = config["affinity_sweep_spec_dir"]
    return config


def _command(species, config, max_points, out_dir, recording_svd_npz, recording_feature_mode, recording_svd_dim, cca_mode, cca_components, pool_window, pool_hop, pool_mode, songs_per_bird, use_feature_cache, recording_level_stats_pool, cca_block_normalize, recording_cca_weight, hdbscan_min_cluster_size, hdbscan_min_samples):
    cmd = [
        str(PYTHON),
        "individual_id/run_individual_id_cca_umap.py",
        "--species", species,
        "--species_display_name", config["display"],
        "--annotation_json", config["annotation"],
        "--spec_dir", config["spec_dir"],
        "--audio_params_stats_dir", config["spec_dir"],
        "--recording_svd_npz", recording_svd_npz,
        "--recording_feature_mode", recording_feature_mode,
        "--recording_svd_dim", str(recording_svd_dim),
        "--cca_mode", cca_mode,
        "--out_dir", str(out_dir.relative_to(ROOT)),
        "--songs_per_bird", str(songs_per_bird),
        "--pool_window", str(pool_window),
        "--pool_hop", str(pool_hop),
        "--pool_mode", pool_mode,
        "--max_points", str(max_points),
        "--hdbscan_min_cluster_size", str(hdbscan_min_cluster_size),
        "--hdbscan_min_samples", str(hdbscan_min_samples),
    ]
    if recording_level_stats_pool:
        cmd.append("--recording_level_stats_pool")
    if cca_block_normalize:
        cmd.append("--cca_block_normalize")
        cmd.extend(["--recording_cca_weight", str(recording_cca_weight)])
    else:
        cmd.append("--no-cca_block_normalize")
    if cca_components is not None:
        cmd.extend(["--cca_components", str(cca_components)])
    feature_key = f"features_{max_points}"
    if use_feature_cache and feature_key in config and (ROOT / config[feature_key]).is_file():
        cmd.extend(["--features_npz", config[feature_key]])
    else:
        cmd.extend(["--features_npz", ""])
    return cmd


def _row(species, max_points, out_dir, status, code, summary=None):
    row = {
        "species": species,
        "max_points": max_points,
        "status": status,
        "exit_code": code,
        "out_dir": str(out_dir),
        "points": "",
        "recording_view_dim": "",
        "cca_ridge_b": "",
        "hdbscan_clusters": "",
        "hdbscan_noise_fraction": "",
        "hdbscan_ari": "",
        "hdbscan_ari_non_noise": "",
        "bird_silhouette": "",
    }
    if summary is None:
        return row
    hdb = summary["hdbscan_summary"]
    row.update(
        {
            "points": summary["points"],
            "recording_view_dim": summary["recording_view_dim"],
            "cca_ridge_b": summary.get("cca_ridge_b"),
            "hdbscan_clusters": hdb["clusters"],
            "hdbscan_noise_fraction": hdb["noise_fraction"],
            "hdbscan_ari": summary["hdbscan_adjusted_rand_index"],
            "hdbscan_ari_non_noise": summary["hdbscan_adjusted_rand_index_non_noise"],
            "bird_silhouette": summary["silhouette_scores"]["bird"]["score"],
        }
    )
    return row


def _write_status(path, rows):
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _palette(labels):
    labels = sorted(set(labels.tolist()))
    return {
        label: colorsys.hsv_to_rgb((index * 0.618033988749895) % 1.0, 0.82, 0.78)
        for index, label in enumerate(labels)
    }


def _purity(labels, clusters):
    keep = clusters != -1
    assert keep.any()
    total = 0
    for cluster in sorted(set(clusters[keep].tolist())):
        _, counts = np.unique(labels[keep][clusters[keep] == cluster], return_counts=True)
        total += counts.max()
    return total / keep.sum()


def _plot_all_species_umaps(root, species_keys):
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.6), dpi=300, squeeze=False)
    axes = axes.ravel()
    for ax, species in zip(axes, species_keys):
        summary = _ensure_ari(_summary_path(root / species))
        hdbscan = summary["hdbscan_summary"]
        points = np.load(_hdbscan_points(summary), allow_pickle=True)
        xy = points["xy"]
        labels = points["bird_labels"]
        clusters = points["hdbscan_clusters"]
        purity = _purity(labels, clusters)
        palette = _palette(labels)
        for bird in sorted(palette):
            idx = labels == bird
            ax.scatter(xy[idx, 0], xy[idx, 1], s=4, alpha=0.18, color=palette[bird], edgecolors="none")
        ax.set_title(SPECIES[species]["display"], fontsize=14, fontweight="bold")
        ax.text(
            0.98,
            0.98,
            (
                f"V-measure {hdbscan['bird_v_measure']:.3f}\n"
                f"Completeness {hdbscan['bird_completeness']:.3f}\n"
                f"Purity {purity:.3f}"
            ),
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(species_keys):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(root / "all_species_umap_individuals.png", bbox_inches="tight", dpi=300)
    fig.savefig(root / "all_species_umap_individuals.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def run_batch(args, max_points):
    species_keys = args.species.split(",") if args.species else list(SPECIES)
    stable_rank_dims = {} if args.recording_svd_dim_mode == "fixed" else _stable_rank_dims(args, species_keys)
    if args.recording_svd_dim_mode == "stable_rank":
        mode_tag = f"rec{args.recording_feature_mode}stablerank"
    elif args.recording_svd_dim_mode == "stable_rank_x2":
        mode_tag = f"rec{args.recording_feature_mode}stablerankx2"
    elif args.recording_svd_dim_mode == "stable_rank_half":
        mode_tag = f"rec{args.recording_feature_mode}stablerankhalf"
    elif args.recording_svd_dim_mode == "stable_rank_quarter":
        mode_tag = f"rec{args.recording_feature_mode}stablerankquarter"
    elif args.recording_feature_mode == "affinity_row":
        mode_tag = "recaffrow"
    elif args.recording_feature_mode == "pca":
        mode_tag = f"recpca{args.recording_svd_dim}"
    elif args.recording_feature_mode == "umap":
        mode_tag = f"recumap{args.recording_svd_dim}"
    else:
        mode_tag = f"recsvd{args.recording_svd_dim}"
    if args.cca_mode != "thin":
        mode_tag = f"{mode_tag}_{args.cca_mode}"
    if args.cca_components is not None:
        mode_tag = f"{mode_tag}_cca{args.cca_components * 2}"
    if args.recording_svd_tag is not None:
        mode_tag = f"{mode_tag}_{args.recording_svd_tag}"
    if args.recording_level_stats_pool:
        mode_tag = f"{mode_tag}_recordingstats"
    if args.cca_block_normalize:
        mode_tag = f"{mode_tag}_blocknorm"
        if float(args.recording_cca_weight) != 1.0:
            weight = str(args.recording_cca_weight).replace(".", "p")
            mode_tag = f"{mode_tag}_recw{weight}"
    if args.pool_window is not None:
        if args.pool_mode != "stats":
            mode_tag = f"{mode_tag}_{args.pool_mode}"
        mode_tag = f"{mode_tag}_w{args.pool_window}_h{args.pool_hop}_songs{args.songs_per_bird}"
    species_tag = "all8" if args.species is None else f"{len(species_keys)}species"
    root = ROOT / "results" / "individual_id_umap" / f"cca_{mode_tag}_{species_tag}_maxpts{max_points}"
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    rows = []
    for species in species_keys:
        base_config = SPECIES[species]
        config = _run_config(args, base_config)
        out_dir = root / species
        summary_path = _summary_path(out_dir)
        if summary_path.exists():
            summary = _ensure_ari(summary_path)
            rows.append(_row(species, max_points, out_dir, "skipped", 0, summary))
            print(f"[batch] skip {species} maxpts={max_points}")
            continue
        if args.recording_svd_dim_mode == "stable_rank":
            recording_svd_dim = stable_rank_dims[species]
        elif args.recording_svd_dim_mode == "stable_rank_x2":
            recording_svd_dim = stable_rank_dims[species] * 2
        elif args.recording_svd_dim_mode == "stable_rank_half":
            recording_svd_dim = max(3, stable_rank_dims[species] // 2)
        elif args.recording_svd_dim_mode == "stable_rank_quarter":
            recording_svd_dim = max(3, stable_rank_dims[species] // 4)
        else:
            recording_svd_dim = args.recording_svd_dim
        pool_window = config["window"] if args.pool_window is None else args.pool_window
        pool_hop = config["hop"] if args.pool_hop is None else args.pool_hop
        songs_per_bird = config["songs"] if args.songs_per_bird is None else args.songs_per_bird
        use_feature_cache = args.pool_window is None and args.songs_per_bird is None
        recording_svd_npz = _recording_svd_path(args, species, config)
        cmd = _command(species, config, max_points, out_dir, recording_svd_npz, args.recording_feature_mode, recording_svd_dim, args.cca_mode, args.cca_components, pool_window, pool_hop, args.pool_mode, songs_per_bird, use_feature_cache, args.recording_level_stats_pool, args.cca_block_normalize, args.recording_cca_weight, args.hdbscan_min_cluster_size, args.hdbscan_min_samples)
        log_path = logs / f"{species}.log"
        print(f"[batch] run {species} maxpts={max_points}")
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        if result.returncode == 0 and summary_path.exists():
            summary = _ensure_ari(summary_path)
            rows.append(_row(species, max_points, out_dir, "ok", result.returncode, summary))
        else:
            rows.append(_row(species, max_points, out_dir, "failed", result.returncode))
            _write_status(root / "summary.tsv", rows)
            return False
        _write_status(root / "summary.tsv", rows)
    _plot_all_species_umaps(root, species_keys)
    return True


def main():
    parser = argparse.ArgumentParser(description="Run all-species CCA UMAP batches.")
    parser.add_argument("--max_points", type=int, action="append", default=None)
    parser.add_argument("--recording_feature_mode", default="umap", choices=["svd", "pca", "umap", "affinity_row"])
    parser.add_argument("--recording_svd_dim_mode", default="fixed", choices=["fixed", "stable_rank", "stable_rank_x2", "stable_rank_half", "stable_rank_quarter"])
    parser.add_argument("--recording_svd_dim", type=int, default=8)
    parser.add_argument("--recording_svd_root", default=DEFAULT_RECORDING_SVD_ROOT)
    parser.add_argument("--recording_svd_tag", default="unbinnocapk64")
    parser.add_argument("--species", default=DEFAULT_SPECIES)
    parser.add_argument("--cca_mode", default="thin", choices=["thin", "rcca_cv", "rcca_fixed"])
    parser.add_argument("--cca_components", type=int, default=None)
    parser.add_argument("--pool_window", type=int, default=5)
    parser.add_argument("--pool_hop", type=int, default=2)
    parser.add_argument("--pool_mode", default="mean", choices=["mean", "stats", "concat", "concat_pca"])
    parser.add_argument("--songs_per_bird", type=int, default=15)
    parser.add_argument("--recording_level_stats_pool", action="store_true")
    parser.add_argument("--cca_block_normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recording_cca_weight", type=float, default=1.0)
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=0)
    parser.add_argument("--hdbscan_min_samples", type=int, default=10)
    args = parser.parse_args()
    assert (args.pool_window is None) == (args.pool_hop is None)

    max_points = args.max_points or [100000]
    for value in max_points:
        assert run_batch(args, value)


if __name__ == "__main__":
    main()
