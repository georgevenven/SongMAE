#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "individual_id"))

import knn_bird_matrix as km  # noqa: E402

SPECIES = ["zf", "bf", "canary", "chiffchaff", "european_starling", "tree_pipit", "little_owl", "ovenbird"]


def _base_args(args, species):
    return SimpleNamespace(
        species_key=species,
        encoder="SongMAE",
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        out_dir=str(args.out_root),
        songs_per_bird=args.songs_per_bird,
        min_songs_per_bird=0,
        max_recordings=args.max_recordings,
        max_points_per_recording=args.max_points_per_recording,
        max_total_points=args.max_total_points,
        feature_memmap_dir=str(args.feature_memmap_dir) if args.feature_memmap_dir else None,
        k_values=",".join(str(k) for k in args.matrix_ks),
        matrix_k=max(args.matrix_ks),
        postprocess_chunk_size=args.postprocess_chunk_size,
        pca_fit_points=args.pca_fit_points,
        knn_chunk_size=args.knn_chunk_size,
        knn_candidate_chunk_size=args.knn_candidate_chunk_size,
        seed=args.seed,
        exclude_same_recording=True,
        cpu=args.cpu,
        embedding_variant="before",
        encoder_layer_idx=None,
        songmae_affinity_features="linear_probe",
        pool_window=1,
        pool_hop=1,
        pool_mode="mean",
        window_mean_pool=False,
        window_concat_pool=False,
        window_token_probe=False,
        feature_postprocess="pca_whiten_l2",
        feature_postprocess_dim=1024,
        spec_normalization="auto",
        normalization_preset=None,
        audio_params_stats_dir=None,
        normalization_stats_dir=None,
        spec_normalization_stats_dir=None,
        annotation_json_override=None,
        spec_dir_override=None,
        recording_mode_override=None,
        songmae_embedding_variant="before",
        aves_model_path=None,
        aves_config_path=None,
        wav_root=None,
        wav_manifest=None,
        wav_exts=".wav,.flac,.ogg,.mp3",
        aves_audio_sr=16000,
        perch_model_name="perch_v2",
        perch_audio_sr=32000,
        perch_window_seconds=5.0,
        hubert_model_name="facebook/hubert-large-ll60k",
        hubert_audio_sr=16000,
        bird_mae_model_name="DBD-research-group/Bird-MAE-Base",
        bird_mae_audio_sr=32000,
        audio_context_seconds=2.0,
        train_audio_speed_min_pct=0.0,
        train_audio_speed_max_pct=0.0,
    )


def _matrix_path(root, matrix_k, species):
    return root / f"k{matrix_k}" / species / "knn_attribution_matrices.npz"


def _run_species(args, species):
    if not args.force and all(_matrix_path(args.out_root, k, species).exists() for k in args.matrix_ks):
        print(f"[sweep] skip {species}")
        return

    run_args = _base_args(args, species)
    selected = km._selected_recordings(run_args)
    rows = km._extract(run_args, selected)
    sampled = km._sample(run_args, rows)
    km._postprocess_sampled_features(run_args, sampled)
    neighbors, device, actual_k = km._knn(run_args, sampled, max(args.matrix_ks))

    for matrix_k in args.matrix_ks:
        out_dir = args.out_root / f"k{matrix_k}" / species
        run_args.out_dir = str(args.out_root / f"k{matrix_k}")
        run_args.matrix_k = matrix_k
        km._write_outputs(run_args, sampled, neighbors, device, actual_k, out_dir)


def _singular_values(path):
    data = np.load(path, allow_pickle=True)
    matrix = data["recording_matrix"].astype(np.float64, copy=False)
    matrix = (matrix + matrix.T) * 0.5
    np.fill_diagonal(matrix, 0.0)
    return np.linalg.svd(matrix, compute_uv=False), data


def _rank_at_energy(singular_values, fraction):
    energy = singular_values * singular_values
    return int(np.searchsorted(np.cumsum(energy) / max(float(energy.sum()), 1e-12), fraction) + 1)


def _stable_rank(singular_values):
    energy = singular_values * singular_values
    return float(energy.sum() / max(float(energy[0]), 1e-12))


def _pearson(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def _spearman(xs, ys):
    return _pearson(np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys)))


def _r2(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if np.allclose(xs, xs[0]):
        return 0.0
    slope, intercept = np.polyfit(xs, ys, deg=1)
    pred = slope * xs + intercept
    denom = np.sum((ys - ys.mean()) ** 2)
    return 0.0 if denom == 0 else float(1.0 - np.sum((ys - pred) ** 2) / denom)


def _subset_rows(matrix, recording_birds, repeats, seed):
    rng = np.random.default_rng(seed)
    bird_codes = np.unique(recording_birds)
    rows = []
    for count in range(1, bird_codes.size + 1):
        n_repeats = 1 if count == bird_codes.size else repeats
        for _ in range(n_repeats):
            birds = bird_codes if count == bird_codes.size else rng.choice(bird_codes, size=count, replace=False)
            keep = np.flatnonzero(np.isin(recording_birds, birds))
            subset = matrix[np.ix_(keep, keep)]
            rows.append({"true_count": count, "stable_rank": km._stable_rank(subset)})
    return rows


def _write_csv(path, rows):
    assert rows
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _analyze(args):
    spectra_rows = []
    subset_summary = []
    for matrix_k in args.matrix_ks:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=180)
        for ax, species in zip(axes.flat, SPECIES):
            path = _matrix_path(args.out_root, matrix_k, species)
            singular_values, data = _singular_values(path)
            energy = singular_values * singular_values
            stats = {
                "matrix_k": matrix_k,
                "species": species,
                "recordings": int(singular_values.size),
                "birds": int(data["bird_ids"].size),
                "stable_rank": _stable_rank(singular_values),
                "rank_90_energy": _rank_at_energy(singular_values, 0.90),
                "rank_95_energy": _rank_at_energy(singular_values, 0.95),
                "rank_99_energy": _rank_at_energy(singular_values, 0.99),
                "top1_energy_fraction": float(energy[0] / max(float(energy.sum()), 1e-12)),
                "svd_npz": str(path.relative_to(ROOT)),
            }
            spectra_rows.append(stats)
            ranks = np.arange(1, singular_values.size + 1)
            ax.plot(ranks, singular_values / max(float(singular_values[0]), 1e-12), linewidth=1.2)
            ax.axvline(stats["stable_rank"], color="#d95f02", linestyle="--", linewidth=1.0)
            ax.axvline(stats["birds"], color="#202020", linestyle=":", linewidth=1.0)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{species} sr={stats['stable_rank']:.1f} birds={stats['birds']}", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            matrix = np.load(path, allow_pickle=True)["recording_matrix"].astype(np.float32, copy=False)
            rows = _subset_rows(matrix, data["recording_birds"].astype(np.int64), args.subset_repeats, args.seed)
            subset_summary.append(
                {
                    "matrix_k": matrix_k,
                    "species": species,
                    "birds": int(data["bird_ids"].size),
                    "stable_rank_r2": _r2([row["stable_rank"] for row in rows], [row["true_count"] for row in rows]),
                    "stable_rank_pearson": _pearson([row["stable_rank"] for row in rows], [row["true_count"] for row in rows]),
                    "stable_rank_spearman": _spearman([row["stable_rank"] for row in rows], [row["true_count"] for row in rows]),
                }
            )
        fig.tight_layout()
        fig.savefig(args.out_root / f"all_species_affinity_rank_spectra_k{matrix_k}.png", bbox_inches="tight")
        fig.savefig(args.out_root / f"all_species_affinity_rank_spectra_k{matrix_k}.pdf", bbox_inches="tight")
        plt.close(fig)

    _write_csv(args.out_root / "affinity_rank_summary_by_matrix_k.csv", spectra_rows)
    _write_csv(args.out_root / "stable_rank_subset_correlation_by_matrix_k.csv", subset_summary)

    correlation_rows = []
    for matrix_k in args.matrix_ks:
        rows = [row for row in spectra_rows if row["matrix_k"] == matrix_k]
        birds = [row["birds"] for row in rows]
        stable = [row["stable_rank"] for row in rows]
        correlation_rows.append(
            {
                "matrix_k": matrix_k,
                "species_count": len(rows),
                "stable_rank_vs_birds_pearson": _pearson(stable, birds),
                "stable_rank_vs_birds_spearman": _spearman(stable, birds),
                "stable_rank_vs_birds_r2": _r2(stable, birds),
                "mean_subset_r2": float(np.mean([row["stable_rank_r2"] for row in subset_summary if row["matrix_k"] == matrix_k])),
                "median_subset_r2": float(np.median([row["stable_rank_r2"] for row in subset_summary if row["matrix_k"] == matrix_k])),
            }
        )
    _write_csv(args.out_root / "stable_rank_correlation_summary.csv", correlation_rows)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    for species in SPECIES:
        rows = [row for row in spectra_rows if row["species"] == species]
        ax.plot([row["matrix_k"] for row in rows], [row["stable_rank"] for row in rows], marker="o", label=species)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Affinity matrix K")
    ax.set_ylabel("Stable rank")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(args.out_root / "stable_rank_by_matrix_k.png", bbox_inches="tight")
    fig.savefig(args.out_root / "stable_rank_by_matrix_k.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=Path, default=ROOT / "results/individual_id_knn_graph_metrics/affinity_matrix_k_sweep_songmae_unbinned_usable_songs30")
    parser.add_argument("--matrix_ks", default="8,16,32,64")
    parser.add_argument("--songs_per_bird", type=int, default=30)
    parser.add_argument("--max_recordings", type=int, default=0)
    parser.add_argument("--max_points_per_recording", type=int, default=400)
    parser.add_argument("--max_total_points", type=int, default=50000)
    parser.add_argument("--feature_memmap_dir", type=Path, default=None)
    parser.add_argument("--postprocess_chunk_size", type=int, default=65536)
    parser.add_argument("--pca_fit_points", type=int, default=200000)
    parser.add_argument("--knn_chunk_size", type=int, default=128)
    parser.add_argument("--knn_candidate_chunk_size", type=int, default=0)
    parser.add_argument("--subset_repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_dir", default=str(ROOT / "runs/xcl_base_100k_p32x1_c010"))
    parser.add_argument("--checkpoint", default="model_step_499999.pth")
    parser.add_argument("--species", default=",".join(SPECIES))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.out_root = (ROOT / args.out_root).resolve() if not args.out_root.is_absolute() else args.out_root
    if args.feature_memmap_dir is not None:
        args.feature_memmap_dir = (ROOT / args.feature_memmap_dir).resolve() if not args.feature_memmap_dir.is_absolute() else args.feature_memmap_dir
    args.matrix_ks = km._parse_ints(args.matrix_ks)
    args.out_root.mkdir(parents=True, exist_ok=True)

    species_keys = [x for x in args.species.split(",") if x]
    for species in species_keys:
        assert species in SPECIES
        _run_species(args, species)
    _analyze(args)


if __name__ == "__main__":
    main()
