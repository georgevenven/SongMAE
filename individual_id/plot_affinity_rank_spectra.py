#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_rows(summary_path):
    with summary_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _matrix_for_svd(path):
    data = np.load(path, allow_pickle=True)
    matrix = data["recording_matrix"].astype(np.float64, copy=False)
    matrix = (matrix + matrix.T) * 0.5
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _rank_at_energy(singular_values, fraction):
    energy = singular_values**2
    cumulative = np.cumsum(energy) / max(float(energy.sum()), 1e-12)
    return int(np.searchsorted(cumulative, fraction) + 1)


def _spectrum_stats(species, display_name, npz_path, singular_values):
    energy = singular_values**2
    stable_rank = float(energy.sum() / max(float(singular_values[0] ** 2), 1e-12))
    tol = max(len(singular_values), 1) * np.finfo(np.float64).eps * float(singular_values[0])
    return {
        "species": species,
        "display_name": display_name,
        "recordings": len(singular_values),
        "stable_rank": stable_rank,
        "rank_90_energy": _rank_at_energy(singular_values, 0.90),
        "rank_95_energy": _rank_at_energy(singular_values, 0.95),
        "rank_99_energy": _rank_at_energy(singular_values, 0.99),
        "numerical_rank": int(np.sum(singular_values > tol)),
        "top1_energy_fraction": float(energy[0] / max(float(energy.sum()), 1e-12)),
        "svd_npz": str(npz_path.relative_to(ROOT)),
    }


def _plot_species(ax, species, singular_values, stats, svd_dim, linear_y):
    ranks = np.arange(1, len(singular_values) + 1)
    normalized = singular_values / max(float(singular_values[0]), 1e-12)
    ax.plot(ranks, normalized, color="#2f6fbb", linewidth=1.5)
    ax.axvline(svd_dim, color="#202020", linestyle="-", linewidth=1.0, label="SVD dim 15")
    ax.axvline(stats["stable_rank"], color="#d95f02", linestyle="--", linewidth=1.0, label="stable rank")
    ax.axvline(stats["rank_95_energy"], color="#1b9e77", linestyle=":", linewidth=1.2, label="95% energy")
    ax.set_xscale("log")
    if not linear_y:
        ax.set_yscale("log")
    ax.set_title(species, fontsize=12, fontweight="bold")
    ax.text(
        0.04,
        0.05,
        f"sr={stats['stable_rank']:.1f}\n95%={stats['rank_95_energy']}\n99%={stats['rank_99_energy']}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    if linear_y:
        ax.set_ylim(0.0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="results/individual_id_umap/default_configs_recsvd15_stats_nn50_cosine_hdbscan_all8/hdbscan_effective_summary.tsv",
    )
    parser.add_argument(
        "--out_dir",
        default="results/individual_id_knn_graph_metrics/default_config_affinity_rank_spectra",
    )
    parser.add_argument("--svd_dim", type=int, default=15)
    parser.add_argument("--linear_y", action="store_true")
    args = parser.parse_args()

    summary_path = ROOT / args.summary
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(summary_path)
    stats_rows = []
    spectra = []
    for row in rows:
        npz_path = ROOT / row["recording_svd_npz"]
        matrix = _matrix_for_svd(npz_path)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        stats = _spectrum_stats(row["species"], row["display_name"], npz_path, singular_values)
        stats_rows.append(stats)
        spectra.append((row["display_name"], singular_values, stats))

    csv_path = out_dir / "affinity_rank_summary.csv"
    _write_csv(csv_path, stats_rows)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=220)
    for ax, (display_name, singular_values, stats) in zip(axes.flat, spectra):
        _plot_species(ax, display_name, singular_values, stats, args.svd_dim, args.linear_y)
    for ax in axes[:, 0]:
        ax.set_ylabel("Singular value / top singular value")
    for ax in axes[-1, :]:
        ax.set_xlabel("Rank")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    suffix = "_linear_y" if args.linear_y else ""
    collage_path = out_dir / f"all_species_affinity_rank_spectra{suffix}.png"
    fig.savefig(collage_path, bbox_inches="tight")
    fig.savefig(out_dir / f"all_species_affinity_rank_spectra{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(csv_path.relative_to(ROOT))
    print(collage_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
