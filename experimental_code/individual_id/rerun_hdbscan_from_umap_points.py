#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from pathlib import Path

import hdbscan
import numpy as np
from sklearn.metrics import completeness_score, homogeneity_score, v_measure_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experimental_code.individual_id.run_individual_id_umap import _hdbscan_recording_rows, _label_metric  # noqa: E402


def _parse_ints(raw):
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    assert values, raw
    return values


def _load_species(points_path):
    summary_path = points_path.parent / "summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        return data.get("species_key") or data.get("species") or points_path.parent.name
    return points_path.parent.name


def _fit_hdbscan(xy, min_cluster_size, min_samples):
    effective_cluster_size = int(min_cluster_size)
    if effective_cluster_size <= 0:
        effective_cluster_size = max(25, int(round(xy.shape[0] * 0.005)))
    effective_min_samples = None if int(min_samples) <= 0 else int(min_samples)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=effective_cluster_size,
        min_samples=effective_min_samples,
    )
    return clusterer.fit_predict(xy), effective_cluster_size, effective_min_samples


def _write_recording_csv(path, rows):
    fieldnames = [
        "bird_id",
        "points",
        "recordings",
        "clusters",
        "noise_fraction",
        "recording_homogeneity",
        "recording_completeness",
        "recording_v_measure",
        "median_clusters_per_recording",
        "median_recordings_per_cluster",
        "interpretation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cluster_summary(xy, bird_labels, recording_labels, clusters):
    non_noise = clusters >= 0
    cluster_ids = sorted(set(clusters[non_noise].tolist()))
    rows = _hdbscan_recording_rows(bird_labels, recording_labels, clusters)
    counts = {}
    for row in rows:
        label = row["interpretation"]
        counts[label] = counts.get(label, 0) + 1

    return {
        "points": int(xy.shape[0]),
        "clusters": int(len(cluster_ids)),
        "noise_fraction": float(np.mean(~non_noise)) if clusters.size else 0.0,
        "bird_homogeneity": _label_metric(bird_labels[non_noise], clusters[non_noise], homogeneity_score),
        "bird_completeness": _label_metric(bird_labels[non_noise], clusters[non_noise], completeness_score),
        "bird_v_measure": _label_metric(bird_labels[non_noise], clusters[non_noise], v_measure_score),
        "recording_summary": rows,
        "interpretation_counts": counts,
    }


def _save_run(out_dir, rep_name, data, clusters, summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{rep_name}_hdbscan_points.npz",
        xy=data["xy"].astype(np.float32, copy=False),
        bird_labels=data["bird_labels"].astype(object, copy=False),
        syllable_labels=data["syllable_labels"].astype(np.int64, copy=False),
        recording_labels=data["recording_labels"].astype(object, copy=False),
        hdbscan_clusters=clusters.astype(np.int64, copy=False),
    )
    _write_recording_csv(out_dir / f"{rep_name}_hdbscan_recording_summary.csv", summary["recording_summary"])
    (out_dir / f"{rep_name}_hdbscan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Refit HDBSCAN on saved individual-ID UMAP coordinates.")
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--min_cluster_sizes", required=True)
    parser.add_argument("--min_samples", default="10")
    parser.add_argument("--out_name", default="hdbscan_param_sweep")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    out_root = result_dir / args.out_name
    points_paths = sorted(path for path in result_dir.glob("*/*_hdbscan_points.npz") if out_root not in path.parents)
    assert points_paths, result_dir

    aggregate_rows = []
    for points_path in points_paths:
        data = np.load(points_path, allow_pickle=True)
        rep_name = points_path.name.removesuffix("_hdbscan_points.npz")
        species = _load_species(points_path)
        for min_cluster_size in _parse_ints(args.min_cluster_sizes):
            for min_samples in _parse_ints(args.min_samples):
                clusters, effective_cluster_size, effective_min_samples = _fit_hdbscan(
                    data["xy"],
                    min_cluster_size,
                    min_samples,
                )
                summary = _cluster_summary(
                    data["xy"],
                    data["bird_labels"],
                    data["recording_labels"],
                    clusters,
                )
                summary["min_cluster_size"] = effective_cluster_size
                summary["min_cluster_size_arg"] = min_cluster_size
                summary["min_samples"] = effective_min_samples
                summary["min_samples_arg"] = min_samples
                combo = f"mcs{min_cluster_size}_ms{min_samples}"
                out_dir = out_root / combo / points_path.parent.name
                _save_run(out_dir, rep_name, data, clusters, summary)
                aggregate_rows.append(
                    {
                        "species": species,
                        "source_dir": points_path.parent.name,
                        "combo": combo,
                        "min_cluster_size_arg": min_cluster_size,
                        "min_cluster_size": effective_cluster_size,
                        "min_samples_arg": min_samples,
                        "min_samples": "" if effective_min_samples is None else effective_min_samples,
                        "points": summary["points"],
                        "clusters": summary["clusters"],
                        "noise_fraction": summary["noise_fraction"],
                        "bird_homogeneity": summary["bird_homogeneity"],
                        "bird_completeness": summary["bird_completeness"],
                        "bird_v_measure": summary["bird_v_measure"],
                        "single_cluster": summary["interpretation_counts"].get("single_cluster", 0),
                        "recording_fracture_risk": summary["interpretation_counts"].get("recording_fracture_risk", 0),
                        "shared_multi_part_structure": summary["interpretation_counts"].get("shared_multi_part_structure", 0),
                        "mixed": summary["interpretation_counts"].get("mixed", 0),
                    }
                )
                print(f"[hdbscan] {species} {combo}: clusters={summary['clusters']} noise={summary['noise_fraction']:.3f}")

    fieldnames = list(aggregate_rows[0])
    with (out_root / "summary.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(aggregate_rows)
    print(f"[hdbscan] wrote {out_root / 'summary.tsv'}")


if __name__ == "__main__":
    main()
