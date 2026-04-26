#!/usr/bin/env python3

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "individual_id" / "evaluate_knn_graph_metrics.py"

SPECIES = {
    "zf": {
        "species": "Zebra_Finch",
        "annotation_json": ROOT / "files" / "zf_annotations.json",
        "spec_dir": "/media/george-vengrovski/disk2/specs/zf_64hop_32khz",
        "songs_per_bird": 30,
        "recording_mode": "full_recordings",
    },
    "bf": {
        "species": "bf",
        "annotation_json": ROOT / "files" / "bf_annotations.json",
        "spec_dir": "/media/george-vengrovski/disk2/specs/bf_64hop_32khz",
        "songs_per_bird": 30,
        "recording_mode": "full_recordings",
    },
    "canary": {
        "species": "canary",
        "annotation_json": ROOT / "files" / "canary_annotations_for_individual_id.json",
        "spec_dir": "/media/george-vengrovski/disk2/specs/canary_individual_identification_64hop_32khz",
        "songs_per_bird": 30,
        "recording_mode": "full_recordings",
    },
    "ovenbird": {
        "species": "ovenbird",
        "annotation_json": ROOT / "files" / "lapp_ovenbird.json",
        "spec_dir": "/media/george-vengrovski/disk2/specs/ovenbird_lapp_sample_64hop_32khz",
        "songs_per_bird": 0,
        "recording_mode": "events",
    },
    "chiffchaff": {
        "species": "chiffchaff",
        "annotation_json": ROOT / "files" / "chiffchaff_annotations.json",
        "spec_dir": "/media/george-vengrovski/disk2/specs/chiffchaff_64hop_32khz",
        "songs_per_bird": 0,
        "recording_mode": "full_recordings",
    },
    "european_starling": {
        "species": "european_starling",
        "annotation_json": ROOT / "files" / "european_starling_annotations_unprefixed.json",
        "spec_dir": "/media/george-vengrovski/disk2/specs/european_starling_64hop_32khz",
        "songs_per_bird": 30,
        "recording_mode": "full_recordings",
    },
    "tree_pipit": {
        "species": "tree_pipit",
        "annotation_json": ROOT / "files" / "tree_pipit_annotations.json",
        "spec_dir": "/media/george-vengrovski/disk2/specs/tree_pipit_64hop_32khz",
        "songs_per_bird": 0,
        "recording_mode": "full_recordings",
    },
    "little_owl": {
        "species": "little_owl",
        "annotation_json": ROOT / "files" / "little_owl_annotations.json",
        "spec_dir": "/media/george-vengrovski/disk2/specs/little_owl_64hop_32khz",
        "songs_per_bird": 0,
        "recording_mode": "full_recordings",
    },
}


def _species_keys(text):
    if text == "all":
        return list(SPECIES)
    keys = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [x for x in keys if x not in SPECIES]
    assert not unknown, f"unknown species keys: {unknown}"
    return keys


def _profile(profile):
    if profile == "safe":
        return {
            "feature_postprocess_dim": "64",
            "max_points_per_recording": "80",
            "points_per_individual": "50",
            "repeats": "1",
            "knn_chunk_size": "256",
            "songs_per_bird_cap": "30",
        }
    assert profile == "paper"
    return {
        "feature_postprocess_dim": "1024",
        "max_points_per_recording": "200",
        "points_per_individual": "100",
        "repeats": "2",
        "knn_chunk_size": "512",
        "songs_per_bird_cap": "30",
    }


def _command(args, key):
    config = SPECIES[key]
    profile = _profile(args.profile)
    out_dir = Path(args.out_root).resolve() / key
    songs_per_bird = config["songs_per_bird"] or int(profile["songs_per_bird_cap"])
    return [
        sys.executable,
        str(EVAL_SCRIPT),
        "--metric",
        args.metric,
        "--species",
        config["species"],
        "--annotation_json",
        str(config["annotation_json"]),
        "--spec_dir",
        config["spec_dir"],
        "--run_dir",
        args.run_dir,
        "--checkpoint",
        args.checkpoint,
        "--recording_mode",
        config["recording_mode"],
        "--songs_per_bird",
        str(songs_per_bird),
        "--max_birds",
        str(args.max_birds),
        "--max_points_per_recording",
        profile["max_points_per_recording"],
        "--k_values",
        args.k_values,
        "--heatmap_k",
        str(args.heatmap_k),
        "--points_per_individual",
        profile["points_per_individual"],
        "--counts",
        args.counts,
        "--repeats",
        profile["repeats"],
        "--graph_k",
        str(args.graph_k),
        "--num_eigenvalues",
        str(args.num_eigenvalues),
        "--heat_scales",
        args.heat_scales,
        "--knn_chunk_size",
        profile["knn_chunk_size"],
        "--feature_postprocess",
        "pca_whiten_l2",
        "--feature_postprocess_dim",
        profile["feature_postprocess_dim"],
        "--out_dir",
        str(out_dir),
    ]


def _run(args):
    for key in _species_keys(args.species):
        out_dir = Path(args.out_root).resolve() / key
        done = out_dir / "knn_purity_summary.json"
        if args.skip_existing and done.exists():
            print(f"[all-knn-graph] skip existing {key}: {out_dir}")
            continue

        command = _command(args, key)
        print("[all-knn-graph] " + " ".join(command))
        if args.dry_run:
            continue
        subprocess.run(command, check=not args.continue_on_error)


def _collate(args):
    rows = []
    for key in _species_keys(args.species):
        path = Path(args.out_root).resolve() / key / "knn_purity_summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        for k, value in zip(summary["k_values"], summary["purity"]):
            rows.append(
                {
                    "species_key": key,
                    "species": summary["species"],
                    "k": int(k),
                    "purity": float(value),
                    "chance": float(summary["chance"]),
                    "purity_minus_chance": float(value - summary["chance"]),
                }
            )

    if not rows:
        return

    out_root = Path(args.out_root).resolve()
    with (out_root / "all_species_knn_purity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)
    for key in _species_keys(args.species):
        species_rows = [row for row in rows if row["species_key"] == key]
        if not species_rows:
            continue
        x = np.asarray([row["k"] for row in species_rows])
        y = np.asarray([row["purity"] for row in species_rows])
        ax.plot(x, y, marker="o", linewidth=1.4, label=key)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("k nearest neighbors")
    ax.set_ylabel("Same-individual fraction")
    ax.set_title("Frame kNN same-individual purity")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_root / "all_species_knn_purity.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_root / "all_species_knn_purity.pdf", bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run kNN graph metrics across species sequentially.")
    parser.add_argument("--species", default="all")
    parser.add_argument("--metric", default="all", choices=["all", "purity", "heat_trace"])
    parser.add_argument("--profile", default="safe", choices=["safe", "paper"])
    parser.add_argument("--run_dir", default="/media/george-vengrovski/Desk SSD/LAMBDA_TRAIN_RUNS/runs/xcl_voronoi_mask_no_normalize_32h_10w_5s_fp8")
    parser.add_argument("--checkpoint", default="model_step_499999.pth")
    parser.add_argument("--out_root", default=str(ROOT / "results" / "individual_id_knn_graph_metrics" / "all_species_safe"))
    parser.add_argument("--max_birds", type=int, default=30)
    parser.add_argument("--k_values", default="1,2,5,10,20,50,100")
    parser.add_argument("--heatmap_k", type=int, default=50)
    parser.add_argument("--counts", default="2,3,4,6,8,12,16,20,23,30")
    parser.add_argument("--graph_k", type=int, default=50)
    parser.add_argument("--num_eigenvalues", type=int, default=80)
    parser.add_argument("--heat_scales", default="1,2,5,10,20,50")
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--collate_only", action="store_true")
    args = parser.parse_args()

    Path(args.out_root).resolve().mkdir(parents=True, exist_ok=True)
    if not args.collate_only:
        _run(args)
    _collate(args)


if __name__ == "__main__":
    main()
