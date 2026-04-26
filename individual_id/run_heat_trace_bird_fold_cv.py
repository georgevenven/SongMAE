#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "individual_id"))

import evaluate_knn_graph_metrics as metrics  # noqa: E402
import plot_recording_embedding_similarity as recording_similarity  # noqa: E402
from run_all_knn_graph_metrics import SPECIES  # noqa: E402


def _split_birds(seed, num_birds):
    rng = np.random.default_rng(seed)
    birds = rng.permutation(num_birds)
    return [np.sort(x) for x in np.array_split(birds, 3)]


def _subset(args, sampled, bird_pool, count, repeat, fold, split_name):
    rng = np.random.default_rng(args.seed + 10007 * fold + 1009 * count + 31 * repeat + len(split_name))
    chosen = np.sort(rng.choice(bird_pool, size=count, replace=False))
    indices = []
    for bird in chosen:
        bird_indices = np.flatnonzero(sampled["point_birds"] == bird)
        assert bird_indices.size >= args.points_per_individual
        indices.append(rng.choice(bird_indices, size=args.points_per_individual, replace=False))
    indices = np.sort(np.concatenate(indices))
    return {
        "features": sampled["features"][indices],
        "point_birds": sampled["point_birds"][indices],
        "point_recordings": sampled["point_recordings"][indices],
        "bird_ids": sampled["bird_ids"][chosen],
    }


def _heat_row(args, sampled, bird_pool, count, repeat, fold, split_name, scales):
    subset = _subset(args, sampled, bird_pool, count, repeat, fold, split_name)
    eigenvalues, components, device, graph_k = metrics._laplacian_eigenvalues(args, subset)
    row = {
        "fold": int(fold),
        "split": split_name,
        "individuals": int(count),
        "repeat": int(repeat),
        "points": int(subset["features"].shape[0]),
        "connected_components": int(components),
        "device": device,
        "graph_k": int(graph_k),
    }
    for scale in scales:
        row[metrics._heat_key(scale)] = float(np.exp(-scale * eigenvalues).sum())
    return row


def _rows_for_pool(args, sampled, bird_pool, fold, split_name, scales):
    counts = [x for x in metrics._parse_ints(args.counts) if x <= len(bird_pool)]
    if len(counts) < 2:
        return []
    rows = []
    for count in counts:
        for repeat in range(args.repeats):
            rows.append(_heat_row(args, sampled, bird_pool, count, repeat, fold, split_name, scales))
            print(
                "[bird-fold-cv] "
                f"fold={fold} split={split_name} count={count} repeat={repeat} "
                f"{metrics._heat_key(scales[0])}={rows[-1][metrics._heat_key(scales[0])]:.3f}"
            )
    return rows


def _fit(train_rows, scales):
    y = np.asarray([row["individuals"] for row in train_rows], dtype=np.float32)
    fits = {}
    for scale in scales:
        key = metrics._heat_key(scale)
        x = np.asarray([row[key] for row in train_rows], dtype=np.float32)
        fits[key] = metrics._fit_line(x, y)
    best_key = max(fits, key=lambda key: fits[key][3])
    return best_key, fits[best_key]


def _run_species(args, species_key):
    config = SPECIES[species_key]
    out_dir = Path(args.out_root).resolve() / species_key
    out_dir.mkdir(parents=True, exist_ok=True)

    species_args = argparse.Namespace(**vars(args))
    species_args.species = config["species"]
    species_args.annotation_json = str(config["annotation_json"])
    species_args.spec_dir = config["spec_dir"]
    species_args.songs_per_bird = config["songs_per_bird"] or args.songs_per_bird_cap
    species_args.out_dir = str(out_dir)
    rows, feature_postprocess = metrics._load_table(species_args)
    sampled = metrics._sample_frames(species_args, rows)
    scales = metrics._parse_floats(args.heat_scales)

    output_rows = []
    folds = _split_birds(args.seed, sampled["bird_ids"].size)
    for fold_index, train_birds in enumerate(folds):
        test_birds = np.sort(np.setdiff1d(np.arange(sampled["bird_ids"].size), train_birds))
        if len(train_birds) < 2 or len(test_birds) < 2:
            continue
        train_rows = _rows_for_pool(species_args, sampled, train_birds, fold_index, "train", scales)
        test_rows = _rows_for_pool(species_args, sampled, test_birds, fold_index, "test", scales)
        if not train_rows or not test_rows:
            continue
        best_key, (slope, intercept, _, train_r2) = _fit(train_rows, scales)

        for row in train_rows:
            row["best_heat_feature"] = best_key
            row["predicted_individuals"] = float(slope * row[best_key] + intercept)
            row["train_r2"] = float(train_r2)
            output_rows.append(row)
        for row in test_rows:
            row["best_heat_feature"] = best_key
            row["predicted_individuals"] = float(slope * row[best_key] + intercept)
            row["train_r2"] = float(train_r2)
            output_rows.append(row)

    metrics._write_json(out_dir / "feature_postprocess.json", recording_similarity._feature_postprocess_summary(feature_postprocess))
    _write_outputs(out_dir, species_key, output_rows, args)


def _score(rows, split):
    subset = [row for row in rows if row["split"] == split]
    y = np.asarray([row["individuals"] for row in subset], dtype=np.float32)
    pred = np.asarray([row["predicted_individuals"] for row in subset], dtype=np.float32)
    r2 = 1.0 - float(np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    mae = float(np.mean(np.abs(y - pred)))
    return r2, mae


def _write_outputs(out_dir, species_key, rows, args):
    assert rows
    with (out_dir / "bird_fold_heat_trace_cv.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    test_r2, test_mae = _score(rows, "test")
    train_r2, train_mae = _score(rows, "train")
    summary = {
        "species_key": species_key,
        "train_fraction": "one_fold_of_three",
        "test_fraction": "two_folds_of_three",
        "points_per_individual": int(args.points_per_individual),
        "repeats": int(args.repeats),
        "graph_k": int(args.graph_k),
        "num_eigenvalues": int(args.num_eigenvalues),
        "heat_scales": metrics._parse_floats(args.heat_scales),
        "train_r2": train_r2,
        "train_mae": train_mae,
        "test_r2": test_r2,
        "test_mae": test_mae,
    }
    metrics._write_json(out_dir / "bird_fold_heat_trace_cv_summary.json", summary)
    _plot(out_dir / "bird_fold_heat_trace_cv", rows, f"{species_key} bird-fold heat-trace CV", test_r2, test_mae)


def _plot(out_prefix, rows, title, r2, mae):
    fig, ax = plt.subplots(figsize=(5.8, 5.4), dpi=300)
    colors = {"train": "0.7", "test": "tab:blue"}
    for split in ["train", "test"]:
        subset = [row for row in rows if row["split"] == split]
        ax.scatter(
            [float(row["individuals"]) for row in subset],
            [float(row["predicted_individuals"]) for row in subset],
            s=34,
            alpha=0.75,
            color=colors[split],
            label=split,
        )
    max_value = max(max(float(row["individuals"]), float(row["predicted_individuals"])) for row in rows)
    lims = (0.0, max_value + 1.5)
    ax.plot(lims, lims, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True individuals in subset")
    ax.set_ylabel("Predicted individuals")
    ax.set_title(title)
    ax.text(0.04, 0.96, f"test R^2={r2:.2f}\ntest MAE={mae:.2f}", transform=ax.transAxes, va="top")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def _collate(args):
    rows = []
    for species_key in _species_keys(args.species):
        path = Path(args.out_root).resolve() / species_key / "bird_fold_heat_trace_cv.csv"
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["species_key"] = species_key
                rows.append(row)
    if not rows:
        return

    out_root = Path(args.out_root).resolve()
    with (out_root / "all_species_bird_fold_heat_trace_cv.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    test_rows = [row for row in rows if row["split"] == "test"]
    y = np.asarray([float(row["individuals"]) for row in test_rows], dtype=np.float32)
    pred = np.asarray([float(row["predicted_individuals"]) for row in test_rows], dtype=np.float32)
    r2 = 1.0 - float(np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    mae = float(np.mean(np.abs(y - pred)))
    metrics._write_json(out_root / "all_species_bird_fold_heat_trace_cv_summary.json", {"test_r2": r2, "test_mae": mae})
    _plot(out_root / "all_species_bird_fold_heat_trace_cv", rows, "All-species bird-fold heat-trace CV", r2, mae)


def _species_keys(text):
    if text == "all":
        return list(SPECIES)
    keys = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [x for x in keys if x not in SPECIES]
    assert not unknown
    return keys


def main():
    parser = argparse.ArgumentParser(description="Train heat-trace count calibration on 1/3 birds and test on 2/3.")
    parser.add_argument("--species", default="all")
    parser.add_argument("--run_dir", default="/media/george-vengrovski/Desk SSD/LAMBDA_TRAIN_RUNS/runs/xcl_voronoi_mask_no_normalize_32h_10w_5s_fp8")
    parser.add_argument("--checkpoint", default="model_step_499999.pth")
    parser.add_argument("--out_root", default=str(ROOT / "results" / "individual_id_knn_graph_metrics" / "bird_fold_heat_trace_cv_safe"))
    parser.add_argument("--songs_per_bird_cap", type=int, default=30)
    parser.add_argument("--max_birds", type=int, default=30)
    parser.add_argument("--max_points_per_recording", type=int, default=80)
    parser.add_argument("--points_per_individual", type=int, default=50)
    parser.add_argument("--counts", default="2,3,4,6,8,12,16,20")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--graph_k", type=int, default=50)
    parser.add_argument("--num_eigenvalues", type=int, default=80)
    parser.add_argument("--heat_scales", default="1,2,5,10,20,50")
    parser.add_argument("--knn_chunk_size", type=int, default=256)
    parser.add_argument("--exclude_same_recording", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--recording_mode", default="full_recordings", choices=["events", "full_recordings"])
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
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--collate_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path(args.out_root).resolve().mkdir(parents=True, exist_ok=True)
    if not args.collate_only:
        for species_key in _species_keys(args.species):
            done = Path(args.out_root).resolve() / species_key / "bird_fold_heat_trace_cv_summary.json"
            if args.skip_existing and done.exists():
                print(f"[bird-fold-cv] skip existing {species_key}")
                continue
            try:
                _run_species(args, species_key)
            except Exception:
                if not args.continue_on_error:
                    raise
                print(f"[bird-fold-cv] failed {species_key}", file=sys.stderr)
    _collate(args)


if __name__ == "__main__":
    main()
