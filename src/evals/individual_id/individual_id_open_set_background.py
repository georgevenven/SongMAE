#!/usr/bin/env python3
"""Open-set individual identification under randomized natural backgrounds."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.evals.individual_id_open_set import (
    METHODS,
    evaluate,
    evenly_spaced,
    load_recordings,
    prepare_support,
    project_queries,
    score_queries,
    split_episode,
    summarize,
)

CONDITIONS = ("clean", "train_aug", "test_p10", "test_0", "test_m10")
QUERY_CONDITIONS = ("clean", "test_p10", "test_0", "test_m10")
SUPPORTS = {
    "clean_only": "clean",
    "clean_plus_random_background": "train_aug",
}


def augmented_support(clean, augmented, max_points):
    clean_points = max_points // 2
    return {
        source: np.concatenate([
            evenly_spaced(clean[source], clean_points),
            evenly_spaced(augmented[source], max_points - clean_points),
        ])
        for source in clean.keys() & augmented.keys()
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding_root", required=True)
    parser.add_argument("--clip_map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--shots", default="1,2,4,8,16")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pca_components", type=int, default=32)
    parser.add_argument("--pca_fit_points_per_recording", type=int, default=32)
    parser.add_argument("--max_points_per_recording", type=int, default=64)
    parser.add_argument("--queries_per_identity", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.embedding_root)
    loaded = {
        condition: load_recordings(
            root / condition,
            args.clip_map,
            condition,
            args.max_points_per_recording,
        )
        for condition in CONDITIONS
    }
    recordings = {condition: value[0] for condition, value in loaded.items()}
    clean_by_bird = loaded["clean"][1]
    common = set.intersection(*(set(value) for value in recordings.values()))
    by_bird = {
        bird: [source for source in sources if source in common]
        for bird, sources in clean_by_bird.items()
    }
    by_bird = {bird: sources for bird, sources in by_bird.items() if sources}
    support_recordings = {
        "clean_only": recordings["clean"],
        "clean_plus_random_background": augmented_support(
            recordings["clean"], recordings["train_aug"], args.max_points_per_recording
        ),
    }

    rows = []
    unsupported = []
    shots = [int(value) for value in args.shots.split(",")]
    for k in shots:
        completed = 0
        for repeat in range(args.repeats):
            seed = args.seed + 1000 * k + repeat
            split = split_episode(
                by_bird, k, args.queries_per_identity, np.random.default_rng(seed)
            )
            if split is None:
                break
            for support, calibration_condition in SUPPORTS.items():
                prepared = prepare_support(
                    support_recordings[support],
                    split,
                    args.pca_components,
                    args.pca_fit_points_per_recording,
                    seed,
                )
                pca, models, calibration_known, test_known, calibration_unknown, test_unknown = prepared
                calibration_queries = calibration_known + calibration_unknown
                projected = project_queries(
                    pca, recordings[calibration_condition], calibration_queries
                )
                calibration_known_scores = score_queries(projected, models, calibration_known)
                calibration_unknown_scores = score_queries(projected, models, calibration_unknown)
                for condition in QUERY_CONDITIONS:
                    queries = test_known + test_unknown
                    projected = project_queries(pca, recordings[condition], queries)
                    test_known_scores = score_queries(projected, models, test_known)
                    test_unknown_scores = score_queries(projected, models, test_unknown)
                    for method in METHODS:
                        rows.append({
                            "support": support,
                            "query_condition": condition,
                            "shots": k,
                            "repeat": repeat,
                            "method": method,
                            "known_individuals": len(models),
                            "calibration_unknown_individuals": len(calibration_unknown),
                            "test_unknown_individuals": len(test_unknown),
                            **evaluate(
                                calibration_known_scores[method],
                                calibration_unknown_scores[method],
                                test_known_scores[method],
                                test_unknown_scores[method],
                            ),
                        })
            completed += 1
        if completed == 0:
            unsupported.append(k)

    summaries = []
    for support in SUPPORTS:
        for condition in QUERY_CONDITIONS:
            selected = [
                row for row in rows
                if row["support"] == support and row["query_condition"] == condition
            ]
            summaries.extend({
                "support": support,
                "query_condition": condition,
                **summary,
            } for k in shots for summary in summarize(selected, k))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "species": args.species,
        "model": args.model,
        "individuals": len(by_bird),
        "recordings": len(common),
        "excluded_short_recordings": {
            condition: value[2] for condition, value in loaded.items()
        },
        "shots": shots,
        "unsupported_shots": unsupported,
        "repeats": args.repeats,
        "support": "k source recordings; augmented support contains equal clean and train-augmentation points within the same fixed point budget",
        "calibration": "clean-only uses clean calibration; augmented support uses train-augmentation calibration; thresholds fixed across query conditions",
        "test_backgrounds": "test donor recordings disjoint from train-augmentation donors",
        "pca_fit_scope": "enrollment_recordings_only",
        "pca_components": args.pca_components,
        "max_points_per_recording": args.max_points_per_recording,
        "queries_per_identity": args.queries_per_identity,
        "known_unknown_split": "identity disjoint; unknown calibration and test identities disjoint",
        "summaries": summaries,
        "episodes": rows,
    }, indent=2) + "\n")
    with output.with_suffix(".tsv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "species": args.species,
        "model": args.model,
        "output": str(output),
        "episodes": len(rows) // len(METHODS),
        "unsupported_shots": unsupported,
    }))


if __name__ == "__main__":
    main()
