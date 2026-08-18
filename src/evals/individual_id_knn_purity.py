#!/usr/bin/env python3
"""Recording-disjoint individual-ID DN4 token-set purity across k values."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.embeddings.syllable_knn import ints, prepare
from src.evals.individual_id_classification import load_annotations, load_embeddings, load_manifest


def dn4(reference, reference_y, query, query_y, query_groups, labels, k_values, cpu):
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    reference = torch.from_numpy(reference).to(device)
    supports = [
        reference[torch.from_numpy(np.flatnonzero(reference_y == label)).to(device)]
        for label in labels
    ]
    max_k = max(k_values)
    assert min(len(support) for support in supports) >= max_k
    observations = {k: [] for k in k_values}

    for group in dict.fromkeys(query_groups.tolist()):
        indices = np.flatnonzero(query_groups == group)
        truth = query_y[indices]
        assert np.all(truth == truth[0])
        points = torch.from_numpy(query[indices]).to(device)
        local = torch.stack([
            (points @ support.T).topk(max_k, dim=1).values.cumsum(dim=1)
            for support in supports
        ], dim=1)
        truth_index = labels.index(truth[0])
        for k in k_values:
            token_scores = local[:, :, k - 1] / k
            purity = float((token_scores.argmax(dim=1) == truth_index).float().mean().item())
            prediction = int(token_scores.mean(dim=0).argmax().item())
            observations[k].append((truth[0], purity, prediction == truth_index, len(indices)))
    return observations, str(device)


def summarize(rows, labels, k, folds, reference_tokens):
    per_bird_purity = {
        label: float(np.mean([purity for truth, purity, _, _ in rows if truth == label]))
        for label in labels
    }
    per_bird_accuracy = {
        label: float(np.mean([correct for truth, _, correct, _ in rows if truth == label]))
        for label in labels
    }
    tokens = sum(count for _, _, _, count in rows)
    same = sum(purity * count for _, purity, _, count in rows) / tokens
    macro = float(np.mean(list(per_bird_purity.values())))
    accuracy = float(np.mean([correct for _, _, correct, _ in rows]))
    macro_accuracy = float(np.mean(list(per_bird_accuracy.values())))
    return {
        "k": k,
        "macro_same_purity": macro,
        "macro_different_purity": 1.0 - macro,
        "micro_same_purity": same,
        "micro_different_purity": 1.0 - same,
        "dn4_accuracy": accuracy,
        "dn4_macro_accuracy": macro_accuracy,
        "query_token_sets": len(rows),
        "query_tokens": tokens,
        "reference_tokens": reference_tokens,
        "classes": len(labels),
        "folds": folds,
        "per_bird_same_purity": per_bird_purity,
        "per_bird_dn4_accuracy": per_bird_accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--audio_scope", choices=("song", "song_and_non_song"), required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--k_values", default="1,5,10,50,100")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--manifest_in")
    parser.add_argument("--manifest_out")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--pca_components", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    data = load_embeddings(args.embeddings, annotations, args.audio_scope, args.layer)
    labels = sorted(set(data["y"].tolist()))
    recording_labels = {stem: annotations[stem]["bird"] for stem in sorted(set(data["stems"]))}
    manifest = load_manifest(args, recording_labels, labels)
    k_values = ints(args.k_values)
    all_observations = {k: [] for k in k_values}
    reference_tokens = 0
    device = None

    for fold_index, fold in enumerate(manifest["folds"]):
        reference_indices = np.flatnonzero(np.isin(data["stems"], fold["train_recordings"]))
        query_indices = np.flatnonzero(np.isin(data["stems"], fold["val_recordings"]))
        reference, query = prepare(
            data["x"][reference_indices],
            data["x"][query_indices],
            args.pca_components,
            args.seed + fold_index,
        )
        observations, device = dn4(
            reference,
            data["y"][reference_indices],
            query,
            data["y"][query_indices],
            data["groups"][query_indices],
            labels,
            k_values,
            args.cpu,
        )
        for k in k_values:
            all_observations[k].extend(observations[k])
        reference_tokens += len(reference_indices)

    rows = [summarize(all_observations[k], labels, k, args.folds, reference_tokens) for k in k_values]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "knn_purity.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = vars(args) | {
        "out_dir": str(args.out_dir),
        "task": "closed_set_individual_id_dn4_purity",
        "analysis_unit": "five_second_token_set",
        "neighbor_unit": "individual_reference_token_set",
        "reference_query_split": "recording_disjoint_cross_validation",
        "maximum_group_seconds": 5.0,
        "token_score": "mean_top_k_cosine_within_individual_support",
        "token_vote_purity": "fraction_of_query_tokens_voting_for_true_individual",
        "dn4_score": "mean_query_token_score_per_individual",
        "standardization": "reference_tokens_feature_zscore",
        "pca_fit_scope": "reference_tokens" if args.pca_components else "disabled",
        "pca_whiten": False,
        "row_normalization": "l2",
        "distance": "cosine",
        "device": device,
        "rows": rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
