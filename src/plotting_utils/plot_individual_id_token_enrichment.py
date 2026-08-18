#!/usr/bin/env python3
"""Measure held-out SongMAE token-neighborhood identity enrichment."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.embedding_store import EmbeddingStore
from src.embeddings.syllable_knn import prepare
from src.evals.individual_id_classification import load_annotations


EMBEDDINGS = Path("/media/george-vengrovski/disk2/individual_id_all_layers_10_species/embeddings/zebra_finch")
ANNOTATIONS = ROOT / "results/individual_id/individual_id_linear_probe/multispecies_background_robustness/zebra_finch/clean_annotations.json"
CLIP_MAP = ANNOTATIONS.parent / "clip_map.json"
OUTPUT = ROOT / "Individual_Id_paper_materials/token_analysis/neighborhood_enrichment"
MODELS = ("xcl_large_500k_p32x4_c010", "xcl_large_500k_p32x1_c005")
LAYER = 11
PCA_COMPONENTS = 768
REFERENCE_PER_INDIVIDUAL = 256
KS = (1, 5, 10, 50, 100)
PRIMARY_K = 50
SHUFFLES = 20
CHUNK = 1024


def load(model):
    path = EMBEDDINGS / model / "clean"
    store = EmbeddingStore(path)
    x = store["encoded_embeddings"][:, LAYER]
    stems = np.asarray(store["recording_stem"]).astype(str)
    starts = np.asarray(store["token_start_ms"], dtype=np.float32)
    ends = np.asarray(store["token_end_ms"], dtype=np.float32)
    annotations = load_annotations(ANNOTATIONS)
    y = np.asarray([annotations[stem]["bird"] for stem in stems])
    centers = (starts + ends) / 2
    kinds = np.asarray([
        "song" if any(onset <= center < offset for onset, offset in annotations[stem]["events"]) else "non_song"
        for stem, center in zip(stems, centers)
    ])
    clips = {
        row["composite_stem"]: (row["source_stem"], int(row["fold"]))
        for row in json.loads(CLIP_MAP.read_text())
        if row["condition"] == "clean"
    }
    sources = np.asarray([clips[stem][0] for stem in stems])
    folds = np.asarray([clips[stem][1] for stem in stems], dtype=np.int8)
    assert all(not set(sources[folds == fold]) & set(sources[folds != fold]) for fold in range(3))
    return x, y, stems, starts, ends, kinds, sources, folds


def balanced_reference(y, candidates, labels, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for label in labels:
        available = candidates[y[candidates] == label]
        assert len(available) >= REFERENCE_PER_INDIVIDUAL
        rows.extend(rng.choice(available, REFERENCE_PER_INDIVIDUAL, replace=False).tolist())
    return np.asarray(rows, dtype=np.int64)


def enrich(model):
    x, y, stems, starts, ends, kinds, sources, folds = load(model)
    labels = sorted(set(y.tolist()))
    label_index = {label: index for index, label in enumerate(labels)}
    encoded_y = np.asarray([label_index[label] for label in y], dtype=np.int16)
    enrichments = {k: np.empty(len(y), dtype=np.float32) for k in KS}
    shuffled = np.empty(len(y), dtype=np.float32)
    probabilities = np.empty((len(y), len(labels)), dtype=np.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chance = 1 / len(labels)

    for fold in range(3):
        query_indices = np.flatnonzero(folds == fold)
        candidates = np.flatnonzero(folds != fold)
        reference_indices = balanced_reference(y, candidates, labels, 42 + fold)
        assert not set(sources[query_indices]) & set(sources[reference_indices])
        reference, query = prepare(x[reference_indices], x[query_indices], PCA_COMPONENTS, 42 + fold)
        reference_y = encoded_y[reference_indices]
        rng = np.random.default_rng(142 + fold)
        shuffled_y = np.stack([rng.permutation(reference_y) for _ in range(SHUFFLES)])
        reference = torch.from_numpy(reference).to(device)

        for start in range(0, len(query), CHUNK):
            stop = min(start + CHUNK, len(query))
            rows = query_indices[start:stop]
            points = torch.from_numpy(query[start:stop]).to(device)
            neighbors = (points @ reference.T).topk(max(KS), dim=1).indices.cpu().numpy()
            neighbor_y = reference_y[neighbors]
            same = neighbor_y == encoded_y[rows, None]
            cumulative = same.cumsum(axis=1)
            for k in KS:
                enrichments[k][rows] = (cumulative[:, k - 1] / k - chance) / (1 - chance)
            counts = np.stack([(neighbor_y[:, :PRIMARY_K] == index).sum(axis=1) for index in range(len(labels))], axis=1)
            probabilities[rows] = counts / PRIMARY_K
            shuffled_same = shuffled_y[:, neighbors[:, :PRIMARY_K]] == encoded_y[rows][None, :, None]
            shuffled[rows] = ((shuffled_same.mean(axis=2) - chance) / (1 - chance)).mean(axis=0)
    return y, stems, starts, ends, kinds, folds, enrichments, shuffled, probabilities, labels, device


def recording_accuracy(y, recordings, enrichment, probabilities, fraction, seed=None):
    rng = np.random.default_rng(seed)
    true, predicted = [], []
    for indices in recordings:
        count = min(round(len(indices) * fraction), len(indices) - 1)
        if count:
            removed = indices[np.argsort(-enrichment[indices])[:count]] if seed is None else rng.choice(indices, count, replace=False)
            indices = np.setdiff1d(indices, removed, assume_unique=True)
        true.append(y[indices[0]])
        predicted.append(probabilities[indices].mean(axis=0).argmax())
    true = np.asarray(true)
    predicted = np.asarray(predicted)
    labels = sorted(set(y.tolist()))
    encoded = np.asarray([labels.index(label) for label in true])
    per_bird = [np.mean(predicted[true == label] == encoded[true == label]) for label in labels]
    return float(np.mean(predicted == encoded)), float(np.mean(per_bird))


def write(model, result):
    y, stems, starts, ends, kinds, folds, enrichments, shuffled, probabilities, labels, device = result
    output = OUTPUT / model
    output.mkdir(parents=True, exist_ok=True)
    with (output / "token_enrichment.tsv").open("w", newline="") as file:
        fields = ("fold", "bird", "kind", "recording_stem", "start_ms", "end_ms", *[f"enrichment_k{k}" for k in KS], "shuffled_enrichment_k50")
        writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index in range(len(y)):
            writer.writerow({
                "fold": int(folds[index]), "bird": y[index], "kind": kinds[index], "recording_stem": stems[index],
                "start_ms": f"{starts[index]:.3f}", "end_ms": f"{ends[index]:.3f}",
                **{f"enrichment_k{k}": f"{enrichments[k][index]:.6f}" for k in KS},
                "shuffled_enrichment_k50": f"{shuffled[index]:.6f}",
            })

    ablation = {}
    grouped = defaultdict(list)
    for index, stem in enumerate(stems):
        grouped[stem].append(index)
    recordings = [np.asarray(grouped[stem]) for stem in sorted(grouped)]
    for fraction in (0, 0.1, 0.25):
        top = recording_accuracy(y, recordings, enrichments[PRIMARY_K], probabilities, fraction)
        random = [recording_accuracy(y, recordings, enrichments[PRIMARY_K], probabilities, fraction, 1000 + repeat) for repeat in range(20)]
        ablation[str(fraction)] = {
            "remove_high_enrichment": {"micro_accuracy": top[0], "macro_accuracy": top[1]},
            "remove_random_mean": {"micro_accuracy": float(np.mean(random, axis=0)[0]), "macro_accuracy": float(np.mean(random, axis=0)[1])},
            "remove_random_std": {"micro_accuracy": float(np.std(random, axis=0)[0]), "macro_accuracy": float(np.std(random, axis=0)[1])},
        }
    summary = {
        "model": model, "species": "zebra_finch", "condition": "clean", "layer": LAYER,
        "pca_components": PCA_COMPONENTS, "l2_normalized": True, "held_out_unit": "source_recording",
        "reference_tokens_per_individual": REFERENCE_PER_INDIVIDUAL, "classes": len(labels), "device": str(device),
        "tokens": len(y), "mean_enrichment": {str(k): float(enrichments[k].mean()) for k in KS},
        "mean_shuffled_enrichment_k50": float(shuffled.mean()),
        "by_kind": {
            kind: {"tokens": int(np.sum(kinds == kind)), "mean_enrichment_k50": float(enrichments[PRIMARY_K][kinds == kind].mean())}
            for kind in ("song", "non_song")
        },
        "recording_ablation": ablation,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(output, enrichments, shuffled, kinds)
    return summary


def plot(output, enrichments, shuffled, kinds):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), dpi=200)
    axes[0].plot(KS, [enrichments[k].mean() for k in KS], "o-", color="#0072B2", linewidth=2)
    axes[0].set_xscale("log")
    axes[0].set_xticks(KS, [str(k) for k in KS])
    axes[0].set_xlabel("Neighborhood size k")
    axes[0].set_ylabel("Mean identity enrichment")
    bins = np.linspace(-0.25, 0.75, 80)
    axes[1].hist(enrichments[PRIMARY_K][kinds == "song"], bins=bins, density=True, histtype="step", linewidth=2, label="Song")
    axes[1].hist(shuffled, bins=bins, density=True, histtype="step", linewidth=2, label="Shuffled labels")
    axes[1].axvline(0, color="#202020", linewidth=1)
    axes[1].set_xlabel(f"Identity enrichment (k={PRIMARY_K})")
    axes[1].set_ylabel("Density")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output / "token_enrichment.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output / "token_enrichment.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_summary(summaries):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), dpi=200)
    colors = ("#0072B2", "#56B4E9")
    labels = ("SongMAE 32 × 4", "SongMAE 32 × 1")
    for summary, label, color in zip(summaries, labels, colors):
        axes[0].plot(KS, [summary["mean_enrichment"][str(k)] for k in KS], "o-", color=color, linewidth=2, label=label)
        fractions = (0, 0.1, 0.25)
        top = [summary["recording_ablation"][str(fraction)]["remove_high_enrichment"]["macro_accuracy"] for fraction in fractions]
        random = [summary["recording_ablation"][str(fraction)]["remove_random_mean"]["macro_accuracy"] for fraction in fractions]
        axes[1].plot(np.asarray(fractions) * 100, np.asarray(top) * 100, "o-", color=color, linewidth=2, label=f"{label} · high E")
        axes[1].plot(np.asarray(fractions) * 100, np.asarray(random) * 100, "o--", color=color, linewidth=1.5, alpha=0.65, label=f"{label} · random")
    axes[0].set_xscale("log")
    axes[0].set_xticks(KS, [str(k) for k in KS])
    axes[0].set_xlabel("Neighborhood size k")
    axes[0].set_ylabel("Mean identity enrichment")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xticks((0, 10, 25))
    axes[1].set_xlabel("Query tokens removed (%)")
    axes[1].set_ylabel("Recording macro accuracy (%)")
    axes[1].legend(frameon=False, fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(OUTPUT / "neighborhood_enrichment_summary.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "neighborhood_enrichment_summary.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model in MODELS:
        path = OUTPUT / model / "summary.json"
        summary = json.loads(path.read_text()) if path.exists() else write(model, enrich(model))
        summary["held_out_unit"] = "source_recording"
        path.write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
    (OUTPUT / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    plot_summary(summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
