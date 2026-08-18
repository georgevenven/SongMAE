#!/usr/bin/env python3
"""Find SongMAE tokens with the largest recording-disjoint DN4 identity margin."""
import csv
import json
import sys
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
from src.evals.individual_id_classification import load_annotations, load_embeddings


EMBEDDINGS = Path("/media/george-vengrovski/disk2/individual_id_all_layers_10_species/embeddings/zebra_finch")
SPECS = Path("/media/george-vengrovski/disk2/specs/individual_id_multispecies_background_robustness_5ms/zebra_finch/clean")
ANNOTATIONS = ROOT / "results/individual_id/individual_id_linear_probe/multispecies_background_robustness/zebra_finch/clean_annotations.json"
MANIFEST = ROOT / "Individual_Id_paper_materials/results/manifests/zebra_finch.json"
OUTPUT = ROOT / "Individual_Id_paper_materials/token_analysis"
MODELS = ("xcl_large_500k_p32x4_c010", "xcl_large_500k_p32x1_c005")
LAYER = 11
PCA_COMPONENTS = 768
CHUNK = 512


def score_tokens(model):
    path = EMBEDDINGS / model / "clean"
    annotations = load_annotations(ANNOTATIONS)
    data = load_embeddings(path, annotations, "song_and_non_song", LAYER)
    store = EmbeddingStore(path)
    starts = np.asarray(store["token_start_ms"])
    ends = np.asarray(store["token_end_ms"])
    assert len(starts) == len(data["x"])

    manifest = json.loads(MANIFEST.read_text())
    labels = sorted(set(data["y"].tolist()))
    assert manifest["class_labels"] == labels
    label_index = {label: index for index, label in enumerate(labels)}
    size = len(data["x"])
    fold_ids = np.full(size, -1, dtype=np.int8)
    true_scores = np.empty(size, dtype=np.float32)
    impostor_scores = np.empty(size, dtype=np.float32)
    predictions = np.empty(size, dtype=object)
    true_matches = np.empty(size, dtype=np.int64)
    impostor_matches = np.empty(size, dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold_index, fold in enumerate(manifest["folds"]):
        reference_indices = np.flatnonzero(np.isin(data["stems"], fold["train_recordings"]))
        query_indices = np.flatnonzero(np.isin(data["stems"], fold["val_recordings"]))
        reference, query = prepare(
            data["x"][reference_indices], data["x"][query_indices], PCA_COMPONENTS, 42 + fold_index
        )
        supports = []
        support_indices = []
        for label in labels:
            indices = np.flatnonzero(data["y"][reference_indices] == label)
            supports.append(torch.from_numpy(reference[indices]).to(device))
            support_indices.append(reference_indices[indices])

        for start in range(0, len(query), CHUNK):
            stop = min(start + CHUNK, len(query))
            points = torch.from_numpy(query[start:stop]).to(device)
            values, matches = [], []
            for support, indices in zip(supports, support_indices):
                value, match = (points @ support.T).max(dim=1)
                values.append(value.cpu().numpy())
                matches.append(indices[match.cpu().numpy()])
            values = np.stack(values, axis=1)
            matches = np.stack(matches, axis=1)
            rows = query_indices[start:stop]
            truth = np.asarray([label_index[label] for label in data["y"][rows]])
            impostor = values.copy()
            impostor[np.arange(len(rows)), truth] = -np.inf
            strongest = impostor.argmax(axis=1)
            predicted = values.argmax(axis=1)
            true_scores[rows] = values[np.arange(len(rows)), truth]
            impostor_scores[rows] = values[np.arange(len(rows)), strongest]
            predictions[rows] = np.asarray(labels)[predicted]
            true_matches[rows] = matches[np.arange(len(rows)), truth]
            impostor_matches[rows] = matches[np.arange(len(rows)), strongest]
            fold_ids[rows] = fold_index

    assert np.all(fold_ids >= 0)
    margins = true_scores - impostor_scores
    return data, starts, ends, fold_ids, true_scores, impostor_scores, margins, predictions, true_matches, impostor_matches, device


def write_results(model, scored):
    data, starts, ends, folds, true_scores, impostor_scores, margins, predictions, true_matches, impostor_matches, device = scored
    output = OUTPUT / model
    output.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-margins)
    fields = (
        "rank", "fold", "bird", "predicted_bird", "kind", "recording_stem", "start_ms", "end_ms",
        "true_score", "strongest_impostor_score", "identity_margin", "nearest_true_stem",
        "nearest_true_start_ms", "strongest_impostor_bird", "strongest_impostor_stem",
        "strongest_impostor_start_ms",
    )
    with (output / "token_identity_margins.tsv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for rank, index in enumerate(order, 1):
            true_match, impostor_match = true_matches[index], impostor_matches[index]
            writer.writerow({
                "rank": rank,
                "fold": int(folds[index]),
                "bird": data["y"][index],
                "predicted_bird": predictions[index],
                "kind": data["kinds"][index],
                "recording_stem": data["stems"][index],
                "start_ms": f"{starts[index]:.3f}",
                "end_ms": f"{ends[index]:.3f}",
                "true_score": f"{true_scores[index]:.6f}",
                "strongest_impostor_score": f"{impostor_scores[index]:.6f}",
                "identity_margin": f"{margins[index]:.6f}",
                "nearest_true_stem": data["stems"][true_match],
                "nearest_true_start_ms": f"{starts[true_match]:.3f}",
                "strongest_impostor_bird": data["y"][impostor_match],
                "strongest_impostor_stem": data["stems"][impostor_match],
                "strongest_impostor_start_ms": f"{starts[impostor_match]:.3f}",
            })

    correct = predictions == data["y"]
    birds = sorted(set(data["y"].tolist()))
    summary = {
        "model": model,
        "species": "zebra_finch",
        "condition": "clean",
        "layer": LAYER,
        "pca_components": PCA_COMPONENTS,
        "score": "nearest_true_similarity_minus_strongest_impostor_similarity",
        "device": str(device),
        "tokens": len(margins),
        "micro_token_accuracy": float(correct.mean()),
        "macro_token_accuracy": float(np.mean([correct[data["y"] == bird].mean() for bird in birds])),
        "mean_margin": float(margins.mean()),
        "macro_mean_margin": float(np.mean([margins[data["y"] == bird].mean() for bird in birds])),
        "median_margin": float(np.median(margins)),
        "by_kind": {},
    }
    for kind in ("song", "non_song"):
        selected = data["kinds"] == kind
        summary["by_kind"][kind] = {
            "tokens": int(selected.sum()),
            "micro_token_accuracy": float(correct[selected].mean()),
            "mean_margin": float(margins[selected].mean()),
            "median_margin": float(np.median(margins[selected])),
            "top_1_percent_share": float(np.mean(data["kinds"][order[:max(1, len(order) // 100)]] == kind)),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_distributions(output, data["kinds"], margins)
    plot_pairs(output, data, starts, ends, margins, true_matches)
    return summary


def plot_distributions(output, kinds, margins):
    fig, axis = plt.subplots(figsize=(6.0, 3.5), dpi=200)
    bins = np.linspace(np.percentile(margins, 1), np.percentile(margins, 99), 70)
    for kind, label, color in (("song", "Song", "#0072B2"), ("non_song", "Non-song", "#D55E00")):
        axis.hist(margins[kinds == kind], bins=bins, density=True, histtype="step", linewidth=2, label=label, color=color)
    axis.axvline(0, color="#202020", linewidth=1)
    axis.set_xlabel("k=1 DN4 identity margin")
    axis.set_ylabel("Density")
    axis.legend(frameon=False)
    axis.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output / "token_margin_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "token_margin_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pairs(output, data, starts, ends, margins, true_matches):
    selected = []
    for index in np.argsort(-margins):
        if data["y"][index] not in {data["y"][row] for row in selected}:
            selected.append(index)
        if len(selected) == 8:
            break
    fig, axes = plt.subplots(len(selected), 2, figsize=(8.2, 11.0), dpi=180)
    for row, index in enumerate(selected):
        match = true_matches[index]
        spectrogram(axes[row, 0], data["stems"][index], starts[index], ends[index])
        spectrogram(axes[row, 1], data["stems"][match], starts[match], ends[match])
        axes[row, 0].set_title(f"Query · {data['y'][index]} · {data['kinds'][index]} · margin {margins[index]:.3f}", fontsize=8)
        axes[row, 1].set_title(f"Nearest same-individual token · {data['stems'][match]}", fontsize=8)
    axes[-1, 0].set_xlabel("Time from token center (s)")
    axes[-1, 1].set_xlabel("Time from token center (s)")
    fig.supylabel("Mel bin")
    fig.tight_layout()
    fig.savefig(output / "most_informative_token_pairs.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "most_informative_token_pairs.pdf", bbox_inches="tight")
    plt.close(fig)


def spectrogram(axis, stem, start_ms, end_ms):
    spec = np.load(SPECS / f"{stem}.npy")
    center = (start_ms + end_ms) / 2
    first = max(0, round((center - 500) / 5))
    last = min(len(spec), round((center + 500) / 5))
    left = first * 5 - center
    right = last * 5 - center
    axis.imshow(spec[first:last].T, origin="lower", aspect="auto", extent=(left / 1000, right / 1000, 0, 128), cmap="magma", vmin=-80, vmax=0)
    axis.axvspan((start_ms - center) / 1000, (end_ms - center) / 1000, color="cyan", alpha=0.35)
    axis.set_xlim(-0.5, 0.5)


def main():
    summaries = [write_results(model, score_tokens(model)) for model in MODELS]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
