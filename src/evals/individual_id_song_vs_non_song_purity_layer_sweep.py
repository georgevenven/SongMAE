#!/usr/bin/env python3
"""Sweep held-out token-neighborhood identity purity across SongMAE layers."""
import csv
import gc
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

from src.embeddings.syllable_knn import prepare
from src.evals.individual_id_song_vs_non_song_layer_sweep import ANNOTATIONS, LAYERS, SCOPES, load
from src.evals.individual_id_song_vs_non_song_probe import PCA_COMPONENTS, SEED, split


OUTPUT = ROOT / "Individual_Id_paper_materials/token_analysis/individual_id_song_vs_non_song_pink_0db_purity_k50_layer_sweep_full_isolated"
K = 50
REFERENCE_PER_BIRD = 256
CHUNK = 2048


def reference_indices(y, train, labels):
    rng = np.random.default_rng(SEED)
    rows = []
    for label in range(len(labels)):
        available = train[y[train] == label]
        assert len(available) >= REFERENCE_PER_BIRD
        rows.extend(rng.choice(available, REFERENCE_PER_BIRD, replace=False))
    return np.asarray(rows)


def evaluate(scope, layer, data, train_stems, test_stems, labels):
    x, stems, y = data
    train = np.flatnonzero(np.isin(stems, list(train_stems)))
    test = np.flatnonzero(np.isin(stems, list(test_stems)))
    reference = reference_indices(y, train, labels)
    reference_x = x[reference] if layer is None else x[reference, layer]
    query_x = x[test] if layer is None else x[test, layer]
    reference_x, query_x = prepare(reference_x, query_x, PCA_COMPONENTS, SEED)
    reference_y = y[reference]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_x = torch.from_numpy(reference_x).to(device)
    purity = np.empty(len(test), dtype=np.float32)
    probabilities = np.empty((len(test), len(labels)), dtype=np.float32)

    for start in range(0, len(test), CHUNK):
        stop = min(start + CHUNK, len(test))
        query = torch.from_numpy(query_x[start:stop]).to(device)
        neighbors = (query @ reference_x.T).topk(K, dim=1).indices.cpu().numpy()
        neighbor_y = reference_y[neighbors]
        truth = y[test[start:stop]]
        purity[start:stop] = np.mean(neighbor_y == truth[:, None], axis=1)
        probabilities[start:stop] = np.stack([
            np.mean(neighbor_y == label, axis=1) for label in range(len(labels))
        ], axis=1)

    token_predictions = probabilities.argmax(axis=1)
    true, predicted = [], []
    for stem in sorted(test_stems):
        positions = np.flatnonzero(stems[test] == stem)
        true.append(int(y[test[positions[0]]]))
        predicted.append(int(probabilities[positions].mean(axis=0).argmax()))
    per_bird = [float(purity[y[test] == label].mean()) for label in range(len(labels))]
    return {
        "scope": scope, "layer": layer, "k": K,
        "reference_tokens": len(reference), "query_tokens": len(test),
        "micro_same_identity_purity": float(purity.mean()),
        "macro_same_identity_purity": float(np.mean(per_bird)),
        "macro_identity_enrichment": float((np.mean(per_bird) - 1 / len(labels)) / (1 - 1 / len(labels))),
        "token_majority_accuracy": float(np.mean(token_predictions == y[test])),
        "recording_accuracy": float(np.mean(np.asarray(predicted) == np.asarray(true))),
        "correct_recordings": int(np.sum(np.asarray(predicted) == np.asarray(true))),
        "per_bird_same_identity_purity": dict(zip(labels, per_bird)),
        "device": str(device),
    }


def plot(results):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), dpi=200)
    for scope, label, color in (("song", "Song events only", "#0072B2"), ("non_song", "Non-song only", "#D55E00")):
        rows = sorted((row for row in results if row["scope"] == scope), key=lambda row: row["layer"])
        layers = [row["layer"] for row in rows]
        axes[0].plot(layers, np.asarray([row["macro_same_identity_purity"] for row in rows]) * 100, "o-", color=color, linewidth=2, label=label)
        axes[1].plot(layers, np.asarray([row["recording_accuracy"] for row in rows]) * 100, "o-", color=color, linewidth=2, label=label)
    axes[0].axhline(100 / 36, color="#555555", linestyle="--", linewidth=1, label="Chance")
    for axis, ylabel in zip(axes, (f"Macro neighborhood purity, k={K} (%)", "Recording accuracy (%)")):
        axis.set(xlabel="Encoder layer", ylabel=ylabel, xticks=list(LAYERS))
        axis.grid(alpha=0.18)
    axes[0].set_ylim(0, 25)
    axes[1].set_ylim(0, 105)
    axes[0].legend(frameon=False)
    fig.suptitle("Zebra-finch identity neighborhoods under 0 dB pink noise", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT / "song_vs_non_song_purity_k50_layer_sweep.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "song_vs_non_song_purity_k50_layer_sweep.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    recordings = {Path(row["recording"]["filename"]).stem: row for row in json.loads(ANNOTATIONS.read_text())["recordings"]}
    labels = sorted({str(row["recording"]["bird_id"]) for row in recordings.values()})
    train_stems, test_stems = split(recordings)
    data = {scope: load(scope, recordings, labels) for scope in SCOPES}
    results = []
    for layer in LAYERS:
        for scope in SCOPES:
            path = OUTPUT / scope / f"layer_{layer:02d}" / "metrics.json"
            if path.exists():
                result = json.loads(path.read_text())
            else:
                result = evaluate(scope, layer, data[scope], train_stems, test_stems, labels)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=2) + "\n")
            results.append(result)
            print(f"{scope} layer {layer}: purity={result['macro_same_identity_purity'] * 100:.1f}%", flush=True)
            gc.collect()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "task": "held_out_token_neighborhood_identity_purity_layer_sweep",
        "species": "zebra_finch", "condition": "pink_0db", "model": "xcl_large_500k_p32x4_c010",
        "layers": list(LAYERS), "k": K, "classes": len(labels),
        "reference_tokens_per_bird": REFERENCE_PER_BIRD,
        "train_recordings": len(train_stems), "test_recordings": len(test_stems),
        "split": "5 complete held-out source recordings per bird",
        "input_isolation": "song events and non-song intervals encoded separately",
        "preprocessing": "training-bank feature z-score, PCA768, row-L2 normalization",
        "neighbor_bank": "256 training tokens sampled per bird",
        "results": results,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUTPUT / "layer_sweep.tsv").open("w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(("scope", "layer", "macro_same_identity_purity", "micro_same_identity_purity", "macro_identity_enrichment", "token_majority_accuracy", "recording_accuracy", "correct_recordings"))
        writer.writerows((row["scope"], row["layer"], row["macro_same_identity_purity"], row["micro_same_identity_purity"], row["macro_identity_enrichment"], row["token_majority_accuracy"], row["recording_accuracy"], row["correct_recordings"]) for row in results)
    plot(results)


if __name__ == "__main__":
    main()
