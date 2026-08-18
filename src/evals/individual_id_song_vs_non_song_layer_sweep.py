#!/usr/bin/env python3
"""Sweep isolated song and non-song identity probes across SongMAE layers."""
import csv
import gc
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.embedding_store import EmbeddingStore
from src.evals.individual_id_classification import prepare_features
from src.evals.individual_id_song_vs_non_song_probe import LOGREG_C, PCA_COMPONENTS, SEED, split


EMBEDDINGS = Path("/media/george-vengrovski/disk2/zebra_finch_pink_noise_event_probe_full/embeddings_songmae_32x4_all_layers_isolated")
ANNOTATIONS = ROOT / "results/individual_id/zebra_finch_pink_noise_event_probe_full/annotations.json"
OUTPUT = ROOT / "Individual_Id_paper_materials/token_analysis/individual_id_song_vs_non_song_pink_0db_layer_sweep_full_isolated"
SCOPES = {"song": "events", "non_song": "background"}
LAYERS = range(12)


def load(scope, recordings, labels):
    store = EmbeddingStore(EMBEDDINGS / SCOPES[scope] / "pink_0db")
    x = store["encoded_embeddings"]
    stems = np.asarray(store["recording_stem"]).astype(str)
    label_index = {label: index for index, label in enumerate(labels)}
    y = np.asarray([label_index[str(recordings[stem]["recording"]["bird_id"])] for stem in stems])
    assert x.shape[1] == len(LAYERS)
    return x, stems, y


def evaluate(scope, layer, data, train_stems, test_stems, labels):
    x, stems, y = data
    train = np.flatnonzero(np.isin(stems, list(train_stems)))
    test = np.flatnonzero(np.isin(stems, list(test_stems)))
    train_x, test_x = prepare_features(x[train, layer], x[test, layer], PCA_COMPONENTS, SEED)
    probe = LogisticRegression(C=LOGREG_C, class_weight="balanced", max_iter=5000)
    probe.fit(train_x, y[train])
    probabilities = probe.predict_proba(test_x)
    token_predictions = probabilities.argmax(axis=1)
    true, predicted = [], []
    for stem in sorted(test_stems):
        positions = np.flatnonzero(stems[test] == stem)
        true.append(int(y[test[positions[0]]]))
        predicted.append(int(probabilities[positions].mean(axis=0).argmax()))
    true, predicted = np.asarray(true), np.asarray(predicted)
    matrix = confusion_matrix(true, predicted, labels=np.arange(len(labels)))
    return {
        "scope": scope, "layer": layer,
        "train_tokens": int(len(train)), "test_tokens": int(len(test)),
        "token_accuracy": float(np.mean(token_predictions == y[test])),
        "recording_accuracy": float(np.mean(predicted == true)),
        "recording_macro_f1": float(f1_score(true, predicted, labels=np.arange(len(labels)), average="macro", zero_division=0)),
        "correct_recordings": int(np.trace(matrix)),
        "confusion_matrix": matrix.tolist(),
    }


def plot(results):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), dpi=200)
    for scope, label, color in (("song", "Song events only", "#0072B2"), ("non_song", "Non-song only", "#D55E00")):
        rows = sorted((row for row in results if row["scope"] == scope), key=lambda row: row["layer"])
        layers = [row["layer"] for row in rows]
        axes[0].plot(layers, np.asarray([row["recording_accuracy"] for row in rows]) * 100, "o-", color=color, linewidth=2, label=label)
        axes[1].plot(layers, np.asarray([row["token_accuracy"] for row in rows]) * 100, "o-", color=color, linewidth=2, label=label)
    for axis, ylabel in zip(axes, ("Recording accuracy (%)", "Token accuracy (%)")):
        axis.set(xlabel="Encoder layer", ylabel=ylabel, xticks=list(LAYERS), ylim=(0, 105))
        axis.grid(alpha=0.18)
    axes[0].legend(frameon=False)
    fig.suptitle("Where zebra-finch identity emerges under 0 dB pink noise", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT / "song_vs_non_song_layer_sweep.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "song_vs_non_song_layer_sweep.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    recordings = {
        Path(row["recording"]["filename"]).stem: row
        for row in json.loads(ANNOTATIONS.read_text())["recordings"]
    }
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
            print(f"{scope} layer {layer}: {result['recording_accuracy'] * 100:.1f}%", flush=True)
            gc.collect()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "task": "closed_set_individual_id_song_vs_non_song_layer_sweep",
        "species": "zebra_finch", "condition": "pink_0db", "snr_db": 0,
        "model": "xcl_large_500k_p32x4_c010", "layers": list(LAYERS),
        "probe": "per_token_multinomial_logistic_regression", "recording_aggregation": "mean_token_probability",
        "input_isolation": "song events and non-song intervals encoded separately",
        "classes": len(labels), "train_recordings": len(train_stems), "test_recordings": len(test_stems),
        "split": "5 complete held-out source recordings per bird",
        "pca_components": PCA_COMPONENTS, "standardization_fit_scope": "training_tokens", "logreg_c": LOGREG_C,
        "results": results,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUTPUT / "layer_sweep.tsv").open("w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(("scope", "layer", "recording_accuracy", "recording_macro_f1", "token_accuracy", "correct_recordings", "train_tokens", "test_tokens"))
        writer.writerows((row["scope"], row["layer"], row["recording_accuracy"], row["recording_macro_f1"], row["token_accuracy"], row["correct_recordings"], row["train_tokens"], row["test_tokens"]) for row in results)
    plot(results)


if __name__ == "__main__":
    main()
