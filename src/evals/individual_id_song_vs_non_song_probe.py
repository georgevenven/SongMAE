#!/usr/bin/env python3
"""Compare song-only and non-song-only token probes for zebra-finch identity."""
import csv
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


EMBEDDINGS = Path("/media/george-vengrovski/disk2/zebra_finch_pink_noise_event_probe_full/embeddings_songmae_32x4_isolated")
ANNOTATIONS = ROOT / "results/individual_id/zebra_finch_pink_noise_event_probe_full/annotations.json"
OUTPUT = ROOT / "Individual_Id_paper_materials/token_analysis/individual_id_song_vs_non_song_pink_0db_full_isolated"
PCA_COMPONENTS = 768
LOGREG_C = 1e-3
SEED = 42
TEST_RECORDINGS_PER_BIRD = 5


def split(recordings):
    by_bird = {}
    for stem, row in recordings.items():
        by_bird.setdefault(str(row["recording"]["bird_id"]), []).append(stem)
    rng = np.random.default_rng(SEED)
    test = {
        stem
        for stems in by_bird.values()
        for stem in rng.choice(sorted(stems), TEST_RECORDINGS_PER_BIRD, replace=False)
    }
    train = set(recordings) - test
    assert len(by_bird) == 36 and len(test) == 36 * TEST_RECORDINGS_PER_BIRD and train.isdisjoint(test)
    assert all(any(stem in train for stem in stems) and any(stem in test for stem in stems) for stems in by_bird.values())
    return train, test


def evaluate(scope, path, recordings, train_stems, test_stems, labels):
    store = EmbeddingStore(path)
    x = np.asarray(store["encoded_embeddings"], dtype=np.float32)
    stems = np.asarray(store["recording_stem"]).astype(str)
    label_index = {label: index for index, label in enumerate(labels)}
    y = np.asarray([label_index[str(recordings[stem]["recording"]["bird_id"])] for stem in stems])
    train = np.flatnonzero(np.isin(stems, list(train_stems)))
    test = np.flatnonzero(np.isin(stems, list(test_stems)))
    assert set(y[train]) == set(y[test]) == set(range(len(labels)))
    train_x, test_x = prepare_features(x[train], x[test], PCA_COMPONENTS, SEED)
    probe = LogisticRegression(C=LOGREG_C, class_weight="balanced", max_iter=5000)
    probe.fit(train_x, y[train])
    token_probabilities = probe.predict_proba(test_x)
    token_predictions = token_probabilities.argmax(axis=1)

    recording_true, recording_predicted, recording_rows = [], [], []
    for stem in sorted(test_stems):
        positions = np.flatnonzero(stems[test] == stem)
        truth = int(y[test[positions[0]]])
        predicted = int(token_probabilities[positions].mean(axis=0).argmax())
        recording_true.append(truth)
        recording_predicted.append(predicted)
        recording_rows.append((scope, stem, labels[truth], labels[predicted], truth == predicted, len(positions)))
    recording_true = np.asarray(recording_true)
    recording_predicted = np.asarray(recording_predicted)
    matrix = confusion_matrix(recording_true, recording_predicted, labels=np.arange(len(labels)))
    return {
        "scope": scope,
        "train_tokens": int(len(train)), "test_tokens": int(len(test)),
        "train_recordings": len(train_stems), "test_recordings": len(test_stems),
        "token_accuracy": float(np.mean(token_predictions == y[test])),
        "recording_accuracy": float(np.mean(recording_predicted == recording_true)),
        "recording_macro_f1": float(f1_score(recording_true, recording_predicted, labels=np.arange(len(labels)), average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "correct_recordings": int(np.trace(matrix)),
        "per_bird_accuracy": {
            label: float(matrix[index, index] / matrix[index].sum())
            for index, label in enumerate(labels)
        },
    }, recording_rows


def main():
    recordings = {
        Path(row["recording"]["filename"]).stem: row
        for row in json.loads(ANNOTATIONS.read_text())["recordings"]
    }
    train_stems, test_stems = split(recordings)
    labels = sorted({str(row["recording"]["bird_id"]) for row in recordings.values()})

    results, rows = [], []
    for scope, directory in (("song", "events"), ("non_song", "background")):
        result, recording_rows = evaluate(scope, EMBEDDINGS / directory / "pink_0db", recordings, train_stems, test_stems, labels)
        results.append(result)
        rows.extend(recording_rows)
    summary = {
        "task": "closed_set_individual_id_song_vs_non_song",
        "species": "zebra_finch", "condition": "pink_0db", "snr_db": 0,
        "model": "xcl_large_500k_p32x4_c010", "layer": 11,
        "probe": "per_token_multinomial_logistic_regression",
        "recording_aggregation": "mean_token_probability",
        "input_isolation": "song events and non-song intervals encoded separately",
        "split": f"{TEST_RECORDINGS_PER_BIRD} complete held-out source recordings per bird",
        "classes": len(labels), "train_recordings": len(train_stems), "test_recordings": len(test_stems),
        "recordings_per_bird": {
            "minimum": min(sum(str(row["recording"]["bird_id"]) == label for row in recordings.values()) for label in labels),
            "mean": len(recordings) / len(labels),
            "maximum": max(sum(str(row["recording"]["bird_id"]) == label for row in recordings.values()) for label in labels),
        },
        "pca_components": PCA_COMPONENTS, "standardization_fit_scope": "training_tokens", "logreg_c": LOGREG_C,
        "results": results,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT / "split.json").write_text(json.dumps({
        "seed": SEED,
        "train_recordings": sorted(train_stems),
        "test_recordings": sorted(test_stems),
    }, indent=2) + "\n")
    with (OUTPUT / "test_recording_predictions.tsv").open("w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(("scope", "recording_stem", "true_bird", "predicted_bird", "correct", "tokens"))
        writer.writerows(rows)
    plot(results)
    print(json.dumps(summary, indent=2))


def plot(results):
    names = ("Song events only", "Non-song only")
    accuracy = np.asarray([row["recording_accuracy"] for row in results]) * 100
    macro_f1 = np.asarray([row["recording_macro_f1"] for row in results]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), dpi=200)
    x = np.arange(2)
    width = 0.34
    axes[0].bar(x - width / 2, accuracy, width, color="#0072B2", label="Accuracy")
    axes[0].bar(x + width / 2, macro_f1, width, color="#56B4E9", label="Macro-F1")
    axes[0].set_xticks(x, names)
    axes[0].set(ylabel="Recording-level score (%)", ylim=(0, 105))
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.18)
    matrices = [np.asarray(row["confusion_matrix"]) for row in results]
    axes[1].bar(0, np.trace(matrices[0]), color="blue", width=0.65)
    axes[1].bar(1, np.trace(matrices[1]), color="orange", width=0.65)
    axes[1].set_xticks(x, names)
    axes[1].set(ylabel="Correctly identified recordings", ylim=(0, 36 * TEST_RECORDINGS_PER_BIRD + 5))
    axes[1].grid(axis="y", alpha=0.18)
    fig.suptitle("Zebra-finch identity under 0 dB pink noise", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT / "song_vs_non_song_identity.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "song_vs_non_song_identity.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
