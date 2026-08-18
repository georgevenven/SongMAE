#!/usr/bin/env python3
"""Fit a recording-disjoint song versus non-song token probe for one bird."""
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.embedding_store import EmbeddingStore
from src.evals.individual_id_classification import prepare_features


BIRD = "B145"
EMBEDDINGS = Path("/media/george-vengrovski/disk2/zebra_finch_pink_noise_event_probe/embeddings_songmae_32x4_b145/pink_0db")
SPECS = Path("/media/george-vengrovski/disk2/zebra_finch_pink_noise_event_probe/specs/pink_0db")
ANNOTATIONS = ROOT / "results/individual_id/zebra_finch_pink_noise_event_probe/annotations.json"
MANIFEST = ROOT / "results/individual_id/zebra_finch_pink_noise_event_probe/manifest.json"
OUTPUT = ROOT / "Individual_Id_paper_materials/token_analysis/single_bird_song_vs_non_song_pink_0db"
PCA_COMPONENTS = 128
LOGREG_C = 1e-3


def main():
    annotations = {
        Path(row["recording"]["filename"]).stem: row
        for row in json.loads(ANNOTATIONS.read_text())["recordings"]
        if str(row["recording"]["bird_id"]) == BIRD
    }
    manifest = json.loads(MANIFEST.read_text())
    split = {row["stem"]: row["split"] for row in manifest["recordings"] if row["bird_id"] == BIRD}
    train_stems = sorted(stem for stem, value in split.items() if value == "train")
    test_stems = sorted(stem for stem, value in split.items() if value == "test")
    assert len(train_stems) == 2 and len(test_stems) == 1 and set(train_stems).isdisjoint(test_stems)

    store = EmbeddingStore(EMBEDDINGS)
    x = np.asarray(store["encoded_embeddings"], dtype=np.float32)
    stems = np.asarray(store["recording_stem"]).astype(str)
    starts = np.asarray(store["token_start_ms"], dtype=np.float32)
    ends = np.asarray(store["token_end_ms"], dtype=np.float32)
    centers = (starts + ends) / 2
    y = np.asarray([
        any(event["onset_ms"] <= center < event["offset_ms"] for event in annotations[stem]["detected_events"])
        for stem, center in zip(stems, centers)
    ], dtype=np.int8)
    train = np.flatnonzero(np.isin(stems, train_stems))
    test = np.flatnonzero(np.isin(stems, test_stems))
    assert set(stems[train]).isdisjoint(set(stems[test])) and set(y[train]) == set(y[test]) == {0, 1}
    train_x, test_x = prepare_features(x[train], x[test], PCA_COMPONENTS, 42)
    probe = LogisticRegression(C=LOGREG_C, class_weight="balanced", max_iter=5000)
    probe.fit(train_x, y[train])
    probability = probe.predict_proba(test_x)[:, 1]
    predicted = probability >= 0.5
    matrix = confusion_matrix(y[test], predicted, labels=(0, 1))
    precision, recall, f1, support = precision_recall_fscore_support(y[test], predicted, labels=(0, 1), zero_division=0)

    summary = {
        "task": "binary_outer_song_event_vs_non_song_token_classification",
        "bird": BIRD, "condition": "pink_0db", "snr_db": 0,
        "model": "xcl_large_500k_p32x4_c010", "layer": 11,
        "split": "two complete source recordings train; one complete source recording test",
        "train_recordings": train_stems, "test_recordings": test_stems,
        "train_tokens": int(len(train)), "test_tokens": int(len(test)),
        "train_song_fraction": float(y[train].mean()), "test_song_fraction": float(y[test].mean()),
        "pca_components": PCA_COMPONENTS, "standardization_fit_scope": "training_tokens", "logreg_c": LOGREG_C,
        "accuracy": float(np.mean(predicted == y[test])),
        "balanced_accuracy": float(balanced_accuracy_score(y[test], predicted)),
        "macro_f1": float(f1_score(y[test], predicted, average="macro")),
        "roc_auc": float(roc_auc_score(y[test], probability)),
        "average_precision": float(average_precision_score(y[test], probability)),
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            name: {"precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]), "tokens": int(support[index])}
            for index, name in enumerate(("non_song", "song"))
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUTPUT / "test_token_predictions.tsv").open("w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(("recording_stem", "start_ms", "end_ms", "true_label", "predicted_label", "song_probability"))
        writer.writerows((stems[index], starts[index], ends[index], "song" if y[index] else "non_song", "song" if pred else "non_song", score) for index, pred, score in zip(test, predicted, probability))
    plot(stems[test][0], centers[test], y[test], probability)
    print(json.dumps(summary, indent=2))


def plot(stem, centers, truth, probability):
    spec = np.load(SPECS / f"{stem}.npy")
    duration = len(spec) * 0.005
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), dpi=200, sharex=True, gridspec_kw={"height_ratios": (2.8, 1)})
    axes[0].imshow(spec.T, origin="lower", aspect="auto", extent=(0, duration, 0, 128), cmap="magma", vmin=-65, vmax=-15)
    axes[0].set(ylabel="Mel bin", title=f"Held-out B145 recording · 0 dB pink noise")
    seconds = centers / 1000
    axes[1].plot(seconds, probability, color="#0072B2", linewidth=1.5, label="Predicted song probability")
    axes[1].fill_between(seconds, 0, 1, where=truth.astype(bool), color="#E69F00", alpha=0.16, step="mid", label="Annotated song event")
    axes[1].axhline(0.5, color="#202020", linewidth=1, linestyle="--")
    axes[1].set(xlabel="Time (s)", ylabel="Song probability", ylim=(0, 1), xlim=(0, duration))
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(OUTPUT / "held_out_recording_predictions.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "held_out_recording_predictions.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
