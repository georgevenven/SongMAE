#!/usr/bin/env python3
"""Decompose a mean-pooled zebra-finch identity probe into token contributions."""
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.embedding_store import EmbeddingStore
from src.evals.individual_id_classification import load_annotations
from src.plotting_utils.plot_individual_id_syllable_enrichment import syllables_by_clip


MODEL = "xcl_large_500k_p32x4_c010"
EMBEDDINGS = Path("/media/george-vengrovski/disk2/individual_id_all_layers_10_species/embeddings/zebra_finch") / MODEL / "clean"
SPECS = Path("/media/george-vengrovski/disk2/specs/individual_id_multispecies_background_robustness_5ms/zebra_finch/clean")
ANNOTATIONS = ROOT / "results/individual_id/individual_id_linear_probe/multispecies_background_robustness/zebra_finch/clean_annotations.json"
CLIP_MAP = ANNOTATIONS.parent / "clip_map.json"
OUTPUT = ROOT / "Individual_Id_paper_materials/token_analysis/probe_decomposition_songmae_32x4"
LAYER = 11
PCA_COMPONENTS = 768
LOGREG_C = 1e-3
TOP_PER_BIRD = 20
MIN_SEPARATION_MS = 250


def pooled(features, labels, stems):
    rows, y, names = [], [], []
    for stem in sorted(set(stems.tolist())):
        indices = np.flatnonzero(stems == stem)
        assert len(set(labels[indices])) == 1
        rows.append(features[indices].mean(axis=0))
        y.append(labels[indices[0]])
        names.append(stem)
    return np.asarray(rows), np.asarray(y), np.asarray(names)


def load():
    store = EmbeddingStore(EMBEDDINGS)
    x = store["encoded_embeddings"][:, LAYER]
    stems = np.asarray(store["recording_stem"]).astype(str)
    starts = np.asarray(store["token_start_ms"], dtype=np.float32)
    ends = np.asarray(store["token_end_ms"], dtype=np.float32)
    annotations = load_annotations(ANNOTATIONS)
    birds = np.asarray([annotations[stem]["bird"] for stem in stems])
    clip_rows = [row for row in json.loads(CLIP_MAP.read_text()) if row["condition"] == "clean"]
    clips = {row["composite_stem"]: row for row in clip_rows}
    folds = np.asarray([int(clips[stem]["fold"]) for stem in stems], dtype=np.int8)
    sources = np.asarray([clips[stem]["source_stem"] for stem in stems])
    for fold in range(3):
        assert not set(sources[folds == fold]) & set(sources[folds != fold])
    return x, birds, stems, starts, ends, folds, sources


def decompose():
    x, birds, stems, starts, ends, folds, sources = load()
    labels = sorted(set(birds.tolist()))
    label_index = {label: index for index, label in enumerate(labels)}
    y = np.asarray([label_index[bird] for bird in birds], dtype=np.int16)
    units = syllables_by_clip()
    rows, predictions = [], {}

    for fold in range(3):
        started = time.perf_counter()
        train_indices = np.flatnonzero(folds != fold)
        val_indices = np.flatnonzero(folds == fold)
        pca = PCA(PCA_COMPONENTS, svd_solver="randomized", random_state=42 + fold)
        train = pca.fit_transform(x[train_indices])
        val = pca.transform(x[val_indices])
        mean = train.mean(axis=0, dtype=np.float64)
        std = np.maximum(train.std(axis=0, dtype=np.float64), 1e-6)
        train = ((train - mean) / std).astype(np.float32)
        val = ((val - mean) / std).astype(np.float32)
        train_x, train_y, _ = pooled(train, y[train_indices], stems[train_indices])
        val_x, val_y, val_stems = pooled(val, y[val_indices], stems[val_indices])
        probe = LogisticRegression(C=LOGREG_C, class_weight="balanced", max_iter=5000)
        probe.fit(train_x, train_y)
        assert np.array_equal(probe.classes_, np.arange(len(labels)))

        token_lookup = {index: row for row, index in enumerate(val_indices)}
        logits = probe.decision_function(val_x)
        for recording, truth, scores in zip(val_stems, val_y, logits):
            competitor_scores = scores.copy()
            competitor_scores[truth] = -np.inf
            competitor = int(competitor_scores.argmax())
            predictions[recording] = (int(scores.argmax()), int(truth))
            indices = np.flatnonzero(stems == recording)
            transformed = val[[token_lookup[index] for index in indices]]
            contributions = transformed @ (probe.coef_[truth] - probe.coef_[competitor])
            bias = probe.intercept_[truth] - probe.intercept_[competitor]
            assert np.isclose(contributions.mean() + bias, scores[truth] - scores[competitor], atol=2e-4)
            for index, contribution in zip(indices, contributions):
                center = (starts[index] + ends[index]) / 2
                syllable = next((unit for unit in units[recording] if unit[0] <= center < unit[1]), None)
                rows.append({
                    "fold": fold, "bird": labels[truth], "recording_stem": recording,
                    "source_recording": sources[index], "start_ms": starts[index], "end_ms": ends[index],
                    "strongest_competitor": labels[competitor], "token_contribution": float(contribution),
                    "recording_logit_margin": float(scores[truth] - scores[competitor]),
                    "bias_margin": float(bias), "syllable_id": "" if syllable is None else syllable[2],
                    "normalized_syllable_time": "" if syllable is None else (center - syllable[0]) / (syllable[1] - syllable[0]),
                })
        print(f"fold {fold + 1}/3 complete in {time.perf_counter() - started:.1f}s", flush=True)
    return rows, predictions, labels


def rank(rows):
    selected = []
    for bird in sorted(set(row["bird"] for row in rows)):
        bird_rows = sorted((row for row in rows if row["bird"] == bird), key=lambda row: -row["token_contribution"])
        for bird_rank, row in enumerate(bird_rows, 1):
            row["bird_rank"] = bird_rank
        for row in bird_rows:
            center = (row["start_ms"] + row["end_ms"]) / 2
            if any(old["recording_stem"] == row["recording_stem"] and abs((old["start_ms"] + old["end_ms"]) / 2 - center) < MIN_SEPARATION_MS for old in selected if old["bird"] == bird):
                continue
            selected.append(row)
            if sum(old["bird"] == bird for old in selected) == TOP_PER_BIRD:
                break
    return selected


def write(rows, selected, predictions, labels):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = (
        "bird", "bird_rank", "fold", "recording_stem", "source_recording", "start_ms", "end_ms",
        "strongest_competitor", "token_contribution", "recording_logit_margin", "bias_margin",
        "syllable_id", "normalized_syllable_time",
    )
    for name, values in (("token_contributions.tsv", rows), ("most_informative_tokens.tsv", selected)):
        with (OUTPUT / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(sorted(values, key=lambda row: (row["bird"], row["bird_rank"])))

    per_bird = {}
    for bird in labels:
        bird_rows = [row for row in rows if row["bird"] == bird]
        records = [stem for stem, (_, truth) in predictions.items() if labels[truth] == bird]
        per_bird[bird] = {
            "recordings": len(records),
            "recording_accuracy": float(np.mean([predictions[stem][0] == predictions[stem][1] for stem in records])),
            "mean_token_contribution": float(np.mean([row["token_contribution"] for row in bird_rows])),
            "positive_token_fraction": float(np.mean([row["token_contribution"] > 0 for row in bird_rows])),
            "top_20_mean_contribution": float(np.mean([row["token_contribution"] for row in selected if row["bird"] == bird])),
        }
    summary = {
        "model": MODEL, "species": "zebra_finch", "condition": "clean", "layer": LAYER,
        "probe": "multinomial_logistic_regression_on_mean_pooled_tokens", "folds": 3,
        "held_out_unit": "source_recording", "pca_components": PCA_COMPONENTS,
        "standardization_fit_scope": "training_fold_tokens", "logreg_c": LOGREG_C,
        "classes": len(labels), "recordings": len(predictions), "tokens": len(rows),
        "recording_accuracy": float(np.mean([prediction == truth for prediction, truth in predictions.values()])),
        "mean_token_contribution": float(np.mean([row["token_contribution"] for row in rows])),
        "positive_token_fraction": float(np.mean([row["token_contribution"] > 0 for row in rows])),
        "per_bird": per_bird,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def context(row):
    spec = np.load(SPECS / f"{row['recording_stem']}.npy")
    center = (row["start_ms"] + row["end_ms"]) / 2
    first = max(0, round((center - 300) / 5))
    last = min(len(spec), round((center + 300) / 5))
    return spec[first:last].T, (first * 5 - center) / 1000, (last * 5 - center) / 1000


def plot(selected):
    top = [min((row for row in selected if row["bird"] == bird), key=lambda row: row["bird_rank"]) for bird in sorted(set(row["bird"] for row in selected))]
    fig, axes = plt.subplots(6, 6, figsize=(12, 10), dpi=180)
    for axis, row in zip(axes.flat, top):
        patch, left, right = context(row)
        axis.imshow(patch, origin="lower", aspect="auto", extent=(left, right, 0, 128), cmap="magma", vmin=-65, vmax=-15)
        center = (row["start_ms"] + row["end_ms"]) / 2
        axis.axvspan((row["start_ms"] - center) / 1000, (row["end_ms"] - center) / 1000, color="cyan", alpha=0.35)
        axis.set_title(f"{row['bird']}  m={row['token_contribution']:.2f}", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Most identity-supporting held-out token for each zebra finch", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.8, w_pad=0.4)
    fig.savefig(OUTPUT / "most_informative_token_per_bird.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "most_informative_token_per_bird.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    rows, predictions, labels = decompose()
    selected = rank(rows)
    summary = write(rows, selected, predictions, labels)
    plot(selected)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
