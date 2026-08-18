#!/usr/bin/env python3
"""Compare syllable and gap contributions from a pink-noise token probe."""
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
EMBEDDINGS = Path("/media/george-vengrovski/disk2/individual_id_all_layers_10_species/embeddings/zebra_finch") / MODEL / "pink_0db"
SPECS = Path("/media/george-vengrovski/disk2/individual_id_pink_noise_same_condition_0db/zebra_finch/spec")
ANNOTATIONS = ROOT / "results/individual_id/individual_id_linear_probe/multispecies_background_robustness/zebra_finch/clean_annotations.json"
CLIP_MAP = ANNOTATIONS.parent / "clip_map.json"
OUTPUT = ROOT / "Individual_Id_paper_materials/token_analysis/probe_decomposition_songmae_32x4_pink_0db_token"
LAYER = 11
PCA_COMPONENTS = 768
LOGREG_C = 1e-3
TOP_PER_BIRD = 20
MIN_SEPARATION_MS = 250


def load():
    store = EmbeddingStore(EMBEDDINGS)
    x = store["encoded_embeddings"][:, LAYER]
    stems = np.asarray(store["recording_stem"]).astype(str)
    starts = np.asarray(store["token_start_ms"], dtype=np.float32)
    ends = np.asarray(store["token_end_ms"], dtype=np.float32)
    annotations = load_annotations(ANNOTATIONS)
    birds = np.asarray([annotations[stem]["bird"] for stem in stems])
    clips = {
        row["composite_stem"]: row for row in json.loads(CLIP_MAP.read_text())
        if row["condition"] == "clean"
    }
    folds = np.asarray([int(clips[stem]["fold"]) for stem in stems], dtype=np.int8)
    sources = np.asarray([clips[stem]["source_stem"] for stem in stems])
    for fold in range(3):
        assert not set(sources[folds == fold]) & set(sources[folds != fold])
    return x, birds, stems, starts, ends, folds, sources


def softmax(logits):
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


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
        probe = LogisticRegression(C=LOGREG_C, class_weight="balanced", max_iter=5000)
        probe.fit(train, y[train_indices])
        assert np.array_equal(probe.classes_, np.arange(len(labels)))
        logits = probe.decision_function(val)
        probabilities = softmax(logits)
        lookup = np.full(len(x), -1, dtype=np.int64)
        lookup[val_indices] = np.arange(len(val_indices))

        for recording in sorted(set(stems[val_indices].tolist())):
            indices = np.flatnonzero(stems == recording)
            positions = lookup[indices]
            truth = int(y[indices[0]])
            averaged = probabilities[positions].mean(axis=0)
            prediction = int(averaged.argmax())
            competitors = averaged.copy()
            competitors[truth] = -np.inf
            competitor = int(competitors.argmax())
            contributions = val[positions] @ (probe.coef_[truth] - probe.coef_[competitor])
            bias = float(probe.intercept_[truth] - probe.intercept_[competitor])
            predictions[recording] = (prediction, truth)
            for index, contribution in zip(indices, contributions):
                center = (starts[index] + ends[index]) / 2
                syllable = next((unit for unit in units[recording] if unit[0] <= center < unit[1]), None)
                rows.append({
                    "fold": fold, "bird": labels[truth], "recording_stem": recording,
                    "source_recording": sources[index], "start_ms": starts[index], "end_ms": ends[index],
                    "strongest_competitor": labels[competitor], "token_contribution": float(contribution),
                    "token_logit_margin": float(contribution + bias), "bias_margin": bias,
                    "syllable_id": "" if syllable is None else syllable[2],
                    "normalized_syllable_time": "" if syllable is None else (center - syllable[0]) / (syllable[1] - syllable[0]),
                })
        print(f"fold {fold + 1}/3 complete in {time.perf_counter() - started:.1f}s", flush=True)
    return rows, predictions, labels


def select(rows):
    selected = []
    for bird in sorted(set(row["bird"] for row in rows)):
        bird_rows = sorted((row for row in rows if row["bird"] == bird), key=lambda row: -row["token_contribution"])
        for bird_rank, row in enumerate(bird_rows, 1):
            row["bird_rank"] = bird_rank
        for row in bird_rows:
            center = (row["start_ms"] + row["end_ms"]) / 2
            nearby = [old for old in selected if old["bird"] == bird and old["recording_stem"] == row["recording_stem"]]
            if any(abs((old["start_ms"] + old["end_ms"]) / 2 - center) < MIN_SEPARATION_MS for old in nearby):
                continue
            selected.append(row)
            if sum(old["bird"] == bird for old in selected) == TOP_PER_BIRD:
                break
    return selected


def summarize(rows, predictions, labels, selected):
    by_bird = {}
    for bird in labels:
        bird_rows = [row for row in rows if row["bird"] == bird]
        inside = [row["token_contribution"] for row in bird_rows if row["syllable_id"] != ""]
        between = [row["token_contribution"] for row in bird_rows if row["syllable_id"] == ""]
        records = [stem for stem, (_, truth) in predictions.items() if labels[truth] == bird]
        by_bird[bird] = {
            "recording_accuracy": float(np.mean([predictions[stem][0] == predictions[stem][1] for stem in records])),
            "mean_syllable_contribution": float(np.mean(inside)),
            "mean_between_syllable_contribution": float(np.mean(between)),
            "syllable_minus_between": float(np.mean(inside) - np.mean(between)),
        }

    top_ten = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["recording_stem"]].append(row)
    for recording_rows in grouped.values():
        count = max(1, round(len(recording_rows) * 0.1))
        top_ten.extend(sorted(recording_rows, key=lambda row: -row["token_contribution"])[:count])
    summary = {
        "model": MODEL, "species": "zebra_finch", "condition": "pink_0db", "layer": LAYER,
        "probe": "token_level_multinomial_logistic_regression", "recording_aggregation": "mean_token_probability",
        "token_score": "true_logit_minus_strongest_recording_competitor_logit_without_bias",
        "folds": 3, "held_out_unit": "source_recording", "pca_components": PCA_COMPONENTS,
        "standardization_fit_scope": "training_fold_tokens", "logreg_c": LOGREG_C,
        "classes": len(labels), "recordings": len(predictions), "tokens": len(rows),
        "recording_accuracy": float(np.mean([prediction == truth for prediction, truth in predictions.values()])),
        "all_token_syllable_share": float(np.mean([row["syllable_id"] != "" for row in rows])),
        "top_10_percent_syllable_share": float(np.mean([row["syllable_id"] != "" for row in top_ten])),
        "top_20_per_bird_syllable_share": float(np.mean([row["syllable_id"] != "" for row in selected])),
        "bird_balanced_mean_syllable_contribution": float(np.mean([row["mean_syllable_contribution"] for row in by_bird.values()])),
        "bird_balanced_mean_between_syllable_contribution": float(np.mean([row["mean_between_syllable_contribution"] for row in by_bird.values()])),
        "birds_with_larger_syllable_contribution": int(sum(row["syllable_minus_between"] > 0 for row in by_bird.values())),
        "per_bird": by_bird,
    }
    return summary


def write(rows, selected, summary):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = (
        "bird", "bird_rank", "fold", "recording_stem", "source_recording", "start_ms", "end_ms",
        "strongest_competitor", "token_contribution", "token_logit_margin", "bias_margin",
        "syllable_id", "normalized_syllable_time",
    )
    for name, values in (("token_contributions.tsv", rows), ("most_informative_tokens.tsv", selected)):
        with (OUTPUT / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(sorted(values, key=lambda row: (row["bird"], row["bird_rank"])))
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def plot_summary(summary):
    birds = sorted(summary["per_bird"])
    between = np.asarray([summary["per_bird"][bird]["mean_between_syllable_contribution"] for bird in birds])
    syllable = np.asarray([summary["per_bird"][bird]["mean_syllable_contribution"] for bird in birds])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), dpi=200)
    for first, last in zip(between, syllable):
        axes[0].plot((0, 1), (first, last), color="#A0A0A0", linewidth=0.8, alpha=0.65)
    axes[0].plot((0, 1), (between.mean(), syllable.mean()), "o-", color="#0072B2", linewidth=3, markersize=7)
    axes[0].set_xticks((0, 1), ("Between syllables", "Annotated syllables"))
    axes[0].set_ylabel("Mean token contribution")
    axes[0].grid(axis="y", alpha=0.18)
    shares = np.asarray([summary["all_token_syllable_share"], summary["top_10_percent_syllable_share"]]) * 100
    axes[1].bar(0, shares[0], color="gray", width=0.65)
    axes[1].bar(1, shares[1], color="tab:orange", width=0.65)
    axes[1].set_xticks((0, 1), ("All tokens", "Top 10% per clip"))
    axes[1].set_ylabel("Tokens inside annotated syllables (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.18)
    fig.suptitle("Pink-noise token probe: where identity evidence occurs", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT / "syllable_vs_between_summary.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "syllable_vs_between_summary.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_tokens(selected):
    top = [min((row for row in selected if row["bird"] == bird), key=lambda row: row["bird_rank"]) for bird in sorted(set(row["bird"] for row in selected))]
    fig, axes = plt.subplots(6, 6, figsize=(12, 10), dpi=180)
    for axis, row in zip(axes.flat, top):
        spec = np.load(SPECS / f"{row['recording_stem']}.npy")
        center = (row["start_ms"] + row["end_ms"]) / 2
        first, last = max(0, round((center - 300) / 5)), min(len(spec), round((center + 300) / 5))
        axis.imshow(spec[first:last].T, origin="lower", aspect="auto", extent=((first * 5 - center) / 1000, (last * 5 - center) / 1000, 0, 128), cmap="magma", vmin=-65, vmax=-15)
        axis.axvspan((row["start_ms"] - center) / 1000, (row["end_ms"] - center) / 1000, color="cyan", alpha=0.35)
        axis.set_title(f"{row['bird']}  m={row['token_contribution']:.2f}", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Most identity-supporting pink-noise token for each zebra finch", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.8, w_pad=0.4)
    fig.savefig(OUTPUT / "most_informative_token_per_bird.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "most_informative_token_per_bird.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    rows, predictions, labels = decompose()
    selected = select(rows)
    summary = summarize(rows, predictions, labels, selected)
    write(rows, selected, summary)
    plot_summary(summary)
    plot_tokens(selected)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
