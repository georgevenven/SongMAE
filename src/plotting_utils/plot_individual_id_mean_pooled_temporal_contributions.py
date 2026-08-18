#!/usr/bin/env python3
"""Decompose a mean-pooled SongMAE identity probe over temporal tokens."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.embedding_store import EmbeddingStore
from src.evals.individual_id_song_vs_non_song_probe import ANNOTATIONS, LOGREG_C, PCA_COMPONENTS, split


SPECS = Path("/media/george-vengrovski/disk2/zebra_finch_pink_noise_event_probe_full/specs/pink_0db")
MODELS = {
    "songmae": {
        "name": "xcl_large_500k_p32x4_c010", "label": "SongMAE 32 × 4", "dimensions": 3072,
        "height_patches": 4, "embeddings": "embeddings_songmae_32x4/pink_0db",
        "output": "mean_pooled_temporal_decomposition_songmae_32x4_pink_0db",
    },
    "hubert": {
        "name": "facebook/hubert-base-ls960", "label": "HuBERT-base", "dimensions": 768,
        "height_patches": None, "embeddings": "embeddings_hubert_base_full/pink_0db",
        "output": "mean_pooled_temporal_decomposition_hubert_base_pink_0db",
    },
}
MODEL = sys.argv[1] if len(sys.argv) == 2 else "songmae"
assert MODEL in MODELS, f"choose one of: {', '.join(MODELS)}"
CONFIG = MODELS[MODEL]
EMBEDDINGS = Path("/media/george-vengrovski/disk2/zebra_finch_pink_noise_event_probe_full") / CONFIG["embeddings"]
OUTPUT = ROOT / "Individual_Id_paper_materials/token_analysis" / CONFIG["output"]
REGIONS = ("syllable", "within_event_gap", "background")
REGION_LABELS = ("Annotated syllable", "Within-event gap", "Outside song event")
COLORS = ("#0072B2", "#E69F00", "#999999")


def load():
    store = EmbeddingStore(EMBEDDINGS)
    x = store["encoded_embeddings"]
    stems = np.asarray(store["recording_stem"]).astype(str)
    starts = np.asarray(store["token_start_ms"], dtype=np.float32)
    ends = np.asarray(store["token_end_ms"], dtype=np.float32)
    metadata = store.metadata
    assert x.ndim == 2 and x.shape[1] == CONFIG["dimensions"]
    if MODEL == "songmae":
        assert metadata["encoder_layer_idx"] == 11 and metadata["num_patches_height"] == 4
    else:
        assert metadata["model_name"] == CONFIG["name"] and metadata["encoder_layer_idx"] is None
    boundaries = np.r_[0, np.flatnonzero(stems[1:] != stems[:-1]) + 1, len(stems)]
    names = stems[boundaries[:-1]]
    assert len(names) == len(set(names.tolist())) == 1033
    pooled = np.stack([x[first:last].mean(axis=0, dtype=np.float64) for first, last in zip(boundaries[:-1], boundaries[1:])])
    return x, stems, starts, ends, boundaries, names, pooled, metadata


def region(row, center):
    event = next((event for event in row["detected_events"] if event["onset_ms"] <= center < event["offset_ms"]), None)
    if event is None:
        return "background", ""
    unit = next((unit for unit in event.get("units", []) if unit["onset_ms"] <= center < unit["offset_ms"]), None)
    return ("within_event_gap", "") if unit is None else ("syllable", str(unit["id"]))


def decompose():
    x, stems, starts, ends, boundaries, names, pooled, metadata = load()
    recordings = {Path(row["recording"]["filename"]).stem: row for row in json.loads(ANNOTATIONS.read_text())["recordings"]}
    labels = sorted({str(row["recording"]["bird_id"]) for row in recordings.values()})
    label_index = {label: index for index, label in enumerate(labels)}
    y = np.asarray([label_index[str(recordings[name]["recording"]["bird_id"])] for name in names])
    train_stems, test_stems = split(recordings)
    train = np.flatnonzero(np.isin(names, list(train_stems)))
    test = np.flatnonzero(np.isin(names, list(test_stems)))

    pca = PCA(PCA_COMPONENTS, svd_solver="randomized", random_state=42)
    train_x = pca.fit_transform(pooled[train])
    test_x = pca.transform(pooled[test])
    mean = train_x.mean(axis=0, dtype=np.float64)
    std = np.maximum(train_x.std(axis=0, dtype=np.float64), 1e-6)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    probe = LogisticRegression(C=LOGREG_C, class_weight="balanced", max_iter=5000)
    probe.fit(train_x, y[train])
    logits = probe.decision_function(test_x)
    predictions = logits.argmax(axis=1)
    rows = []

    for position, recording_index in enumerate(test):
        first, last = boundaries[recording_index:recording_index + 2]
        name = names[recording_index]
        truth = int(y[recording_index])
        competitors = logits[position].copy()
        competitors[truth] = -np.inf
        competitor = int(competitors.argmax())
        tokens = (pca.transform(np.asarray(x[first:last], dtype=np.float64)) - mean) / std
        assert np.allclose(tokens.mean(axis=0), test_x[position], atol=5e-4)
        contributions = tokens @ (probe.coef_[truth] - probe.coef_[competitor])
        bias = float(probe.intercept_[truth] - probe.intercept_[competitor])
        margin = float(logits[position, truth] - logits[position, competitor])
        assert np.isclose(contributions.mean() + bias, margin, atol=5e-4)
        for index, contribution in zip(range(first, last), contributions):
            center = float((starts[index] + ends[index]) / 2)
            token_region, syllable_id = region(recordings[name], center)
            rows.append({
                "bird": labels[truth], "recording_stem": name,
                "start_ms": float(starts[index]), "end_ms": float(ends[index]),
                "region": token_region, "syllable_id": syllable_id,
                "strongest_competitor": labels[competitor], "token_contribution": float(contribution),
                "recording_logit_margin": margin, "bias_margin": bias,
                "predicted_bird": labels[int(predictions[position])], "correct": bool(predictions[position] == truth),
            })
    return rows, names[test], y[test], predictions, labels, metadata


def summarize(rows, names, truth, predictions, labels, metadata):
    by_bird_region = defaultdict(lambda: defaultdict(list))
    grouped = defaultdict(list)
    for row in rows:
        by_bird_region[row["bird"]][row["region"]].append(row["token_contribution"])
        grouped[row["recording_stem"]].append(row)
    region_means = {
        key: float(np.mean([np.mean(by_bird_region[bird][key]) for bird in labels])) for key in REGIONS
    }
    region_positive = {
        key: float(np.mean([np.mean(np.asarray(by_bird_region[bird][key]) > 0) for bird in labels])) for key in REGIONS
    }
    all_counts = {key: sum(row["region"] == key for row in rows) for key in REGIONS}
    top = []
    for recording_rows in grouped.values():
        count = max(1, round(len(recording_rows) * 0.1))
        top.extend(sorted(recording_rows, key=lambda row: -row["token_contribution"])[:count])
    top_counts = {key: sum(row["region"] == key for row in top) for key in REGIONS}
    return {
        "model": CONFIG["name"], "species": "zebra_finch", "condition": "pink_0db",
        "layer": 11, "classes": len(labels), "train_recordings": 853, "test_recordings": len(names),
        "probe": "multinomial_logistic_regression_on_mean_pooled_temporal_tokens",
        "temporal_token": "four height patches concatenated at each time position" if MODEL == "songmae" else "one waveform-derived vector per time position",
        "height_patches_per_temporal_token": CONFIG["height_patches"], "temporal_token_dimensions": CONFIG["dimensions"],
        "pca_components": PCA_COMPONENTS, "pca_fit_scope": "training recording means",
        "standardization_fit_scope": "training recording means", "logreg_c": LOGREG_C,
        "split": "5 complete held-out source recordings per bird",
        "recording_accuracy": float(np.mean(predictions == truth)),
        "recording_macro_f1": float(f1_score(truth, predictions, labels=np.arange(len(labels)), average="macro", zero_division=0)),
        "correct_recordings": int(np.sum(predictions == truth)), "held_out_temporal_tokens": len(rows),
        "decomposition": "mean temporal contribution plus bias equals true-bird minus strongest-competitor logit margin",
        "interpretation": "linear readout contribution of each final-layer temporal token, not causal input saliency",
        "bird_balanced_mean_contribution": region_means,
        "bird_balanced_positive_contribution_fraction": region_positive,
        "all_token_region_fraction": {key: all_counts[key] / len(rows) for key in REGIONS},
        "top_10_percent_region_fraction": {key: top_counts[key] / len(top) for key in REGIONS},
    }


def write(rows, summary):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0])
    with (OUTPUT / "temporal_token_contributions.tsv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def plot_summary(summary):
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5), dpi=200)
    means = [summary["bird_balanced_mean_contribution"][key] for key in REGIONS]
    positive = [100 * summary["bird_balanced_positive_contribution_fraction"][key] for key in REGIONS]
    axes[0].bar(range(3), means, color=COLORS)
    axes[0].axhline(0, color="#202020", linewidth=1)
    axes[0].set_ylabel("Mean temporal contribution")
    axes[1].bar(range(3), positive, color=COLORS)
    axes[1].set_ylabel("Positive contributions (%)")
    bottom = np.zeros(2)
    for key, label, color in zip(REGIONS, REGION_LABELS, COLORS):
        values = 100 * np.asarray([summary["all_token_region_fraction"][key], summary["top_10_percent_region_fraction"][key]])
        axes[2].bar((0, 1), values, bottom=bottom, color=color, label=label)
        bottom += values
    axes[2].set_xticks((0, 1), ("All tokens", "Top 10%"))
    axes[2].set_ylabel("Temporal-token composition (%)")
    axes[2].legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    for axis in axes[:2]:
        axis.set_xticks(range(3), REGION_LABELS, rotation=24, ha="right")
    for axis in axes:
        axis.grid(axis="y", alpha=0.18)
    fig.suptitle(f"Where a mean-pooled identity probe reads {CONFIG['label']}", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT / "temporal_contribution_summary.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "temporal_contribution_summary.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_example(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["recording_stem"]].append(row)
    candidates = []
    for name, recording_rows in grouped.items():
        values = np.asarray([row["token_contribution"] for row in recording_rows])
        duration = max(row["end_ms"] for row in recording_rows) / 1000
        if recording_rows[0]["correct"] and 2 <= duration <= 12:
            candidates.append((float(np.percentile(values, 95) - np.percentile(values, 5)), name))
    _, name = max(candidates)
    recording_rows = sorted(grouped[name], key=lambda row: row["start_ms"])
    starts = np.asarray([row["start_ms"] for row in recording_rows]) / 1000
    ends = np.asarray([row["end_ms"] for row in recording_rows]) / 1000
    centers = (starts + ends) / 2
    values = np.asarray([row["token_contribution"] for row in recording_rows])
    spec = np.load(SPECS / f"{name}.npy").T
    limit = max(abs(np.percentile(values, 1)), abs(np.percentile(values, 99)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    fig = plt.figure(figsize=(11, 5.6), dpi=200)
    grid = fig.add_gridspec(3, 2, width_ratios=(1, 0.025), height_ratios=(3.2, 0.28, 1.35), hspace=0.12, wspace=0.12)
    axes = [fig.add_subplot(grid[0, 0])]
    axes.append(fig.add_subplot(grid[1, 0], sharex=axes[0]))
    axes.append(fig.add_subplot(grid[2, 0], sharex=axes[0]))
    colorbar_axis = fig.add_subplot(grid[:, 1])
    duration = spec.shape[1] * 0.005
    axes[0].imshow(spec, origin="lower", aspect="auto", extent=(0, duration, 0, 128), cmap="viridis", vmin=-65, vmax=-15)
    overlay = np.tile(np.interp(np.arange(spec.shape[1]) * 0.005, centers, values), (128, 1))
    axes[0].imshow(overlay, origin="lower", aspect="auto", extent=(0, duration, 0, 128), cmap="RdBu_r", norm=norm, alpha=0.16)
    axes[0].set_ylabel("Mel bin")
    axes[0].tick_params(labelbottom=False)
    gradient = axes[1].imshow(values[None], aspect="auto", extent=(starts[0], ends[-1], 0, 1), cmap="RdBu_r", norm=norm, interpolation="nearest")
    axes[1].set_yticks([])
    axes[1].set_ylabel("m(t)", rotation=0, labelpad=22, va="center")
    axes[1].tick_params(labelbottom=False)
    axes[2].axhline(0, color="#202020", linewidth=1)
    axes[2].plot(centers, values, color="#0072B2", linewidth=1.3)
    for key, color in zip(REGIONS, COLORS):
        mask = np.asarray([row["region"] == key for row in recording_rows])
        axes[2].fill_between(centers, 0, values, where=mask, color=color, alpha=0.28)
    axes[2].set(xlabel="Time (s)", ylabel="Temporal contribution", xlim=(0, duration))
    axes[2].grid(axis="y", alpha=0.18)
    fig.colorbar(gradient, cax=colorbar_axis).set_label("True-bird minus competitor contribution")
    first = recording_rows[0]
    fig.suptitle(f"Mean-pooled identity evidence through one held-out recording · {first['bird']} vs {first['strongest_competitor']}", fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.1, top=0.88)
    fig.savefig(OUTPUT / "whole_recording_temporal_contributions.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "whole_recording_temporal_contributions.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (OUTPUT / "whole_recording_temporal_contributions.json").write_text(json.dumps({
        "recording_stem": name, "bird": first["bird"], "strongest_competitor": first["strongest_competitor"],
        "duration_seconds": duration, "selection": "largest 95th-minus-5th contribution range among correct 2-to-12-second held-out recordings",
    }, indent=2) + "\n")


def main():
    rows, names, truth, predictions, labels, metadata = decompose()
    summary = summarize(rows, names, truth, predictions, labels, metadata)
    write(rows, summary)
    plot_summary(summary)
    plot_example(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
