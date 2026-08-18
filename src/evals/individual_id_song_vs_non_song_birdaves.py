#!/usr/bin/env python3
"""Compare final-layer BirdAVES and SongMAE identity tiers."""
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.embedding_store import EmbeddingStore
from src.evals.individual_id_song_vs_non_song_probe import ANNOTATIONS, evaluate as probe, split
from src.evals.individual_id_song_vs_non_song_purity_layer_sweep import evaluate as purity


EMBEDDINGS = Path("/media/george-vengrovski/disk2/zebra_finch_pink_noise_event_probe_full/embeddings_birdaves_isolated")
OUTPUT = ROOT / "Individual_Id_paper_materials/token_analysis/individual_id_song_vs_non_song_pink_0db_birdaves_comparison"
SONGMAE_PROBE = ROOT / "Individual_Id_paper_materials/token_analysis/individual_id_song_vs_non_song_pink_0db_full_isolated/summary.json"
SONGMAE_PURITY = ROOT / "Individual_Id_paper_materials/token_analysis/individual_id_song_vs_non_song_pink_0db_purity_k50_layer_sweep_full_isolated/summary.json"
SCOPES = {"song": "events", "non_song": "background"}


def load(scope, recordings, labels):
    store = EmbeddingStore(EMBEDDINGS / SCOPES[scope] / "pink_0db")
    x = np.asarray(store["encoded_embeddings"], dtype=np.float32)
    stems = np.asarray(store["recording_stem"]).astype(str)
    label_index = {label: index for index, label in enumerate(labels)}
    y = np.asarray([label_index[str(recordings[stem]["recording"]["bird_id"])] for stem in stems])
    return x, stems, y


def songmae_rows():
    probe_rows = {row["scope"]: row for row in json.loads(SONGMAE_PROBE.read_text())["results"]}
    purity_rows = {
        row["scope"]: row for row in json.loads(SONGMAE_PURITY.read_text())["results"] if row["layer"] == 11
    }
    return [{"model": "SongMAE 32 × 4", "scope": scope, **purity_rows[scope],
             "logistic_token_accuracy": probe_rows[scope]["token_accuracy"],
             "logistic_recording_accuracy": probe_rows[scope]["recording_accuracy"]} for scope in SCOPES]


def plot(rows):
    metrics = (
        ("macro_same_identity_purity", "Purity"),
        ("token_majority_accuracy", "kNN token"),
        ("recording_accuracy", "kNN recording"),
        ("logistic_token_accuracy", "Logistic token"),
        ("logistic_recording_accuracy", "Logistic recording"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), dpi=200, sharey=True)
    colors = ("#0072B2", "#D55E00")
    for axis, scope, title in zip(axes, SCOPES, ("Song events only", "Non-song only")):
        selected = [row for row in rows if row["scope"] == scope]
        x = np.arange(len(metrics))
        for index, (row, color) in enumerate(zip(selected, colors)):
            values = [100 * row[key] for key, _ in metrics]
            axis.bar(x + (index - 0.5) * 0.36, values, 0.36, color=color, label=row["model"])
        axis.set(title=title, xticks=x, xticklabels=[label for _, label in metrics], ylim=(0, 105))
        axis.tick_params(axis="x", rotation=28)
        axis.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Score (%)")
    axes[0].legend(frameon=False)
    fig.suptitle("Zebra-finch identity under 0 dB pink noise", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT / "birdaves_vs_songmae_identity_tiers.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "birdaves_vs_songmae_identity_tiers.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    recordings = {Path(row["recording"]["filename"]).stem: row for row in json.loads(ANNOTATIONS.read_text())["recordings"]}
    labels = sorted({str(row["recording"]["bird_id"]) for row in recordings.values()})
    train_stems, test_stems = split(recordings)
    rows = songmae_rows()
    for scope, directory in SCOPES.items():
        data = load(scope, recordings, labels)
        neighborhood = purity(scope, None, data, train_stems, test_stems, labels)
        supervised, _ = probe(scope, EMBEDDINGS / directory / "pink_0db", recordings, train_stems, test_stems, labels)
        rows.append({"model": "BirdAVES", **neighborhood,
                     "logistic_token_accuracy": supervised["token_accuracy"],
                     "logistic_recording_accuracy": supervised["recording_accuracy"],
                     "logistic_recording_macro_f1": supervised["recording_macro_f1"]})
        print(f"BirdAVES {scope}: purity={neighborhood['macro_same_identity_purity'] * 100:.1f}%, logistic recording={supervised['recording_accuracy'] * 100:.1f}%", flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "species": "zebra_finch", "condition": "pink_0db", "classes": len(labels),
        "split": "5 complete held-out source recordings per bird", "pca_components": 768,
        "purity": "fraction of k=50 nearest balanced training tokens with the same bird",
        "reference_tokens_per_bird": 256, "input_isolation": "song and non-song encoded separately",
        "rows": rows,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = ("model", "scope", "macro_same_identity_purity", "macro_identity_enrichment", "token_majority_accuracy", "recording_accuracy", "logistic_token_accuracy", "logistic_recording_accuracy")
    with (OUTPUT / "comparison.tsv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    plot(rows)


if __name__ == "__main__":
    main()
