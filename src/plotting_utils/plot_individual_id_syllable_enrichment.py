#!/usr/bin/env python3
"""Resolve token identity enrichment onto annotated zebra-finch syllables."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Individual_Id_paper_materials/token_analysis/neighborhood_enrichment"
CLIP_MAP = ROOT / "results/individual_id/individual_id_linear_probe/multispecies_background_robustness/zebra_finch/clip_map.json"
ANNOTATIONS = ROOT / "files/annotation jsons/zf_annotations.json"
MODELS = {
    "xcl_large_500k_p32x4_c010": "SongMAE 32 × 4",
    "xcl_large_500k_p32x1_c005": "SongMAE 32 × 1",
}
BINS = np.linspace(0, 1, 21)


def syllables_by_clip():
    data = json.loads(ANNOTATIONS.read_text())
    recordings = {Path(row["recording"]["filename"]).stem: row for row in data["recordings"]}
    clips = {}
    for row in json.loads(CLIP_MAP.read_text()):
        if row["condition"] != "clean":
            continue
        event = recordings[row["source_stem"]]["detected_events"][int(row["source_event_index"])]
        origin = event["onset_ms"]
        clips[row["composite_stem"]] = [
            (unit["onset_ms"] - origin, unit["offset_ms"] - origin, int(unit["id"]))
            for unit in event["units"]
        ]
    return clips


def load_tokens(model, clips):
    tokens = []
    with (OUTPUT / model / "token_enrichment.tsv").open() as file:
        for row in csv.DictReader(file, delimiter="\t"):
            center = (float(row["start_ms"]) + float(row["end_ms"])) / 2
            match = next((unit for unit in clips[row["recording_stem"]] if unit[0] <= center < unit[1]), None)
            tokens.append({
                "bird": row["bird"], "recording": row["recording_stem"],
                "enrichment": float(row["enrichment_k50"]),
                "shuffled": float(row["shuffled_enrichment_k50"]),
                "syllable_id": None if match is None else match[2],
                "position": None if match is None else (center - match[0]) / (match[1] - match[0]),
            })
    return tokens


def mean_sem(values):
    values = np.asarray(values)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def summarize(model, tokens):
    contexts = []
    for context, inside in (("annotated_syllable", True), ("between_syllables", False)):
        rows = [row for row in tokens if (row["syllable_id"] is not None) == inside]
        by_bird = defaultdict(list)
        for row in rows:
            by_bird[row["bird"]].append(row["enrichment"])
        mean, sem = mean_sem([np.mean(values) for values in by_bird.values()])
        contexts.append((model, context, len(rows), len(by_bird), mean, sem))

    time_rows = []
    for first, last in zip(BINS[:-1], BINS[1:]):
        rows = [row for row in tokens if row["position"] is not None and first <= row["position"] < last]
        by_bird = defaultdict(list)
        shuffled = defaultdict(list)
        for row in rows:
            by_bird[row["bird"]].append(row["enrichment"])
            shuffled[row["bird"]].append(row["shuffled"])
        mean, sem = mean_sem([np.mean(values) for values in by_bird.values()])
        chance, _ = mean_sem([np.mean(values) for values in shuffled.values()])
        time_rows.append((model, first, last, (first + last) / 2, len(rows), len(by_bird), mean, sem, chance))

    grouped = defaultdict(list)
    for row in tokens:
        if row["syllable_id"] is not None:
            grouped[row["bird"], row["syllable_id"]].append(row)
    type_rows = []
    for (bird, syllable_id), rows in sorted(grouped.items()):
        values = [row["enrichment"] for row in rows]
        type_rows.append((
            model, bird, syllable_id, len(rows), len(set(row["recording"] for row in rows)),
            float(np.mean(values)), float(np.mean([row["shuffled"] for row in rows])),
        ))
    return contexts, time_rows, type_rows


def write_table(path, fields, rows):
    with path.open("w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(fields)
        writer.writerows(rows)


def type_heatmap(model, tokens):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in tokens:
        if row["position"] is None:
            continue
        index = min(np.searchsorted(BINS, row["position"], side="right") - 1, len(BINS) - 2)
        grouped[row["bird"], row["syllable_id"]][index].append(row["enrichment"])
    matrix = np.full((len(grouped), len(BINS) - 1), np.nan)
    for row_index, values in enumerate(grouped.values()):
        for column, scores in values.items():
            matrix[row_index, column] = np.mean(scores)
    order = np.argsort(np.nanmean(matrix, axis=1))
    return matrix[order]


def plot(all_tokens, contexts, time_rows):
    colors = ("#0072B2", "#56B4E9")
    fig = plt.figure(figsize=(9.2, 6.1), dpi=200)
    grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 0.045), height_ratios=(1, 1.15), hspace=0.42, wspace=0.3)
    curve = fig.add_subplot(grid[0, 0])
    bars = fig.add_subplot(grid[0, 1])
    heatmaps = [fig.add_subplot(grid[1, index]) for index in range(2)]
    colorbar = fig.add_subplot(grid[1, 2])

    for (model, label), color in zip(MODELS.items(), colors):
        rows = [row for row in time_rows if row[0] == model]
        x = np.asarray([row[3] for row in rows])
        y = np.asarray([row[6] for row in rows])
        sem = np.asarray([row[7] for row in rows])
        curve.plot(x, y, color=color, linewidth=2, label=label)
        curve.fill_between(x, y - sem, y + sem, color=color, alpha=0.2, linewidth=0)
    curve.axhline(0, color="#202020", linewidth=1)
    curve.set(xlabel="Normalized syllable time", ylabel="Identity enrichment (k=50)", xlim=(0, 1))
    curve.legend(frameon=False, fontsize=8)
    curve.grid(alpha=0.18)

    x = np.arange(len(MODELS))
    width = 0.34
    for offset, context, label, color in (
        (-width / 2, "annotated_syllable", "Annotated syllable", "#D55E00"),
        (width / 2, "between_syllables", "Between syllables", "#009E73"),
    ):
        rows = [[row for row in contexts if row[0] == model and row[1] == context][0] for model in MODELS]
        bars.bar(x + offset, [row[4] for row in rows], width, yerr=[row[5] for row in rows], color=color, label=label, capsize=3)
    bars.set_xticks(x, ["32 × 4", "32 × 1"])
    bars.set(xlabel="SongMAE patch", ylabel="Bird-balanced mean enrichment")
    bars.legend(frameon=False, fontsize=8)
    bars.grid(axis="y", alpha=0.18)

    matrices = [type_heatmap(model, all_tokens[model]) for model in MODELS]
    limit = max(np.nanpercentile(np.abs(matrix), 99) for matrix in matrices)
    for axis, matrix, label in zip(heatmaps, matrices, MODELS.values()):
        image = axis.imshow(matrix, origin="lower", aspect="auto", extent=(0, 1, 0, len(matrix)), cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set_title(label)
        axis.set(xlabel="Normalized syllable time", ylabel="Bird-specific syllable types (sorted)")
    fig.colorbar(image, cax=colorbar, label="Mean identity enrichment (k=50)")
    fig.suptitle("Where SongMAE tokens encode zebra-finch identity", fontsize=14)
    fig.subplots_adjust(left=0.09, right=0.94, bottom=0.09, top=0.92)
    fig.savefig(OUTPUT / "syllable_enrichment.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "syllable_enrichment.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    clips = syllables_by_clip()
    all_tokens, contexts, time_rows, type_rows = {}, [], [], []
    for model in MODELS:
        all_tokens[model] = load_tokens(model, clips)
        result = summarize(model, all_tokens[model])
        contexts.extend(result[0])
        time_rows.extend(result[1])
        type_rows.extend(result[2])
    write_table(OUTPUT / "syllable_context_summary.tsv", ("model", "context", "tokens", "birds", "mean_enrichment_k50", "sem"), contexts)
    write_table(OUTPUT / "normalized_syllable_time.tsv", ("model", "bin_start", "bin_end", "bin_center", "tokens", "birds", "mean_enrichment_k50", "sem", "mean_shuffled_k50"), time_rows)
    write_table(OUTPUT / "bird_syllable_type_summary.tsv", ("model", "bird", "syllable_id", "tokens", "recordings", "mean_enrichment_k50", "mean_shuffled_k50"), type_rows)
    plot(all_tokens, contexts, time_rows)


if __name__ == "__main__":
    main()
