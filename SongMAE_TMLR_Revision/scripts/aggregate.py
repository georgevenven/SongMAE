#!/usr/bin/env python3
"""Aggregate revision results into SongMAE_TMLR_Revision/results/tables.md.

Protocols: `loso`, where each species is scored at the layer selected on the other two, and `canary`, where layers
are selected on canary (a development set; zf and bf are the test sets). Averaging is birds within species, then
species. Spread: SD across birds within species, mean SD across the 3 folds, and SD across the 5 kNN sampling seeds.
kNN purity leaves out classes in fewer than two recordings of a bird's kNN selection (results/knn_rare_classes.json).
"""
import collections
import csv
import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "SongMAE_TMLR_Revision/results"
SEL = json.loads((OUT / "layer_selection.json").read_text())
RARE = json.loads((OUT / "knn_rare_classes.json").read_text())
SPECIES = ("canary", "zf", "bf")
NAMES = {"canary": "Canary", "zf": "Zebra finch", "bf": "Bengalese finch"}
ORDER = ["xcl_large_500k_p32x1_c005", "xcl_large_500k_p32x4_c010", "xcl_large_500k_p32x4_c0025", "xcl_base_500k_p32x1_c005",
         "xcl_base_500k_p32x4_c010", "xcl_micro_500k_p32x1_c005", "xcl_micro_500k_p32x4_c010", "birdaves_biox_base",
         "birdaves_biox_base_speed0p5", "birdaves_biox_base_speed0p25", "hubert_base_ls960", "beats_iter3_plus_as2m",
         "beats_iter3_plus_as2m_speed0p5", "beats_iter3_plus_as2m_speed0p25", "beats_iter3_plus_as2m_speed0p125",
         "beats_iter3_plus_as2m_speed0p0625", "beats_iter3_plus_as2m_speed0p03125", "birdmae_base_speed1", "birdmae_base_speed0p5",
         "birdmae_base_speed0p25", "birdmae_base_speed0p125", "birdmae_base_speed0p0625", "birdmae_base_speed0p03125"]
ORDER = [c for c in ORDER if c in SEL]  # conditions still awaiting layer selection are skipped
LABEL = {"xcl_large_500k_p32x1_c005": "SongMAE-L 32×5 ms", "xcl_large_500k_p32x4_c010": "SongMAE-L 32×20 ms",
         "xcl_large_500k_p32x4_c0025": "SongMAE-L 32×20 ms (2.5% seeds)", "xcl_base_500k_p32x1_c005": "SongMAE-B 32×5 ms",
         "xcl_base_500k_p32x4_c010": "SongMAE-B 32×20 ms", "xcl_micro_500k_p32x1_c005": "SongMAE-Micro 32×5 ms",
         "xcl_micro_500k_p32x4_c010": "SongMAE-Micro 32×20 ms", "birdaves_biox_base": "BirdAVES 1×",
         "birdaves_biox_base_speed0p5": "BirdAVES ½×", "birdaves_biox_base_speed0p25": "BirdAVES ¼×", "hubert_base_ls960": "HuBERT",
         "beats_iter3_plus_as2m": "BEATs 1×", "beats_iter3_plus_as2m_speed0p5": "BEATs ½×", "beats_iter3_plus_as2m_speed0p25": "BEATs ¼×",
         "beats_iter3_plus_as2m_speed0p125": "BEATs ⅛×", "beats_iter3_plus_as2m_speed0p0625": "BEATs 1/16×",
         "beats_iter3_plus_as2m_speed0p03125": "BEATs 1/32×",
         "birdmae_base_speed1": "Bird-MAE 1×", "birdmae_base_speed0p5": "Bird-MAE ½×", "birdmae_base_speed0p25": "Bird-MAE ¼×",
         "birdmae_base_speed0p125": "Bird-MAE ⅛×", "birdmae_base_speed0p0625": "Bird-MAE 1/16×", "birdmae_base_speed0p03125": "Bird-MAE 1/32×"}


def layer(condition, species, protocol):
    return SEL[condition]["test_" + species if protocol == "loso" else "canary"]


def load(pattern, read, key):
    """key(parts) -> (table key, species, bird); read(path, species, bird) -> value."""
    out = collections.defaultdict(dict)
    for path in glob.glob(str(ROOT / pattern)):
        k, species, bird = key(Path(path).relative_to(ROOT).parts)
        out[k][bird] = read(path, species, bird)
    return out


def probe_value(path, species, bird):
    d = json.loads(Path(path).read_text())
    folds = [f["macro_fer"] for f in d["fold_metrics"]]
    return 100 * d["macro_fer"], 100 * d["macro_parsing_error"], 100 * d["macro_identity_error"], 100 * st.pstdev(folds)


def knn_value(path, species, bird):
    row = next(r for r in json.loads(Path(path).read_text())["rows"] if r["k"] == 100)
    rare = {str(c) for c in RARE.get(f"{species}/{bird}", [])}
    return 100 * st.mean(v for c, v in row["per_class_same_purity"].items() if c not in rare)


def kmeans_value(path, species, bird):
    return float(next(csv.DictReader(open(path)))["v_measure"])


def species_stats(values):  # values: {bird: tuple or float}; returns mean (per component) and SD across birds
    rows = [v if isinstance(v, tuple) else (v,) for v in values.values()]
    return [st.mean(c) for c in zip(*rows)], (st.stdev([r[0] for r in rows]) if len(rows) > 1 else 0.0), len(rows)


def fmt_row(label, per, extra=""):
    mean = st.mean(per[s][0][0] for s in SPECIES)
    test = st.mean(per[s][0][0] for s in ("zf", "bf"))
    cells = " | ".join(f"{per[s][0][0]:.2f} ± {per[s][1]:.2f}" for s in SPECIES)
    return f"| {label} | {cells} | {mean:.2f} | {test:.2f} |{extra}"


lines = ["# Revision results", "", __doc__.strip().split("\n\n", 1)[1], ""]

probe_key = lambda p: ((p[4], int(p[5][6:]), p[2], p[6][4:] if p[6].startswith("cap_") else "full"), p[2], p[3])
probes = load("results/probes/*/*/*/layer_*/metrics.json", probe_value, probe_key)
probes.update(load("results/probes/*/*/*/layer_*/cap_*/metrics.json", probe_value, probe_key))
for protocol, title in [("loso", "leave-one-species-out"), ("canary", "canary-only selection (canary = development)")]:
    lines += [f"## Linear-probe Macro FER (%), {title}", "",
              "Mean ± SD across birds. Parsing / identity are species-averaged; fold SD is the mean within-bird SD across folds.", "",
              "| Model | " + " | ".join(NAMES[s] for s in SPECIES) + " | Mean (3 spp.) | Mean (zf+bf) | Parsing | Identity | Fold SD |",
              "|---" * 9 + "|"]
    for c in ORDER:
        if any((c, layer(c, s, protocol), s, "full") not in probes for s in SPECIES):
            continue
        per = {s: species_stats(probes[(c, layer(c, s, protocol), s, "full")]) for s in SPECIES}
        parsing, identity, fold = (st.mean(per[s][0][i] for s in SPECIES) for i in (1, 2, 3))
        lines.append(fmt_row(LABEL[c], per, f" {parsing:.2f} | {identity:.2f} | {fold:.2f} |"))
    lines.append("")

caps = sorted({k[3] for k in probes if k[3] != "full"}, key=int)
lines += ["## Label budget: Macro FER (%) by labeled occurrences per class, leave-one-species-out", "",
          "| Model | " + " | ".join(f"N = {int(c)}" for c in caps) + " |", "|---" * (len(caps) + 1) + "|"]
for c in ORDER:
    if all((c, layer(c, s, "loso"), s, caps[0]) in probes for s in SPECIES):
        cells = [st.mean(species_stats(probes[(c, layer(c, s, "loso"), s, cap)])[0][0] for s in SPECIES) for cap in caps]
        lines.append(f"| {LABEL[c]} | " + " | ".join(f"{v:.2f}" for v in cells) + " |")
lines.append("")

knn = load("results/knn/review_baselines_all_layers/*/*/*/layer_*/summary.json", knn_value,
           lambda p: ((p[5], int(p[6][6:]), p[3], "sweep"), p[3], p[4]))
seeds = load("results/knn/seeds/*/*/*/layer_*/seed_*/summary.json", knn_value,
             lambda p: ((p[5], int(p[6][6:]), p[3], p[7]), p[3], p[4]))
lines += ["## kNN purity (%, k = 100), leave-one-species-out", "",
          "Mean ± SD across birds (layer sweep, seed 42). Seed SD: SD across sampling seeds of the species-averaged purity, where available.", "",
          "| Model | " + " | ".join(NAMES[s] for s in SPECIES) + " | Mean (3 spp.) | Mean (zf+bf) | Seed SD |", "|---" * 7 + "|"]
for c in ORDER:
    if any((c, layer(c, s, "loso"), s, "sweep") not in knn for s in SPECIES):
        continue
    per = {s: species_stats(knn[(c, layer(c, s, "loso"), s, "sweep")]) for s in SPECIES}
    seed_means = []
    for seed in sorted({k[3] for k in seeds if k[0] == c}):
        keys = [(c, layer(c, s, "loso"), s, seed) for s in SPECIES]
        if all(k in seeds and len(seeds[k]) == len(knn[(c, layer(c, s, "loso"), s, "sweep")]) for k, s in zip(keys, SPECIES)):
            seed_means.append(st.mean(st.mean(seeds[k].values()) for k in keys))
    sd = f"{st.stdev(seed_means):.2f} (n={len(seed_means)})" if len(seed_means) > 1 else "pending"
    lines.append(fmt_row(LABEL[c], per, f" {sd} |"))
lines.append("")

kmeans = load("results/kmeans/*/*/*/layer_*/metrics.csv", kmeans_value, lambda p: ((p[4], int(p[5][6:]), p[2], "kmeans"), p[2], p[3]))
lines += ["## K-means V-measure (K = classes + silence, PCA-128), leave-one-species-out", "",
          "| Model | " + " | ".join(NAMES[s] for s in SPECIES) + " | Mean (3 spp.) | Mean (zf+bf) |", "|---" * 6 + "|"]
done = 0
for c in ORDER:
    if all((c, layer(c, s, "loso"), s, "kmeans") in kmeans for s in SPECIES):
        per = {s: species_stats(kmeans[(c, layer(c, s, "loso"), s, "kmeans")]) for s in SPECIES}
        lines.append(fmt_row(LABEL[c], per)); done += 1
if not done:
    lines.append("| pending | | | | | |")
lines += ["", "## Selected layers", "", "| Model | all | canary | test canary | test zf | test bf |", "|---" * 6 + "|"]
lines += [f"| {LABEL[c]} | " + " | ".join(f"L{SEL[c][v]}" for v in ("all", "canary", "test_canary", "test_zf", "test_bf")) + " |"
          for c in ORDER if c in SEL]
(OUT / "tables.md").write_text("\n".join(lines) + "\n")
print(OUT / "tables.md")
