#!/usr/bin/env python3
"""Replay the paper's kNN-purity model selection (k=100) on subsets of evaluation birds."""
import collections, glob, json, random, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "results"
LARGE = ["xcl_large_500k_p32x1_c005", "xcl_large_500k_p32x4_c010", "birdaves_biox_base", "hubert_base_ls960"]
PAPER = {"masking": "p32x1_c010", "patch": "p32x1_c010", "coarse_patch": "p32x4_c010", "seed_5ms": "p32x1_c005",
         "seed_20ms": "p32x4_c010", "large_20ms": "c010", **dict(zip(LARGE, [11, 10, 7, 0]))}
scores = collections.defaultdict(dict)  # key -> {(species, bird): purity}


def load(pattern, key):
    for path in glob.glob(str(ROOT / pattern)):
        parts = Path(path).relative_to(ROOT).parts
        row = next(r for r in json.loads(Path(path).read_text())["rows"] if r["k"] == 100)
        species, bird, name = key(parts)
        scores[name][(species, bird)] = row["macro_same_purity"]


load("knn/micro_ablations_all_k_manuscript/raw/*/*/*/layer_*/end_of_block/summary.json",
     lambda p: (p[3], p[4], p[5].lower().removeprefix("xcl_micro_100k_")))
load("knn/four_models_all_layers_raw/*/*/*/layer_*/summary.json", lambda p: (p[2], p[3], (p[4], int(p[5][6:]))))
load("knn/four_models_all_layers_raw/*/*/*/layer_*/end_of_block/summary.json", lambda p: (p[2], p[3], (p[4], int(p[5][6:]))))
load("results_archive/xcl_large_500k_p32x4_c0025_raw/knn_best_layers/*/*/*/layer_*/end_of_block/summary.json",
     lambda p: (p[3], p[4], ("large_p32x4_c0025", int(p[6][6:]))))
birds = collections.defaultdict(list)
for species, bird in scores["p32x1_c010"]:
    birds[species].append(bird)


def mean(name, subset):  # birds within species, then species
    return st.mean(st.mean(scores[name][(s, b)] for b in bs) for s, bs in subset.items() if bs)


def select(subset):
    best = lambda names: max(names, key=lambda n: mean(n, subset))
    layer = lambda model: max(range(12), key=lambda l: mean((model, l), subset))
    peak = lambda model: mean((model, layer(model)), subset)
    return {
        "masking": best(["p32x1_random", "p32x1_c010"]),
        "patch": best(["p128x1_default", "p32x1_c010", "p16x1_default", "p32x4_c010", "p4x4_default"]),
        "coarse_patch": best(["p32x4_c010", "p4x4_default"]),
        "seed_5ms": best(["p32x1_c0025", "p32x1_c005", "p32x1_c010"]),
        "seed_20ms": best(["p32x4_c0025", "p32x4_c005", "p32x4_c010"]),
        "large_20ms": "c0025" if peak("large_p32x4_c0025") > peak(LARGE[1]) else "c010",
        **{model: layer(model) for model in LARGE},
    }


def differs(subset):
    return {k: v for k, v in select(subset).items() if v != PAPER[k]} or "none"


def without(species=None, bird=None):
    return {s: [b for b in bs if b != bird] for s, bs in birds.items() if s != species}


print("Choices that differ from the submitted paper (k=100 kNN purity, submitted pipeline results).\n")
print("all 50 birds:", differs(birds))
print("canary only:", differs({"canary": birds["canary"]}))
for held in birds:
    print(f"leave-one-species-out, test={held}:", differs(without(species=held)))
lobo = collections.Counter(str(differs(without(bird=b))) for bs in birds.values() for b in bs)
print("leave-one-bird-out (50 folds):", dict(lobo))
rng, counts = random.Random(0), collections.Counter()
for _ in range(2000):
    for k, v in select({s: rng.sample(bs, {"canary": 1, "bf": 4, "zf": 12}[s]) for s, bs in birds.items()}).items():
        counts[k] += v != PAPER[k]
print("random bird splits (val = 1 canary, 4 bf, 12 zf; 2000 draws), fraction differing:",
      {k: f"{v / 2000:.1%}" for k, v in counts.items() if v})
