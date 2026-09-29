#!/usr/bin/env python3
"""Pick each condition's layer by kNN purity (k=100) under every selection variant; write results/layer_selection.json.

Variants: all three species, canary only, and leave-one-species-out (select on two, test the third). Averaging is
birds within species, then species. `union` lists every layer any variant picks, so probes cover all of them.
Classes in fewer than two recordings of a bird's kNN selection (results/knn_rare_classes.json) are left out of its
macro purity: with same-recording neighbours excluded they cannot score for any model.
"""
import collections
import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KNN = ROOT / "results/knn/review_baselines_all_layers"
BIRDS = {"canary": 3, "zf": 36, "bf": 11}
RARE = json.loads((ROOT / "SongMAE_TMLR_Revision/results/knn_rare_classes.json").read_text())
VARIANTS = {"all": ["canary", "zf", "bf"], "canary": ["canary"], "test_canary": ["zf", "bf"],
            "test_zf": ["bf", "canary"], "test_bf": ["zf", "canary"]}

scores = collections.defaultdict(dict)  # (condition, layer) -> {(species, bird): purity}
for path in glob.glob(str(KNN / "*/*/*/layer_*/summary.json")):
    species, bird, condition, layer = Path(path).relative_to(KNN).parts[:4]
    row = next(r for r in json.loads(Path(path).read_text())["rows"] if r["k"] == 100)
    rare = {str(c) for c in RARE.get(f"{species}/{bird}", [])}
    scores[(condition, int(layer[6:]))][(species, bird)] = st.mean(
        v for c, v in row["per_class_same_purity"].items() if c not in rare)

out = {}
for condition in sorted({c for c, _ in scores}):
    layers = sorted(l for c, l in scores if c == condition)
    counts = collections.Counter(s for s, _ in scores[(condition, layers[-1])])
    if any(counts[s] != n for s, n in BIRDS.items()):
        print(f"skip {condition}: incomplete {dict(counts)}")
        continue
    mean = lambda l, spp: st.mean(st.mean(v for (s, _), v in scores[(condition, l)].items() if s == sp) for sp in spp)
    picks = {name: max(layers, key=lambda l: mean(l, spp)) for name, spp in VARIANTS.items()}
    out[condition] = picks | {"union": sorted(set(picks.values()))}
(ROOT / "SongMAE_TMLR_Revision/results/layer_selection.json").write_text(json.dumps(out, indent=2) + "\n")
for condition, picks in out.items():
    print(f"{condition:32s} " + " ".join(f"{k}=L{v}" for k, v in picks.items() if k != "union") + f"  union={picks['union']}")
