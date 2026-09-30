#!/usr/bin/env python3
"""Stage compact per-bird result tables in SongMAE_TMLR_Revision/results/per_bird/ (tracked in git).

Raw outputs stay on the work desktop's disk1 (results/{probes,kmeans,knn/...} are symlinks there); these CSVs hold every
number the paper's tables are built from. kNN purity drops classes in fewer than two recordings of a bird's kNN
selection (results/knn_rare_classes.json); probe FER already drops classes in fewer than three recordings.
"""
import csv
import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "SongMAE_TMLR_Revision/results/per_bird"
RARE = json.loads((ROOT / "SongMAE_TMLR_Revision/results/knn_rare_classes.json").read_text())
OUT.mkdir(parents=True, exist_ok=True)


def write(name, header, rows):
    rows = sorted(rows)
    with (OUT / f"{name}.csv").open("w", newline="") as f:
        csv.writer(f).writerows([header, *rows])
    print(f"{name}.csv: {len(rows)} rows")


def parts(path, base):
    return Path(path).relative_to(ROOT / base).parts


rows = []
for path in glob.glob(str(ROOT / "results/probes/*/*/*/layer_*/metrics.json")) + \
        glob.glob(str(ROOT / "results/probes/*/*/*/layer_*/cap_*/metrics.json")):
    p = parts(path, "results/probes")
    d = json.loads(Path(path).read_text())
    folds = [f["macro_fer"] for f in d["fold_metrics"]]
    rows.append((p[0], p[1], p[2], int(p[3][6:]), int(p[4][4:]) if p[4].startswith("cap_") else 0,
                 round(100 * d["macro_fer"], 4), round(100 * d["macro_parsing_error"], 4),
                 round(100 * d["macro_identity_error"], 4), round(100 * st.pstdev(folds), 4),
                 len(d["class_labels"]), ";".join(map(str, d.get("excluded_rare_classes", [])))))
write("probes", ["species", "bird", "condition", "layer", "label_cap (0 = all)", "macro_fer", "parsing", "identity",
                 "fold_sd", "scored_classes", "excluded_rare_classes"], rows)


def knn_rows(base, seed_level):
    out = []
    for path in glob.glob(str(ROOT / base / ("*/*/*/layer_*/seed_*/summary.json" if seed_level else "*/*/*/layer_*/summary.json"))):
        p = parts(path, base)
        rare = {str(c) for c in RARE.get(f"{p[0]}/{p[1]}", [])}
        for row in json.loads(Path(path).read_text())["rows"]:
            purity = st.mean(v for c, v in row["per_class_same_purity"].items() if c not in rare)
            out.append((p[0], p[1], p[2], int(p[3][6:]), *((int(p[4][5:]),) if seed_level else ()), row["k"],
                        round(100 * purity, 4), round(100 * row["macro_same_purity"], 4)))
    return out


write("knn_layers", ["species", "bird", "condition", "layer", "k", "purity", "purity_all_classes"],
      knn_rows("results/knn/review_baselines_all_layers", False))
write("knn_seeds", ["species", "bird", "condition", "layer", "seed", "k", "purity", "purity_all_classes"],
      knn_rows("results/knn/seeds", True))

rows = []
for path in glob.glob(str(ROOT / "results/kmeans/*/*/*/layer_*/metrics.csv")):
    p = parts(path, "results/kmeans")
    r = next(csv.DictReader(open(path)))
    rows.append((p[0], p[1], p[2], int(p[3][6:]), round(float(r["v_measure"]), 5), round(float(r["homogeneity"]), 5),
                 round(float(r["completeness"]), 5), int(r["clusters"]), int(r["frames"])))
write("kmeans", ["species", "bird", "condition", "layer", "v_measure", "homogeneity", "completeness", "clusters", "frames"], rows)
