#!/usr/bin/env python3
"""Table 2 on canary, the model-selection species: kNN purity of the SongMAE-Micro sweep -> latex/blocks/table-2.tex.

k = 100, final encoder layer (L5), same-recording neighbours excluded, classes in fewer than two recordings of a bird's
kNN selection dropped (results/knn_rare_classes.json). Each run is averaged over the 3 canary birds; each row is the
mean ± SD over pretraining seeds. Runs without results yet are skipped, so rerun as seeds finish.
"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KNN = ROOT / "results/knn/review_baselines_all_layers/canary"
REV = ROOT / "SongMAE_TMLR_Revision"
RARE = json.loads((REV / "results/knn_rare_classes.json").read_text())
SILENCE = "0"
P5, P20 = "32 mels × 5 ms", "32 mels × 20 ms"


def seeds(first, stem):
    return [first, f"{stem}_seed1", f"{stem}_seed2"]


GROUPS = [
    [("Random", P5, "--", seeds("xcl_micro_100k_p32x1_random", "xcl_micro_100k_p32x1_random")),
     ("Voronoi", P5, "10.0", seeds("Xcl_micro_100k_p32x1_c010", "xcl_micro_100k_p32x1_c010"))],
    [("Voronoi", "128 mels × 5 ms", "10.0", seeds("Xcl_micro_100k_p128x1_default", "xcl_micro_100k_p128x1_c010")),
     ("Voronoi", "16 mels × 5 ms", "10.0", seeds("Xcl_micro_100k_p16x1_default", "xcl_micro_100k_p16x1_c010")),
     ("Voronoi", P20, "10.0", seeds("xcl_micro_100k_p32x4_c010", "xcl_micro_100k_p32x4_c010")),
     ("Voronoi", "4 mels × 20 ms", "10.0", seeds("Xcl_micro_100k_p4x4_default", "xcl_micro_100k_p4x4_c010"))],
    [("Voronoi", P5, "2.5", seeds("Xcl_micro_100k_p32x1_c0025", "xcl_micro_100k_p32x1_c0025")),
     ("Voronoi", P5, "5.0", seeds("Xcl_micro_100k_p32x1_c005", "xcl_micro_100k_p32x1_c005")),
     ("Time", P5, "5.0", [f"xcl_micro_100k_p32x1_time_seed{s}" for s in range(3)]),
     ("Frequency", P5, "--", [f"xcl_micro_100k_p32x1_frequency_seed{s}" for s in range(3)])],
    [("Voronoi", P20, "2.5", seeds("xcl_micro_100k_p32x4_c0025", "xcl_micro_100k_p32x4_c0025")),
     ("Voronoi", P20, "5.0", seeds("xcl_micro_100k_p32x4_c005", "xcl_micro_100k_p32x4_c005"))],
]


def run(condition):
    """(all, vocal, silence) purity in %, averaged over canary birds; None if any bird is missing."""
    birds = []
    for bird in sorted(KNN.iterdir()):
        path = bird / condition / "layer_5/summary.json"
        if not path.exists():
            return None
        row = next(r for r in json.loads(path.read_text())["rows"] if r["k"] == 100)
        rare = {str(c) for c in RARE.get(f"canary/{bird.name}", [])}
        purity = {c: v for c, v in row["per_class_same_purity"].items() if c not in rare}
        birds.append((st.mean(purity.values()), st.mean(v for c, v in purity.items() if c != SILENCE), purity[SILENCE]))
    return [100 * st.mean(values) for values in zip(*birds)]


rows = []
for group in GROUPS:
    for masking, patch, seed_pct, runs in group:
        values = [v for v in map(run, runs) if v]
        if values:
            stats = [(st.mean(m), st.stdev(m) if len(m) > 1 else None) for m in zip(*values)]
            rows.append((masking, patch, seed_pct, len(values), stats))
    rows.append(None)
best = [max(r[4][i][0] for r in rows if r) for i in range(3)]


def cell(mean, sd, top):
    text = f"{mean:.1f}" + (f" $\\pm$ {sd:.1f}" if sd is not None else "")
    return f"\\textbf{{{text}}}" if round(mean, 1) == round(top, 1) else text


body = []
for r in rows[:-1]:
    if r is None:
        body.append("\\addlinespace")
        continue
    masking, patch, seed_pct, n, stats = r
    body.append(" & ".join([masking, patch, seed_pct, str(n)] + [cell(m, s, b) for (m, s), b in zip(stats, best)]) + " \\\\")

(REV / "latex/blocks/table-2.tex").write_text(r"""\noindent

\begin{minipage}{\linewidth}
\phantomsection\label{tab:2}
\noindent %CAPTION%
\par\nopagebreak\vspace{0.5\baselineskip}
\begingroup
\setlength{\tabcolsep}{4pt}
\centering
\small
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llccccc@{}}
\toprule
& & & & \multicolumn{3}{c}{\textbf{Canary kNN purity (\%)}} \\
\cmidrule(lr){5-7}
\textbf{Masking} & \textbf{Patch size} & Seed patches (\%) & Seeds & All & Vocal & Silence \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular*}
\endgroup

\end{minipage}
\par\vspace{0.5\baselineskip}
""")
print("\n".join(body))
