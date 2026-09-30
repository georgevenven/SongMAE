#!/usr/bin/env python3
"""Table 3 (linear-probe Macro FER, leave-one-species-out) from results/tables.md -> latex/blocks/table-3.tex.

Rows missing from tables.md (e.g. BEATs speeds still running) are skipped; rerun scripts/aggregate.py, then this.
Slowed playback is written as the output resolution in original-audio time, with the speed in parentheses.
"""
import re
from pathlib import Path

REV = Path(__file__).resolve().parents[1]
FRAC = {"1×": "", "½×": "1/2", "¼×": "1/4", "⅛×": "1/8", "1/16×": "1/16", "1/32×": "1/32"}
GROUPS = [  # model, base resolution (ms), tables.md labels
    ("SongMAE-Large", None, ["SongMAE-L 32×5 ms", "SongMAE-L 32×20 ms"]),
    ("BirdAVES", 20, ["BirdAVES 1×", "BirdAVES ½×", "BirdAVES ¼×"]),
    ("HuBERT", 20, ["HuBERT"]),
    ("BEATs", 160, [f"BEATs {s}" for s in FRAC]),
    ("Bird-MAE", 160, [f"Bird-MAE {s}" for s in FRAC]),
]

text = (REV / "results/tables.md").read_text()
section = text.split("## Linear-probe Macro FER (%), leave-one-species-out")[1].split("\n## ")[0]
rows = {}
for line in section.splitlines():
    cells = [c.strip() for c in line.strip("|").split("|")]
    if len(cells) == 9 and re.match(r"[\d.]+ ±", cells[1]):
        species = [float(c.split(" ±")[0]) for c in cells[1:4]]
        rows[cells[0]] = species + [float(cells[4]), float(cells[6]), float(cells[7])]


def resolution(label, base):
    if base is None:
        return label.split("32×")[1]
    speed = label.split(" ")[-1] if " " in label.replace("Bird-MAE", "BirdMAE") else "1×"
    ms = base * eval(speed[:-1].replace("½", "1/2").replace("¼", "1/4").replace("⅛", "1/8"))
    return f"{ms:g} ms" + (f" ({FRAC[speed]}×)" if FRAC[speed] else "")


table = [(model, resolution(label, base), rows[label]) for model, base, labels in GROUPS for label in labels if label in rows]
best = [min(r[2][i] for r in table) for i in range(6)]
body, previous = [], None
for model, res, values in table:
    if previous and model != previous:
        body.append("\\addlinespace")
    cells = [f"\\textbf{{{v:.2f}}}" if v == b else f"{v:.2f}" for v, b in zip(values, best)]
    body.append(" & ".join([model if model != previous else "", res] + cells) + " \\\\")
    previous = model

block = (REV / "latex/blocks/table-3.tex").read_text()
start, end = block.index("\\midrule\n") + len("\\midrule\n"), block.index("\\bottomrule")
block = block[:start] + "\n".join(body) + "\n" + block[end:]
(REV / "latex/blocks/table-3.tex").write_text(block.replace(r"\textbf{Patch size}", r"\textbf{Temporal resolution}"))
print("\n".join(body))
