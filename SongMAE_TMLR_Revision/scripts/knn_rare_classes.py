#!/usr/bin/env python3
"""Classes present in fewer than two recordings within each bird's kNN data selection (first 200k timebins of events).

kNN excludes same-recording neighbours, so such a class has no valid same-class neighbour and scores zero for every
model. Aggregation drops them (the probe drops classes in fewer recordings than folds). Writes results/knn_rare_classes.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from src.core.extract_embedding import load_recording_segments  # noqa: E402

SPECS = {"canary": "canary_5ms", "zf": "zebra_finch_5ms", "bf": "bengalese_finch_5ms"}
out = {}
for species, spec in SPECS.items():
    path = ROOT / "files/annotation jsons" / f"{species}_annotations.json"
    recordings = json.loads(path.read_text())["recordings"]
    units = {Path(r["recording"]["filename"]).stem: [u["id"] + 1 for e in r.get("detected_events", []) for u in e["units"]]
             for r in recordings}
    for bird in sorted({r["recording"]["bird_id"] for r in recordings}):
        segments = load_recording_segments({"spec_dir": f"/media/george-vengrovski/disk2/specs/{spec}", "json_path": str(path),
                                            "bird": bird, "recording_mode": "events", "num_timebins": 200000})["segments"]
        stems = {s["recording_stem"] for s in segments}
        counts = {}
        for stem in stems:
            for c in set(units[stem]):
                counts[c] = counts.get(c, 0) + 1
        rare = sorted(c for c, n in counts.items() if n < 2)
        if rare:
            out[f"{species}/{bird}"] = rare
            print(f"{species}/{bird}: rare {rare} (recordings in selection: {len(stems)})")
(ROOT / "SongMAE_TMLR_Revision/results/knn_rare_classes.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"{len(out)} birds with rare classes")
