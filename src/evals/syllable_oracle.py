#!/usr/bin/env python3
"""Oracle Macro FER at fixed output resolutions: each output bin predicts one ground-truth-derived label.

`majority`: the bin's majority label. `macro_optimal`: the label maximizing (class frames in bin) / (class frames
overall); bins are independent and Macro FER is additive over frames with weight 1 / class frames, so this is the
lowest Macro FER any model with that output grid can reach. (Majority is not a bound: it never predicts classes
shorter than half a bin, which Macro FER weights equally.)

Uses the probe's data selection (event segments, balanced events, timebin cap) and scoring (1 ms expansion,
parsing/identity split).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.extract_embedding import load_recording_segments
from src.core.utils import timebins_to_ms
from src.evals.syllable_classification import confusion_matrix, ground_truth, load_units, metrics


def spans_for_width(segments, width, audio_params):
    spans = []
    for segment in segments:
        n = segment["spectrogram"].shape[1]
        for start in range(0, n, width):
            end = min(start + width, n)
            spans.append((
                segment["recording_stem"],
                int(np.rint(segment["start_ms"] + timebins_to_ms(start, audio_params))),
                int(np.rint(segment["start_ms"] + timebins_to_ms(end, audio_params))),
            ))
    return spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in "spec_dir annotations bird".split():
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--widths", default="1,4,8,16,32", help="output bin widths in spectrogram timebins")
    parser.add_argument("--num_timebins", type=int, default=720000)
    parser.add_argument("--balanced_events", type=int, default=3)
    parser.add_argument("--event_seed", type=int, default=42)
    args = parser.parse_args()

    extracted = load_recording_segments({
        "spec_dir": args.spec_dir, "json_path": args.annotations, "bird": args.bird, "recording_mode": "events",
        "num_timebins": args.num_timebins, "balanced_events": args.balanced_events, "event_seed": args.event_seed,
    })
    units = load_units(args.annotations)
    rows = []
    for width in (int(w) for w in args.widths.split(",")):
        spans = spans_for_width(extracted["segments"], width, extracted["audio_params"])
        truths = [ground_truth(units, *span) for span in spans]
        labels = sorted(set(np.concatenate(truths).tolist()))
        counts = [np.bincount(truth, minlength=labels[-1] + 1) for truth in truths]
        frames = np.sum(counts, axis=0)
        weights = np.divide(1.0, frames, out=np.zeros(len(frames)), where=frames > 0)
        for oracle, score in [("majority", lambda c: c), ("macro_optimal", lambda c: c * weights)]:
            predictions = [int(score(c).argmax()) for c in counts]
            row = metrics(labels, confusion_matrix(predictions, spans, units, labels))
            for key in ("class_labels", "confusion_matrix", "per_class"):
                del row[key]
            rows.append({"oracle": oracle, "width_timebins": width,
                         "bin_ms": timebins_to_ms(width, extracted["audio_params"]), "bins": len(spans), **row})
    print(json.dumps({"bird": args.bird, "segments": len(extracted["segments"]), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
