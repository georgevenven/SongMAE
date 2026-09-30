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


def spans_for_width(segments, bin_ms, audio_params):
    spans = []
    for segment in segments:
        duration = timebins_to_ms(segment["spectrogram"].shape[1], audio_params)
        for k in range(int(np.ceil(duration / bin_ms - 1e-9))):
            start, end = k * bin_ms, min((k + 1) * bin_ms, duration)
            spans.append((segment["recording_stem"], int(np.rint(segment["start_ms"] + start)),
                          int(np.rint(segment["start_ms"] + end))))
    return spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in "spec_dir annotations bird".split():
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--bin_ms", default="1,2,5,20,40,80,160", help="output bin widths in ms")
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
    for bin_ms in (float(w) for w in args.bin_ms.split(",")):
        spans = spans_for_width(extracted["segments"], bin_ms, extracted["audio_params"])
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
            rows.append({"oracle": oracle, "bin_ms": bin_ms, "bins": len(spans), **row})
    print(json.dumps({"bird": args.bird, "segments": len(extracted["segments"]), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
