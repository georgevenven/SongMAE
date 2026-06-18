#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ANNOTATIONS = [
    Path("files/annotation jsons/bf_annotations.json"),
    Path("files/annotation jsons/canary_annotations.json"),
    Path("files/annotation jsons/zf_annotations.json"),
]


def percentile(values, q):
    assert values
    values = sorted(values)
    pos = (len(values) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def load_gaps(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metadata"]["units"] == "ms"

    gaps = []
    overlaps = []
    shortest = []

    for rec in data["recordings"]:
        filename = rec["recording"]["filename"]
        bird_id = rec["recording"].get("bird_id", "")

        for event_idx, event in enumerate(rec["detected_events"]):
            units = sorted(event["units"], key=lambda unit: unit["onset_ms"])
            for left, right in zip(units, units[1:]):
                gap = right["onset_ms"] - left["offset_ms"]
                row = (gap, filename, bird_id, event_idx, left["id"], right["id"])
                if gap < 0:
                    overlaps.append(row)
                else:
                    gaps.append(gap)
                    shortest.append(row)

    return gaps, overlaps, sorted(shortest)[:10]


def print_stats(name, gaps, overlaps, shortest):
    print(f"\n{name}")
    print(f"  gaps: {len(gaps)}")
    print(f"  overlaps: {len(overlaps)}")
    if not gaps:
        return

    print(
        "  ms: "
        f"min={min(gaps):.3f} "
        f"p1={percentile(gaps, 1):.3f} "
        f"p5={percentile(gaps, 5):.3f} "
        f"p10={percentile(gaps, 10):.3f} "
        f"median={percentile(gaps, 50):.3f}"
    )
    for threshold in (0.5, 1, 2, 4, 5, 8, 10, 20):
        count = sum(gap < threshold for gap in gaps)
        print(f"  gaps < {threshold:>4g} ms: {count:>6} ({count / len(gaps):6.2%})")

    print("  shortest:")
    for gap, filename, bird_id, event_idx, left_id, right_id in shortest:
        print(
            f"    {gap:8.3f} ms  {bird_id}  {filename}  "
            f"event={event_idx}  {left_id}->{right_id}"
        )


def main():
    parser = argparse.ArgumentParser(description="Measure adjacent unit gaps in TinyBird annotation JSON files.")
    parser.add_argument("annotations", nargs="*", type=Path, default=DEFAULT_ANNOTATIONS)
    args = parser.parse_args()

    all_gaps = []
    all_overlaps = []
    all_shortest = []

    for path in args.annotations:
        gaps, overlaps, shortest = load_gaps(path)
        print_stats(path.name, gaps, overlaps, shortest)
        all_gaps.extend(gaps)
        all_overlaps.extend(overlaps)
        all_shortest.extend(
            (gap, f"{path.name}:{filename}", bird_id, event_idx, left_id, right_id)
            for gap, filename, bird_id, event_idx, left_id, right_id in shortest
        )

    if len(args.annotations) > 1:
        print_stats("all_annotations", all_gaps, all_overlaps, sorted(all_shortest)[:10])


if __name__ == "__main__":
    main()
