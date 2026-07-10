#!/usr/bin/env python3
"""Generate the SongMAE supplemental temporal-resolution table.

Values are percentages of nonnegative inter-syllable gaps shorter than each
resolution. The mean weights each species equally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SPECIES = [
    ("Canary", Path("files/annotation jsons/canary_annotations.json")),
    ("Zebra", Path("files/annotation jsons/zf_annotations.json")),
    ("Bengalese", Path("files/annotation jsons/bf_annotations.json")),
]
RESOLUTIONS_MS = (1, 2, 5, 10, 20, 200)


def load_gaps(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metadata"]["units"] == "ms"

    gaps = []
    for rec in data["recordings"]:
        for event in rec["detected_events"]:
            units = sorted(event["units"], key=lambda unit: unit["onset_ms"])
            for left, right in zip(units, units[1:]):
                gap = right["onset_ms"] - left["offset_ms"]
                if gap >= 0:
                    gaps.append(gap)
    assert gaps
    return gaps


def percentage_below(gaps, resolution_ms):
    return 100 * sum(gap < resolution_ms for gap in gaps) / len(gaps)


def format_percentage(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


def print_table(gaps_by_species, markdown):
    headers = ["Temporal resolution (ms)"] + [f"{name} (%)" for name, _ in SPECIES] + ["Mean (%)"]
    rows = []
    for resolution_ms in RESOLUTIONS_MS:
        values = [percentage_below(gaps_by_species[name], resolution_ms) for name, _ in SPECIES]
        values.append(sum(values) / len(values))
        rows.append([f"{resolution_ms:g}"] + [format_percentage(value) for value in values])

    if not markdown:
        print("\t".join(headers))
        for row in rows:
            print("\t".join(row))
        return

    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main():
    parser = argparse.ArgumentParser(
        description="Print the percent of inter-syllable gaps below each temporal resolution."
    )
    parser.add_argument("--format", choices=["tsv", "markdown"], default="tsv")
    args = parser.parse_args()

    gaps_by_species = {name: load_gaps(path) for name, path in SPECIES}
    print_table(gaps_by_species, args.format == "markdown")


if __name__ == "__main__":
    main()
