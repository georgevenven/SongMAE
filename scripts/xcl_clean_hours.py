#!/usr/bin/env python3
import json
from pathlib import Path


SPLITS = [
    ("Train", Path("/media/george-vengrovski/disk1/data/XCL_clean")),
    ("Test", Path("/media/george-vengrovski/disk1/data/XCL_val_clean")),
]


def split_stats(root):
    params = json.loads((root / "audio_params.json").read_text())
    lines = (root / "shards" / "index.tsv").read_text().splitlines()
    assert lines[0] == "name\tshard\tstart\tend"

    rows = [line.split("\t") for line in lines[1:]]
    timebins = sum(int(end) - int(start) for _, _, start, end in rows)
    hours = timebins * params["hop_size"] / params["sr"] / 3600
    return len(rows), hours


def main():
    rows = [(name, *split_stats(root)) for name, root in SPLITS]
    print("Split\tRecordings\tHours")
    for name, recordings, hours in rows:
        print(f"{name}\t{recordings}\t{hours:.2f}")
    print(f"Total\t{sum(row[1] for row in rows)}\t{sum(row[2] for row in rows):.2f}")


if __name__ == "__main__":
    main()
