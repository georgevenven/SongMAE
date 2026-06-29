#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from pathlib import Path


ANNOTATIONS = Path("/media/george-vengrovski/disk1/data/XCL/XCL_train_annotations.json")
BOXES = {
    "Canary Islands": [-18.25, 27.60, -13.30, 29.45],
    "Azores": [-31.35, 36.90, -24.70, 39.80],
}


def location(recording):
    lat = recording.get("lat")
    lon = recording.get("long")
    if lat is None or lon is None:
        return None
    lat = float(lat)
    lon = float(lon)
    if math.isnan(lat) or math.isnan(lon):
        return None
    return lat, lon


def in_box(lat, lon, box):
    min_lon, min_lat, max_lon, max_lat = box
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    args = parser.parse_args()

    data = json.loads(args.annotations.read_text())

    writer = csv.writer(sys.stdout)
    writer.writerow(["island_set", "filename", "lat", "long"])
    for row in data["recordings"]:
        recording = row["recording"]
        loc = location(recording)
        if loc is None:
            continue
        lat, lon = loc
        for island_set, box in BOXES.items():
            if in_box(lat, lon, box):
                writer.writerow([island_set, recording["filename"], lat, lon])


if __name__ == "__main__":
    main()
