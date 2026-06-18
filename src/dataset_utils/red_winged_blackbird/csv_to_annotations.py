#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC_DIR = Path("/media/george-vengrovski/disk2/raw_data/red_winged_blackbird")
DEFAULT_DST_PATH = ROOT / "files" / "annotation jsons" / "red_winged_blackbird_annotations.json"
DATASET_NAME = "Agelaius_phoeniceus_XC00001-XC00430"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Red-winged blackbird CSV labels to TinyBird annotations.")
    parser.add_argument("--src_dir", type=Path, default=DEFAULT_SRC_DIR)
    parser.add_argument("--dst_path", type=Path, default=DEFAULT_DST_PATH)
    return parser.parse_args()


def csv_paths(src_dir: Path) -> list[Path]:
    base = src_dir / "data" / "zenodo_17958514" / "Agelaius_phoeniceus_data" / DATASET_NAME
    paths = []
    for split in ("train", "val"):
        paths += sorted((base / f"{DATASET_NAME}_{split}").glob("*.wav.csv"))
    assert paths, f"No label CSV files found under {base}"
    return paths


def label_ids(paths: list[Path]) -> dict[str, int]:
    labels = set()
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            labels.update(row["label"] for row in csv.DictReader(handle))
    return {label: index for index, label in enumerate(sorted(labels))}


def make_recording(path: Path, ids: dict[str, int]) -> dict:
    units = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            units.append(
                {
                    "onset_ms": round(float(row["onset_s"]) * 1000.0, 6),
                    "offset_ms": round(float(row["offset_s"]) * 1000.0, 6),
                    "id": ids[row["label"]],
                }
            )

    assert units, path
    filename = path.name.removesuffix(".csv")
    return {
        "recording": {
            "filename": filename,
            "detected_vocalizations": len(units),
        },
        "detected_events": [
            {
                "onset_ms": units[0]["onset_ms"],
                "offset_ms": units[-1]["offset_ms"],
                "units": units,
            }
        ],
    }


def main() -> None:
    args = parse_args()
    paths = csv_paths(args.src_dir.expanduser())
    ids = label_ids(paths)
    recordings = [make_recording(path, ids) for path in sorted(paths, key=lambda path: path.name)]
    payload = {"metadata": {"units": "ms"}, "recordings": recordings}

    dst_path = args.dst_path.expanduser()
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {len(recordings)} recordings to {dst_path}")


if __name__ == "__main__":
    main()
