#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC_DIR = Path("/media/george-vengrovski/disk2/raw_data/cassins_vireo")
DEFAULT_DST_PATH = ROOT / "files" / "annotation jsons" / "cassins_vireo_annotations.json"
SONG_LABEL = re.compile(r"^[a-z]{2}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Cassin's vireo TextGrid labels to TinyBird annotations.")
    parser.add_argument("--src_dir", type=Path, default=DEFAULT_SRC_DIR)
    parser.add_argument("--dst_path", type=Path, default=DEFAULT_DST_PATH)
    return parser.parse_args()


def textgrid_paths(src_dir: Path) -> list[Path]:
    base = src_dir / "data" / "figshare_3081814" / "textgrids" / "Textgrids"
    paths = sorted((path for path in base.iterdir() if path.suffix.lower() == ".textgrid"), key=lambda path: int(path.stem))
    assert paths, f"No TextGrid files found under {base}"
    return paths


def bird_ids(src_dir: Path) -> dict[str, str]:
    path = src_dir / "data" / "figshare_3081814" / "gilman_zlavian_2025" / "birdDB.csv"
    assert path.exists(), f"Missing BirdDB metadata: {path}"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            Path(row["Textgrid_file"]).stem: row["SubjectName"]
            for row in rows
            if row["Species_short_name"] == "CAVI" and row["SubjectName"]
        }


def intervals(path: Path) -> list[dict]:
    output = []
    tier = ""
    item = {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("name ="):
            tier = line.split("=", 1)[1].strip().strip('"')
        elif tier.startswith("Cavi") and line.startswith("xmin ="):
            item = {"onset_ms": round(float(line.split("=", 1)[1]) * 1000.0, 6)}
        elif item and line.startswith("xmax ="):
            item["offset_ms"] = round(float(line.split("=", 1)[1]) * 1000.0, 6)
        elif item and line.startswith("text ="):
            item["text"] = line.split("=", 1)[1].strip().strip('"')
            if SONG_LABEL.fullmatch(item["text"]):
                output.append(item)
            item = {}
    assert output, path
    return output


def label_ids(paths: list[Path]) -> dict[str, int]:
    labels = {interval["text"] for path in paths for interval in intervals(path)}
    return {label: index for index, label in enumerate(sorted(labels))}


def make_recording(path: Path, ids: dict[str, int], bird_id: str) -> dict:
    units = [
        {"onset_ms": item["onset_ms"], "offset_ms": item["offset_ms"], "id": ids[item["text"]]}
        for item in intervals(path)
    ]
    events = [{"onset_ms": unit["onset_ms"], "offset_ms": unit["offset_ms"], "units": [unit]} for unit in units]
    return {
        "recording": {
            "filename": path.with_suffix(".wav").name,
            "bird_id": bird_id,
            "detected_vocalizations": len(events),
        },
        "detected_events": events,
    }


def main() -> None:
    args = parse_args()
    src_dir = args.src_dir.expanduser()
    paths = textgrid_paths(src_dir)
    ids_by_file = bird_ids(src_dir)
    paths = [path for path in paths if path.stem in ids_by_file]
    assert paths, "No TextGrid files had BirdDB individual metadata."
    ids = label_ids(paths)
    payload = {
        "metadata": {"units": "ms"},
        "recordings": [make_recording(path, ids, ids_by_file[path.stem]) for path in paths],
    }

    dst_path = args.dst_path.expanduser()
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {len(payload['recordings'])} recordings to {dst_path}")


if __name__ == "__main__":
    main()
