#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC_DIR = Path("/media/george-vengrovski/disk2/raw_data/white_crowned_sparrow")
DEFAULT_DST_DIR = ROOT / "files" / "annotation jsons"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert White-crowned sparrow Raven tables to TinyBird annotations.")
    parser.add_argument("--src_dir", type=Path, default=DEFAULT_SRC_DIR)
    parser.add_argument("--dst_dir", type=Path, default=DEFAULT_DST_DIR)
    parser.add_argument("--kind", choices=("manual", "predicted", "all"), default="all")
    return parser.parse_args()


def annotation_dir(src_dir: Path, name: str) -> Path:
    path = src_dir / "data" / "dryad_tx95x6b4j" / "annotations" / name
    assert path.is_dir(), f"Missing annotation directory: {path}"
    return path


def rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("Begin Time (s)") != "Begin Time (s)"
        ]


def label_ids(paths: list[Path]) -> dict[str, int]:
    labels = {row[type_key(row)] for path in paths for row in rows(path)}
    return {label: index for index, label in enumerate(sorted(labels))}


def type_key(row: dict[str, str]) -> str:
    return "TYPE" if "TYPE" in row else "type"


def unit(row: dict[str, str], ids: dict[str, int]) -> dict:
    return {
        "onset_ms": round(float(row["Begin Time (s)"]) * 1000.0, 6),
        "offset_ms": round(float(row["End Time (s)"]) * 1000.0, 6),
        "id": ids[row[type_key(row)]],
    }


def manual_filename(path: Path) -> str:
    return path.name.split(".wav_", 1)[0] + ".wav"


def predicted_filename(row: dict[str, str], path: Path) -> str:
    name = row.get("Begin File", "")
    if name:
        return Path(name).name
    return path.name.split(".selections.MASTER", 1)[0] + ".wav"


def make_recording(filename: str, events: list[dict]) -> dict:
    return {
        "recording": {
            "filename": filename,
            "detected_vocalizations": sum(len(event["units"]) for event in events),
        },
        "detected_events": events,
    }


def build_manual(src_dir: Path) -> dict:
    paths = sorted(annotation_dir(src_dir, "ManualAnnotations").glob("*.gz"))
    ids = label_ids(paths)
    grouped = {}
    for path in paths:
        units = [unit(row, ids) for row in rows(path)]
        assert units, path
        event = {
            "onset_ms": units[0]["onset_ms"],
            "offset_ms": units[-1]["offset_ms"],
            "units": units,
        }
        grouped.setdefault(manual_filename(path), []).append(event)

    recordings = []
    for filename, events in sorted(grouped.items()):
        events.sort(key=lambda event: event["onset_ms"])
        recordings.append(make_recording(filename, events))
    return {"metadata": {"units": "ms"}, "recordings": recordings}


def build_predicted(src_dir: Path) -> dict:
    paths = sorted(annotation_dir(src_dir, "PredictedAnnotations").glob("*.gz"))
    ids = label_ids(paths)
    recordings = []
    for path in paths:
        data = rows(path)
        units = [unit(row, ids) for row in data]
        assert units, path
        events = [
            {
                "onset_ms": units[0]["onset_ms"],
                "offset_ms": units[-1]["offset_ms"],
                "units": units,
            }
        ]
        recordings.append(make_recording(predicted_filename(data[0], path), events))
    return {"metadata": {"units": "ms"}, "recordings": recordings}


def write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {len(payload['recordings'])} recordings to {path}")


def main() -> None:
    args = parse_args()
    src_dir = args.src_dir.expanduser()
    dst_dir = args.dst_dir.expanduser()
    if args.kind in ("manual", "all"):
        write(build_manual(src_dir), dst_dir / "white_crowned_sparrow_manual_annotations.json")
    if args.kind in ("predicted", "all"):
        write(build_predicted(src_dir), dst_dir / "white_crowned_sparrow_predicted_annotations.json")


if __name__ == "__main__":
    main()
