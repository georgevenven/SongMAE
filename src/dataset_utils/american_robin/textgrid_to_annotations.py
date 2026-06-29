#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC_DIR = Path("/media/george-vengrovski/disk2/raw_data/american_robin")
DEFAULT_DST_PATH = ROOT / "files" / "annotation jsons" / "american_robin_annotations.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RMBL American robin TextGrid labels to TinyBird annotations.")
    parser.add_argument("--src_dir", type=Path, default=DEFAULT_SRC_DIR)
    parser.add_argument("--dst_path", type=Path, default=DEFAULT_DST_PATH)
    return parser.parse_args()


def textgrid_paths(src_dir: Path) -> list[Path]:
    paths = sorted((src_dir / "data" / "rmbl_robin" / "RMBL-Robin" / "data").glob("*.TextGrid"))
    assert paths, f"No TextGrid files found under {src_dir}"
    return paths


def intervals(path: Path) -> dict[str, list[dict]]:
    tiers = {}
    tier = None
    item = {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("name ="):
            tier = line.split("=", 1)[1].strip().strip('"').replace(" ", "-")
            tiers.setdefault(tier, [])
        elif tier and line.startswith("xmin ="):
            item = {"onset_ms": round(float(line.split("=", 1)[1]) * 1000.0, 6)}
        elif tier and item and line.startswith("xmax ="):
            item["offset_ms"] = round(float(line.split("=", 1)[1]) * 1000.0, 6)
        elif tier and item and line.startswith("text ="):
            item["text"] = line.split("=", 1)[1].strip().strip('"')
            tiers[tier].append(item)
            item = {}
    assert "song" in tiers and "syllable-quality" in tiers, path
    return tiers


def label_ids(paths: list[Path]) -> dict[str, int]:
    labels = {
        syllable_label(interval)
        for path in paths
        for interval in intervals(path)["syllable-quality"]
        if interval["text"]
    }
    return {label: index for index, label in enumerate(sorted(labels, key=int))}


def syllable_label(interval: dict) -> str:
    return interval["text"].split("-", 1)[0]


def unit(interval: dict, ids: dict[str, int]) -> dict:
    item = {
        "onset_ms": interval["onset_ms"],
        "offset_ms": interval["offset_ms"],
        "id": ids[syllable_label(interval)],
    }
    parts = interval["text"].split("-", 1)
    if len(parts) == 2:
        item["quality"] = int(parts[1])
    return item


def make_recording(path: Path, ids: dict[str, int]) -> dict:
    data = intervals(path)
    songs = [interval for interval in data["song"] if interval["text"]]
    syllables = [interval for interval in data["syllable-quality"] if interval["text"]]
    used = set()
    events = []

    for song in songs:
        indexes = [
            index
            for index, syllable in enumerate(syllables)
            if syllable["onset_ms"] >= song["onset_ms"] - 0.001
            and syllable["offset_ms"] <= song["offset_ms"] + 0.001
        ]
        assert indexes, path
        used.update(indexes)
        events.append(
            {
                "onset_ms": song["onset_ms"],
                "offset_ms": song["offset_ms"],
                "units": [unit(syllables[index], ids) for index in indexes],
            }
        )

    for index, syllable in enumerate(syllables):
        if index in used:
            continue
        events.append(
            {
                "onset_ms": syllable["onset_ms"],
                "offset_ms": syllable["offset_ms"],
                "units": [unit(syllable, ids)],
            }
        )

    events.sort(key=lambda event: event["onset_ms"])
    filename = path.with_suffix(".WAV").name
    return {
        "recording": {
            "filename": filename,
            "bird_id": filename.split("-", 1)[0],
            "detected_vocalizations": len(events),
        },
        "detected_events": events,
    }


def main() -> None:
    args = parse_args()
    paths = textgrid_paths(args.src_dir.expanduser())
    ids = label_ids(paths)
    payload = {
        "metadata": {
            "units": "ms",
            "unit_label_type": "american_robin_syllable_pattern",
            "unit_id_to_label": {str(index): label for label, index in ids.items()},
        },
        "recordings": [make_recording(path, ids) for path in paths],
    }

    dst_path = args.dst_path.expanduser()
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {len(payload['recordings'])} recordings to {dst_path}")


if __name__ == "__main__":
    main()
