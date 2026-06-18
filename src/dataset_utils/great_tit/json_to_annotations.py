#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC_DIR = Path("/media/george-vengrovski/disk2/raw_data/great_tit")
DEFAULT_DST_PATH = ROOT / "files" / "annotation jsons" / "great_tit_annotations.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Great tit JSON sidecars to TinyBird annotations.")
    parser.add_argument("--src_dir", type=Path, default=DEFAULT_SRC_DIR)
    parser.add_argument("--dst_path", type=Path, default=DEFAULT_DST_PATH)
    return parser.parse_args()


def load_song_rows(src_dir: Path) -> list[tuple[str, dict]]:
    paths = sorted(
        (src_dir / "data" / "songs").glob("*/JSON/*.JSON"),
        key=lambda path: path.with_suffix(".wav").name,
    )
    assert paths, f"No song JSON files found under {src_dir}"
    return [(path.with_suffix(".wav").name, json.loads(path.read_text())) for path in paths]


def make_recording(filename: str, row: dict, label_ids: dict[str, int]) -> dict:
    bird_id = row["ID"]
    song_type = row["class_id"]
    onsets = row["onsets"]
    offsets = row["offsets"]
    assert len(onsets) == len(offsets), filename

    units = [
        {
            "onset_ms": round(onset * 1000.0, 6),
            "offset_ms": round(offset * 1000.0, 6),
            "id": label_ids[song_type],
        }
        for onset, offset in zip(onsets, offsets)
    ]

    return {
        "recording": {
            "filename": filename,
            "bird_id": bird_id,
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
    src_dir = args.src_dir.expanduser()
    dst_path = args.dst_path.expanduser()
    assert src_dir.is_dir(), f"Missing source directory: {src_dir}"

    song_rows = load_song_rows(src_dir)
    labels = sorted({row["class_id"] for _, row in song_rows})
    label_ids = {label: index for index, label in enumerate(labels)}
    recordings = [make_recording(filename, row, label_ids) for filename, row in song_rows]

    payload = {
        "metadata": {"units": "ms"},
        "recordings": recordings,
    }

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {len(recordings)} recordings to {dst_path}")


if __name__ == "__main__":
    main()
