#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC_DIR = Path("/media/george-vengrovski/disk2/raw_data/european_starling")
DEFAULT_DST_PATH = ROOT / "files" / "annotation jsons" / "european_starling_annotations.json"
EXCLUDED_BIRDS = {"B335", "B336", "B337", "B338"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert European starling wav folders to TinyBird annotations.")
    parser.add_argument("--src_dir", type=Path, default=DEFAULT_SRC_DIR)
    parser.add_argument("--dst_path", type=Path, default=DEFAULT_DST_PATH)
    return parser.parse_args()


def wav_root(src_dir: Path) -> Path:
    path = src_dir / "audio_files" / "3237218"
    assert path.is_dir(), f"Missing European starling wav folder: {path}"
    return path


def duration_ms(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate() * 1000.0


def make_recording(path: Path) -> dict:
    bird_id = path.parent.name
    filename = f"{bird_id}__{path.name}"
    event = {"onset_ms": 0.0, "offset_ms": duration_ms(path), "units": []}
    return {
        "recording": {"filename": filename, "bird_id": bird_id, "detected_vocalizations": 1},
        "detected_events": [event],
    }


def main() -> None:
    args = parse_args()
    paths = [
        path
        for path in sorted(wav_root(args.src_dir.expanduser()).glob("*/*.wav"))
        if path.parent.name not in EXCLUDED_BIRDS
    ]
    assert paths, "No usable European starling wav files found."
    payload = {"metadata": {"units": "ms"}, "recordings": [make_recording(path) for path in paths]}

    dst_path = args.dst_path.expanduser()
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(payload['recordings'])} recordings to {dst_path}")


if __name__ == "__main__":
    main()
