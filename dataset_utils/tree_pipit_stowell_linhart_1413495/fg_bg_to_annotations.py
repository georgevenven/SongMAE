#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert non-Dryad Tree Pipit fg/bg folders plus one-hot CSVs into TinyBird annotations.json."
    )
    parser.add_argument(
        "--src_dir",
        required=True,
        type=Path,
        help="Directory containing csv/, fg/, and bg/ for the Tree Pipit Stowell-Linhart dataset.",
    )
    parser.add_argument("--dst_dir", required=True, type=Path, help="Directory where annotations.json will be written.")
    return parser.parse_args()


def wav_duration_ms(path: Path) -> float:
    with path.open("rb") as handle:
        with wave.open(handle, "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
    return (frames / sample_rate) * 1000.0


def build_audio_map(audio_dir: Path) -> dict[str, Path]:
    members: dict[str, Path] = {}
    for path in audio_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("._") or path.suffix.lower() != ".wav":
            continue
        members[path.name] = path
    return members


def resolve_bird_id(row: dict[str, str], label_columns: list[str], filename: str) -> str:
    active = []
    for bird_id in label_columns:
        value = (row.get(bird_id) or "").strip()
        if value not in {"", "0", "0.0"}:
            active.append(bird_id)
    if len(active) != 1:
        raise ValueError(f"Expected exactly one active bird_id for {filename}, found {active}")
    return active[0]


def add_or_merge(recordings_by_filename: dict[str, dict[str, object]], recording: dict[str, object]) -> None:
    filename = recording["recording"]["filename"]
    existing = recordings_by_filename.get(filename)
    if existing is None:
        recordings_by_filename[filename] = recording
        return
    if existing["recording"]["bird_id"] != recording["recording"]["bird_id"]:
        raise ValueError(f"Conflicting bird_id for {filename}")
    existing["recording"]["detected_vocalizations"] = len(existing["detected_events"])


def main() -> None:
    args = parse_args()
    src_dir = args.src_dir.expanduser().resolve()
    csv_dir = src_dir / "csv"
    fg_dir = src_dir / "fg"
    bg_dir = src_dir / "bg"
    assert csv_dir.is_dir(), f"Missing csv directory: {csv_dir}"
    assert fg_dir.is_dir(), f"Missing fg directory: {fg_dir}"
    assert bg_dir.is_dir(), f"Missing bg directory: {bg_dir}"

    audio_members = {
        "fg": build_audio_map(fg_dir),
        "bg": build_audio_map(bg_dir),
    }
    recordings_by_filename: dict[str, dict[str, object]] = {}

    for csv_path in sorted(csv_dir.glob("pipit-*.csv")):
        recording_type = "fg" if "-fg-" in csv_path.name else "bg"
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames is not None
            label_columns = [field for field in reader.fieldnames if field != "wavfilename"]

            for row in reader:
                filename = Path((row.get("wavfilename") or "").strip()).name
                if not filename:
                    continue

                bird_id = resolve_bird_id(row, label_columns, filename)
                audio_path = audio_members[recording_type].get(filename)
                if audio_path is None:
                    raise FileNotFoundError(f"Audio file not found for CSV row: {csv_path.name} / {filename}")

                detected_events = []
                if recording_type == "fg":
                    detected_events.append(
                        {
                            "onset_ms": 0.0,
                            "offset_ms": wav_duration_ms(audio_path),
                            "units": [],
                        }
                    )

                add_or_merge(
                    recordings_by_filename,
                    {
                        "recording": {
                            "filename": filename,
                            "bird_id": bird_id,
                            "detected_vocalizations": len(detected_events),
                        },
                        "detected_events": detected_events,
                    },
                )

    payload = {
        "metadata": {
            "units": "ms",
            "species": "Tree Pipit",
        },
        "recordings": sorted(
            recordings_by_filename.values(),
            key=lambda item: (item["recording"]["bird_id"], item["recording"]["filename"]),
        ),
    }
    args.dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = args.dst_dir / "annotations.json"
    with dst_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {len(payload['recordings'])} recordings to {dst_path}")


if __name__ == "__main__":
    main()
