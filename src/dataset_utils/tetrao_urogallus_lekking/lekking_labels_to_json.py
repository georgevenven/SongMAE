#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = Path("/home/george-vengrovski/Downloads/lekking_labels.csv")
DEFAULT_DST = ROOT / "files" / "tetrao_urogallus_lekking_annotations.json"
REQUIRED_COLUMNS = {
    "start_time_ms",
    "end_time_ms",
    "song",
    "class",
    "annotation_file",
    "duration_ms",
}


def filename_for_song(song: str) -> str:
    path = Path(song)
    if path.suffix:
        return path.name
    return f"{song}.wav"


def load_rows(csv_path: Path) -> list[dict[str, object]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing columns {sorted(missing)}: {csv_path}")

        rows: list[dict[str, object]] = []
        for row_idx, row in enumerate(reader, start=2):
            song = row["song"].strip()
            label = row["class"].strip()
            annotation_file = row["annotation_file"].strip()
            start_ms = float(row["start_time_ms"])
            end_ms = float(row["end_time_ms"])
            duration_ms = float(row["duration_ms"])

            if not song or not label:
                continue
            if end_ms <= start_ms:
                raise ValueError(f"Row {row_idx} has non-positive event duration")
            if abs((end_ms - start_ms) - duration_ms) > 1e-6:
                raise ValueError(f"Row {row_idx} duration_ms does not match start/end")

            rows.append(
                {
                    "song": song,
                    "label": label,
                    "annotation_file": annotation_file,
                    "onset_ms": start_ms,
                    "offset_ms": end_ms,
                }
            )

    if not rows:
        raise ValueError(f"No usable annotations found in {csv_path}")
    return rows


def build_recordings(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}

    for row in rows:
        song = str(row["song"])
        entry = grouped.get(song)
        if entry is None:
            entry = {
                "recording": {
                    "filename": filename_for_song(song),
                    "detected_vocalizations": 0,
                    "source_annotation_file": row["annotation_file"],
                    "event_class": row["label"],
                },
                "detected_events": [],
            }
            grouped[song] = entry

        event = {
            "onset_ms": row["onset_ms"],
            "offset_ms": row["offset_ms"],
        }
        entry["detected_events"].append(event)  # type: ignore[union-attr]
        entry["recording"]["detected_vocalizations"] += 1  # type: ignore[index]

    recordings = list(grouped.values())
    for recording in recordings:
        recording["detected_events"].sort(key=lambda item: item["onset_ms"])  # type: ignore[union-attr]

    return sorted(recordings, key=lambda item: item["recording"]["filename"])  # type: ignore[index]


def validate_specs(recordings: list[dict[str, object]], spec_dir: Path, strict: bool) -> None:
    spec_stems = {path.stem for path in spec_dir.glob("*.npy")}
    missing = [
        Path(recording["recording"]["filename"]).stem  # type: ignore[index]
        for recording in recordings
        if Path(recording["recording"]["filename"]).stem not in spec_stems  # type: ignore[index]
    ]
    if not missing:
        print(f"All {len(recordings)} recording stems matched specs in {spec_dir}")
        return

    message = "Missing spec stems:\n" + "\n".join(f"  {stem}" for stem in missing)
    if strict:
        raise FileNotFoundError(message)
    print(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Tetrao urogallus lekking CSV labels to TinyBird JSON.")
    parser.add_argument("--csv_path", default=DEFAULT_CSV, type=Path, help="Input lekking_labels.csv path.")
    parser.add_argument("--dst_path", default=DEFAULT_DST, type=Path, help="Output annotations JSON path.")
    parser.add_argument("--spec_dir", type=Path, help="Optional spec directory to validate recording stem matches.")
    parser.add_argument("--strict_spec_match", action="store_true", help="Fail if --spec_dir is missing any CSV songs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    rows = load_rows(args.csv_path)
    recordings = build_recordings(rows)
    payload = {
        "metadata": {
            "units": "ms",
            "species": "Tetrao urogallus",
            "annotation_type": "lekking_events",
        },
        "recordings": recordings,
    }

    args.dst_path.parent.mkdir(parents=True, exist_ok=True)
    with args.dst_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(f"Wrote {len(rows)} events across {len(recordings)} recordings to {args.dst_path}")
    if args.spec_dir is not None:
        validate_specs(recordings, args.spec_dir, args.strict_spec_match)


if __name__ == "__main__":
    main()
