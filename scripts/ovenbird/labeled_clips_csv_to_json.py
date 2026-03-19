"""
Convert labeled clip CSV annotations into TinyBird JSON.

Expected CSV columns:
  localization_event_id,array,event_timestamp,bird_position_x,bird_position_y,
  distance_to_mic_m,aiid_label,data_split,clip_name,song_center_time,start_time,
  end_time,file

The output format matches the clip-level TinyBird annotation structure:

{
  "metadata": {"units": "ms"},
  "recordings": [
    {
      "recording": {
        "filename": "1946_oven20.mp3",
        "bird_id": "20",
        "detected_vocalizations": 1,
        ...
      },
      "detected_events": [
        {"onset_ms": 4000.0, "offset_ms": 6000.0}
      ]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_COLUMNS = {
    "aiid_label",
    "clip_name",
    "start_time",
    "end_time",
}


def _to_float(value: str, field: str, row_idx: int) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {field!r} on row {row_idx}: {value!r}") from exc


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing required columns {sorted(missing)}: {csv_path}")

        rows = []
        for row_idx, row in enumerate(reader, start=2):
            clip_name = (row.get("clip_name") or "").strip()
            bird_id = (row.get("aiid_label") or "").strip()
            if not clip_name or not bird_id:
                continue

            start_s = _to_float((row.get("start_time") or "").strip(), "start_time", row_idx)
            end_s = _to_float((row.get("end_time") or "").strip(), "end_time", row_idx)
            if end_s < start_s:
                start_s, end_s = end_s, start_s

            rows.append(
                {
                    "clip_name": clip_name,
                    "bird_id": bird_id,
                    "start_s": start_s,
                    "end_s": end_s,
                    "array": (row.get("array") or "").strip(),
                    "event_timestamp": (row.get("event_timestamp") or "").strip(),
                    "bird_position_x": (row.get("bird_position_x") or "").strip(),
                    "bird_position_y": (row.get("bird_position_y") or "").strip(),
                    "distance_to_mic_m": (row.get("distance_to_mic_m") or "").strip(),
                    "data_split": (row.get("data_split") or "").strip(),
                    "song_center_time": (row.get("song_center_time") or "").strip(),
                    "localization_event_id": (row.get("localization_event_id") or "").strip(),
                    "source_file": Path((row.get("file") or "").strip()).name,
                }
            )
    if not rows:
        raise ValueError(f"No usable rows found in {csv_path}")
    return rows


def build_recordings(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}

    for row in rows:
        clip_name = row["clip_name"]
        entry = grouped.get(clip_name)
        if entry is None:
            recording = {
                "filename": clip_name,
                "bird_id": row["bird_id"],
                "detected_vocalizations": 0,
            }

            # Preserve useful source metadata without changing the core JSON layout.
            optional_fields = [
                "array",
                "event_timestamp",
                "bird_position_x",
                "bird_position_y",
                "distance_to_mic_m",
                "data_split",
                "song_center_time",
                "localization_event_id",
                "source_file",
            ]
            for field in optional_fields:
                value = row[field]
                if value != "":
                    recording[field] = value

            entry = {"recording": recording, "detected_events": []}
            grouped[clip_name] = entry

        event = {
            "onset_ms": 1000.0 * row["start_s"],
            "offset_ms": 1000.0 * row["end_s"],
        }
        entry["detected_events"].append(event)
        entry["recording"]["detected_vocalizations"] += 1

    return [grouped[key] for key in sorted(grouped)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ovenbird labeled clip CSV into TinyBird JSON.")
    parser.add_argument("--csv_path", required=True, type=Path, help="Input CSV path.")
    parser.add_argument("--dst_dir", required=True, type=Path, help="Directory where annotations.json will be written.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path: Path = args.csv_path
    dst_dir: Path = args.dst_dir

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    payload = {
        "metadata": {"units": "ms"},
        "recordings": build_recordings(rows),
    }

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / "annotations.json"
    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {dst_path}")


if __name__ == "__main__":
    main()
