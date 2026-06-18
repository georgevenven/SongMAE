#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_ROOT = Path("/media/george-vengrovski/disk2/raw_data")
DEFAULT_DST_DIR = ROOT / "files" / "annotation jsons"
CSV_NAME_RE = re.compile(r"^(?P<code>[a-z]+)-.+-(?P<kind>fg|bg)-.+\.csv$")
SPECIES = {
    "chiffchaff": (
        "chiffchaff_annotations.json",
        RAW_DATA_ROOT / "chiffchaff" / "audio_files" / "stowell_linhart_1413495",
    ),
    "littleowl": (
        "little_owl_annotations.json",
        RAW_DATA_ROOT / "little_owl" / "audio_files" / "stowell_linhart_1413495",
    ),
    "pipit": (
        "tree_pipit_annotations.json",
        RAW_DATA_ROOT / "tree_pipit" / "audio_files" / "stowell_linhart_1413495",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Stowell-Linhart 1413495 labels to TinyBird annotations.")
    parser.add_argument("--dst_dir", type=Path, default=DEFAULT_DST_DIR)
    return parser.parse_args()


def wav_duration_ms(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate() * 1000.0


def wavs(path: Path) -> dict[str, Path]:
    return {wav.name: wav for wav in path.rglob("*.wav") if not wav.name.startswith("._")}


def bird_id(row: dict[str, str], columns: list[str], filename: str) -> str:
    active = [column for column in columns if row.get(column, "").strip() not in {"", "0", "0.0"}]
    assert len(active) == 1, f"Expected one active bird_id for {filename}, found {active}"
    return active[0]


def load_species(path: Path, code: str) -> list[dict]:
    recordings = {}
    audio = {"fg": wavs(path / "fg"), "bg": wavs(path / "bg")}
    for csv_path in sorted((path / "csv").glob(f"{code}-*.csv")):
        match = CSV_NAME_RE.match(csv_path.name)
        assert match, csv_path
        kind = match["kind"]
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = csv.DictReader(handle)
            assert rows.fieldnames and rows.fieldnames[0] == "wavfilename", csv_path
            columns = rows.fieldnames[1:]
            for row in rows:
                filename = Path(row["wavfilename"].strip()).name
                if not filename:
                    continue
                row_bird_id = bird_id(row, columns, filename)
                assert filename in audio[kind], f"Missing wav for {csv_path.name}: {filename}"
                item = recordings.setdefault(
                    filename,
                    {
                        "recording": {
                            "filename": filename,
                            "bird_id": row_bird_id,
                            "detected_vocalizations": 0,
                        },
                        "detected_events": [],
                    },
                )
                assert item["recording"]["bird_id"] == row_bird_id
                if kind == "fg" and not item["detected_events"]:
                    item["detected_events"].append(
                        {
                            "onset_ms": 0.0,
                            "offset_ms": wav_duration_ms(audio[kind][filename]),
                            "units": [],
                        }
                    )
                    item["recording"]["detected_vocalizations"] = 1
    return sorted(recordings.values(), key=lambda item: (item["recording"]["bird_id"], item["recording"]["filename"]))


def main() -> None:
    args = parse_args()
    args.dst_dir.mkdir(parents=True, exist_ok=True)
    for code, (filename, path) in SPECIES.items():
        assert path.is_dir(), f"Missing dataset directory: {path}"
        payload = {"metadata": {"units": "ms"}, "recordings": load_species(path, code)}
        dst_path = args.dst_dir / filename
        dst_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {len(payload['recordings'])} recordings to {dst_path}")


if __name__ == "__main__":
    main()
