#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import wave
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Zenodo 3237218 European starling bird bundles into TinyBird annotations.json."
    )
    parser.add_argument("--src_dir", required=True, type=Path, help="Directory containing the per-bird ZIP files.")
    parser.add_argument("--dst_dir", required=True, type=Path, help="Directory where annotations.json will be written.")
    return parser.parse_args()


def _ignore_member(name: str) -> bool:
    basename = Path(name).name
    return basename.startswith("._") or "__MACOSX" in Path(name).parts


def _wav_duration_ms_from_reader(reader) -> float:
    with wave.open(reader, "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
    return (frames / sample_rate) * 1000.0


def _load_bird_metadata(src_dir: Path) -> dict[str, dict[str, str]]:
    metadata_path = src_dir / "bird_IDs.txt"
    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        return {
            row["individual_ID"]: {
                "recorded_by": row.get("recorded_by", ""),
                "recording_year": row.get("recording_year", ""),
            }
            for row in reader
            if row.get("individual_ID")
        }


def _parse_sidecar_csv(data: bytes) -> dict[str, str]:
    text = data.decode("utf-8", errors="replace")
    row = next(csv.reader(io.StringIO(text)))
    if len(row) < 3:
        raise ValueError(f"Unexpected starling sidecar CSV row: {row}")
    return {
        "bird_id": row[0],
        "source_wav_path": row[1],
        "clip_stem": row[2],
    }


def main() -> None:
    args = parse_args()
    src_dir = args.src_dir.expanduser().resolve()
    bird_metadata = _load_bird_metadata(src_dir)

    recordings = []
    for zip_path in sorted(src_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            sidecars: dict[str, dict[str, str]] = {}
            wav_members: list[str] = []

            for name in zf.namelist():
                if _ignore_member(name):
                    continue
                if name.endswith(".csv"):
                    sidecar = _parse_sidecar_csv(zf.read(name))
                    sidecars[sidecar["clip_stem"]] = sidecar
                elif name.endswith(".wav"):
                    wav_members.append(name)

            for wav_name in sorted(wav_members):
                basename = Path(wav_name).name
                clip_stem = Path(basename).stem
                with zf.open(wav_name) as handle:
                    duration_ms = _wav_duration_ms_from_reader(handle)
                sidecar = sidecars.get(clip_stem, {})

                bird_id = sidecar.get("bird_id", zip_path.stem)
                filename = f"{bird_id}__{basename}"

                recordings.append(
                    {
                        "recording": {
                            "filename": filename,
                            "bird_id": bird_id,
                            "detected_vocalizations": 1,
                        },
                        "detected_events": [
                            {
                                "onset_ms": 0.0,
                                "offset_ms": duration_ms,
                                "units": [],
                            }
                        ],
                    }
                )

    payload = {
        "metadata": {
            "units": "ms",
            "species": "European Starling",
        },
        "recordings": recordings,
    }
    args.dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = args.dst_dir / "annotations.json"
    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(recordings)} recordings to {dst_path}")


if __name__ == "__main__":
    main()
