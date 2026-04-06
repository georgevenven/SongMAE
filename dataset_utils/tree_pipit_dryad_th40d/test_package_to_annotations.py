#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import subprocess
import wave
import zipfile
from pathlib import Path


TEST_PACKAGE_ZIP = "Tree+Pipit+male+ID+-+test+package.zip"
SOLUTION_PDF = "Tree+Pipit+male+ID+-+test+solution.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Dryad Tree Pipit blind-test package into TinyBird annotations.json."
    )
    parser.add_argument("--src_dir", required=True, type=Path, help="Directory containing the Dryad files.")
    parser.add_argument("--dst_dir", required=True, type=Path, help="Directory where annotations.json will be written.")
    return parser.parse_args()


def _extract_pdf_text(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _parse_solution_mapping(pdf_text: str) -> dict[int, str]:
    lines = [line.strip() for line in pdf_text.splitlines() if line.strip()]
    male_idx = lines.index("male ID")
    recording_idx = lines.index("recording nos.")
    bi_idx = lines.index("bi-syllable(s)")

    male_ids = lines[male_idx + 1 : recording_idx]
    recording_lists = lines[recording_idx + 1 : bi_idx]
    if len(male_ids) != len(recording_lists):
        raise ValueError("Could not align male IDs with recording-number rows in the solution PDF")

    mapping: dict[int, str] = {}
    for male_id, recording_list in zip(male_ids, recording_lists):
        for item in recording_list.split(","):
            recording_no = int(item.strip())
            mapping[recording_no] = male_id
    return mapping


def _wav_duration_ms_from_reader(reader) -> float:
    with wave.open(reader, "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
    return (frames / sample_rate) * 1000.0


def main() -> None:
    args = parse_args()
    src_dir = args.src_dir.expanduser().resolve()
    solution_text = _extract_pdf_text(src_dir / SOLUTION_PDF)
    bird_ids_by_recording = _parse_solution_mapping(solution_text)

    recordings = []
    with zipfile.ZipFile(src_dir / TEST_PACKAGE_ZIP) as zf:
        wav_members = [
            name
            for name in zf.namelist()
            if name.lower().endswith(".wav") and not Path(name).name.startswith("._")
        ]

        for wav_name in sorted(wav_members, key=lambda name: int(Path(name).stem)):
            filename = Path(wav_name).name
            recording_no = int(Path(filename).stem)
            if recording_no not in bird_ids_by_recording:
                raise ValueError(f"No male ID found for recording {recording_no}")

            with zf.open(wav_name) as handle:
                duration_ms = _wav_duration_ms_from_reader(handle)
            bird_id = bird_ids_by_recording[recording_no]

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
            "species": "Tree Pipit",
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
