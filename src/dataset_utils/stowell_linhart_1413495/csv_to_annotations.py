#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import wave
import zipfile
from pathlib import Path


CSV_NAME_RE = re.compile(
    r"^(?P<species>[a-z]+)-(?P<protocol>[a-z]+)-(?P<recording_type>fg|bg)-(?P<split>trn|tst)\.csv$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Zenodo 1413495 one-hot CSV split files into TinyBird annotations.json."
    )
    parser.add_argument("--dataset_dir", required=True, type=Path, help="Directory containing csv.zip and the species audio ZIPs.")
    parser.add_argument("--csv_name", required=True, help="CSV filename inside csv.zip, for example pipit-withinyear-fg-trn.csv.")
    parser.add_argument("--dst_dir", required=True, type=Path, help="Directory where annotations.json will be written.")
    return parser.parse_args()


def _read_csv_text(dataset_dir: Path, csv_name: str) -> str:
    csv_zip = dataset_dir / "csv.zip"
    if csv_zip.exists():
        member_name = f"csv/{csv_name}"
        with zipfile.ZipFile(csv_zip) as zf:
            return zf.read(member_name).decode("utf-8")

    csv_path = dataset_dir / "csv" / csv_name
    if csv_path.exists():
        return csv_path.read_text(encoding="utf-8")

    raise FileNotFoundError(f"Could not find {csv_name} in {csv_zip} or {csv_path}")


def _build_zip_member_map(zf: zipfile.ZipFile) -> dict[str, str]:
    members: dict[str, str] = {}
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        basename = Path(name).name
        if basename.startswith("._"):
            continue
        if not basename.lower().endswith(".wav"):
            continue
        members[basename] = name
    return members


def _wav_duration_ms_from_reader(reader) -> float:
    with wave.open(reader, "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
    return (frames / sample_rate) * 1000.0


def _build_dir_member_map(audio_dir: Path) -> dict[str, Path]:
    members: dict[str, Path] = {}
    for path in audio_dir.rglob("*.wav"):
        if path.name.startswith("._"):
            continue
        members[path.name] = path
    return members


def _load_audio_lookup(dataset_dir: Path, species: str, recording_type: str):
    zip_path = dataset_dir / f"{species}-{recording_type}.zip"
    if zip_path.exists():
        zf = zipfile.ZipFile(zip_path)
        return ("zip", zf, _build_zip_member_map(zf))

    audio_dir = dataset_dir / "wav" / f"{species}-{recording_type}"
    if audio_dir.exists():
        return ("dir", None, _build_dir_member_map(audio_dir))

    extracted_dir = dataset_dir / f"{species}-{recording_type}"
    if extracted_dir.exists():
        return ("dir", None, _build_dir_member_map(extracted_dir))

    raise FileNotFoundError(
        f"Could not find audio for {species}-{recording_type} in {zip_path}, {audio_dir}, or {extracted_dir}"
    )


def _get_duration_ms(lookup_type: str, zf: zipfile.ZipFile | None, members, filename: str) -> float:
    if filename not in members:
        raise FileNotFoundError(f"Audio file not found for CSV row: {filename}")

    if lookup_type == "zip":
        assert zf is not None
        with zf.open(members[filename]) as handle:
            return _wav_duration_ms_from_reader(handle)

    path: Path = members[filename]
    with path.open("rb") as handle:
        return _wav_duration_ms_from_reader(handle)


def _resolve_bird_id(row: dict[str, str], label_columns: list[str], filename: str) -> str:
    active = []
    for bird_id in label_columns:
        value = (row.get(bird_id) or "").strip()
        if value not in {"", "0", "0.0"}:
            active.append(bird_id)

    if len(active) != 1:
        raise ValueError(f"Expected exactly one active bird_id for {filename}, found {active}")
    return active[0]


def main() -> None:
    args = parse_args()
    match = CSV_NAME_RE.match(args.csv_name)
    if match is None:
        raise ValueError(f"CSV name does not match the expected schema: {args.csv_name}")

    species = match.group("species")
    protocol = match.group("protocol")
    recording_type = match.group("recording_type")
    split = match.group("split")

    csv_text = _read_csv_text(args.dataset_dir.expanduser().resolve(), args.csv_name)
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    if reader.fieldnames is None or "wavfilename" not in reader.fieldnames:
        raise ValueError(f"CSV is missing wavfilename column: {args.csv_name}")

    label_columns = [field for field in reader.fieldnames if field != "wavfilename"]
    lookup_type, zf, members = _load_audio_lookup(args.dataset_dir.expanduser().resolve(), species, recording_type)

    recordings = []
    try:
        for row in rows:
            filename = Path((row.get("wavfilename") or "").strip()).name
            if not filename:
                continue

            bird_id = _resolve_bird_id(row, label_columns, filename)
            duration_ms = _get_duration_ms(lookup_type, zf, members, filename)

            detected_events = []
            if recording_type == "fg":
                detected_events.append(
                    {
                        "onset_ms": 0.0,
                        "offset_ms": duration_ms,
                        "units": [],
                    }
                )

            recordings.append(
                {
                    "recording": {
                        "filename": filename,
                        "bird_id": bird_id,
                        "detected_vocalizations": len(detected_events),
                    },
                    "detected_events": detected_events,
                }
            )
    finally:
        if zf is not None:
            zf.close()

    args.dst_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "units": "ms",
            "species": species,
        },
        "recordings": recordings,
    }
    dst_path = args.dst_dir / "annotations.json"
    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(recordings)} recordings to {dst_path}")


if __name__ == "__main__":
    main()
