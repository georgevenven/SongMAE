#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES_DIR = ROOT / "files"
RAW_DATA_ROOT = Path("/media/george-vengrovski/disk2/raw_data")

SPECIES_OUTPUTS = {
    "Chiffchaff": FILES_DIR / "chiffchaff_annotations.json",
    "Little Owl": FILES_DIR / "little_owl_annotations.json",
    "Tree Pipit": FILES_DIR / "tree_pipit_annotations.json",
}

SPECIES_CODE_TO_NAME = {
    "chiffchaff": "Chiffchaff",
    "littleowl": "Little Owl",
    "pipit": "Tree Pipit",
}

DATASET_1413495_DIRS = {
    "chiffchaff": RAW_DATA_ROOT / "chiffchaff" / "audio_files" / "stowell_linhart_1413495",
    "littleowl": RAW_DATA_ROOT / "little_owl" / "audio_files" / "stowell_linhart_1413495",
    "pipit": RAW_DATA_ROOT / "tree_pipit" / "audio_files" / "stowell_linhart_1413495",
}

CSV_NAME_RE = re.compile(
    r"^(?P<species>[a-z]+)-(?P<protocol>[a-z]+)-(?P<recording_type>fg|bg)-(?P<split>trn|tst)\.csv$"
)


def wav_duration_ms(path: Path) -> float:
    with path.open("rb") as handle:
        with wave.open(handle, "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
    return (frames / sample_rate) * 1000.0


def build_audio_member_map(audio_dir: Path) -> dict[str, Path]:
    members: dict[str, Path] = {}
    for path in audio_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("._") or path.suffix.lower() != ".wav":
            continue
        members[path.name] = path
    return members


def resolve_one_hot_bird_id(row: dict[str, str], label_columns: list[str], filename: str) -> str:
    active = []
    for bird_id in label_columns:
        value = (row.get(bird_id) or "").strip()
        if value not in {"", "0", "0.0"}:
            active.append(bird_id)
    if len(active) != 1:
        raise ValueError(f"Expected exactly one active bird_id for {filename}, found {active}")
    return active[0]


def add_or_merge(
    recordings_by_species: dict[str, dict[str, dict[str, object]]],
    species: str,
    recording: dict[str, object],
) -> None:
    filename = recording["recording"]["filename"]
    existing = recordings_by_species.setdefault(species, {}).get(filename)
    if existing is None:
        recordings_by_species[species][filename] = recording
        return
    if existing["recording"]["bird_id"] != recording["recording"]["bird_id"]:
        raise ValueError(f"Conflicting bird_id for {species} / {filename}")
    existing["recording"]["detected_vocalizations"] = len(existing["detected_events"])


def load_1413495_species(recordings_by_species: dict[str, dict[str, dict[str, object]]]) -> None:
    for species_code, dataset_dir in DATASET_1413495_DIRS.items():
        csv_dir = dataset_dir / "csv"
        assert csv_dir.is_dir(), f"Missing CSV directory: {csv_dir}"

        audio_members = {
            recording_type: build_audio_member_map(dataset_dir / recording_type)
            for recording_type in ("fg", "bg")
        }
        species = SPECIES_CODE_TO_NAME[species_code]

        for csv_path in sorted(csv_dir.glob(f"{species_code}-*.csv")):
            match = CSV_NAME_RE.match(csv_path.name)
            if match is None:
                continue

            recording_type = match.group("recording_type")
            rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
            if not rows:
                continue

            with csv_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                assert reader.fieldnames is not None
                label_columns = [field for field in reader.fieldnames if field != "wavfilename"]
                members = audio_members[recording_type]

                for row in reader:
                    filename = Path((row.get("wavfilename") or "").strip()).name
                    if not filename:
                        continue

                    bird_id = resolve_one_hot_bird_id(row, label_columns, filename)
                    audio_path = members.get(filename)
                    if audio_path is None:
                        raise FileNotFoundError(f"Audio file not found for CSV row: {csv_path.name} / {filename}")
                    duration_ms = wav_duration_ms(audio_path)

                    detected_events = []
                    if recording_type == "fg":
                        detected_events.append(
                            {
                                "onset_ms": 0.0,
                                "offset_ms": duration_ms,
                                "units": [],
                            }
                        )

                    add_or_merge(
                        recordings_by_species,
                        species,
                        {
                            "recording": {
                                "filename": filename,
                                "bird_id": bird_id,
                                "detected_vocalizations": len(detected_events),
                            },
                            "detected_events": detected_events,
                        },
                    )


def main() -> None:
    recordings_by_species: dict[str, dict[str, dict[str, object]]] = {}
    load_1413495_species(recordings_by_species)

    FILES_DIR.mkdir(parents=True, exist_ok=True)
    for species, dst_path in SPECIES_OUTPUTS.items():
        recordings = sorted(
            recordings_by_species.get(species, {}).values(),
            key=lambda item: (item["recording"]["bird_id"], item["recording"]["filename"]),
        )
        payload = {
            "metadata": {
                "units": "ms",
                "species": species,
            },
            "recordings": recordings,
        }
        with dst_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"Wrote {len(recordings)} recordings to {dst_path}")


if __name__ == "__main__":
    main()
