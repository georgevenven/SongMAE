#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path
import wave


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path("/home/george-vengrovski/Downloads")
FILES_DIR = ROOT / "files"

DATASET_1413495 = DOWNLOADS / "1413495"
DATASET_3237218 = DOWNLOADS / "3237218"
DATASET_DRYAD = DOWNLOADS / "doi_10_5061_dryad_th40d__v20161009"

SPECIES_OUTPUTS = {
    "Chiffchaff": FILES_DIR / "chiffchaff_annotations.json",
    "Little Owl": FILES_DIR / "little_owl_annotations.json",
    "Tree Pipit": FILES_DIR / "tree_pipit_annotations.json",
    "European Starling": FILES_DIR / "european_starling_annotations.json",
}

SPECIES_CODE_TO_NAME = {
    "chiffchaff": "Chiffchaff",
    "littleowl": "Little Owl",
    "pipit": "Tree Pipit",
}

CSV_NAME_RE = re.compile(
    r"^(?P<species>[a-z]+)-(?P<protocol>[a-z]+)-(?P<recording_type>fg|bg)-(?P<split>trn|tst)\.csv$"
)

TEST_PACKAGE_ZIP = "Tree+Pipit+male+ID+-+test+package.zip"
SOLUTION_PDF = "Tree+Pipit+male+ID+-+test+solution.pdf"


def wav_duration_ms_from_reader(reader) -> float:
    with wave.open(reader, "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
    return (frames / sample_rate) * 1000.0


def read_text_from_zip(zip_path: Path, member_name: str) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.read(member_name).decode("utf-8")


def build_audio_zip_member_map(zf: zipfile.ZipFile) -> dict[str, str]:
    members: dict[str, str] = {}
    for name in zf.namelist():
        basename = Path(name).name
        if basename.startswith("._") or not basename.lower().endswith(".wav"):
            continue
        members[basename] = name
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


def add_or_merge(recordings_by_species: dict[str, dict[str, dict[str, object]]], species: str, recording: dict[str, object]) -> None:
    filename = recording["recording"]["filename"]
    existing = recordings_by_species.setdefault(species, {}).get(filename)
    if existing is None:
        recordings_by_species[species][filename] = recording
        return

    if existing["recording"]["bird_id"] != recording["recording"]["bird_id"]:
        raise ValueError(f"Conflicting bird_id for {species} / {filename}")

    existing["recording"]["detected_vocalizations"] = len(existing["detected_events"])


def load_1413495_species(recordings_by_species: dict[str, dict[str, dict[str, object]]]) -> None:
    csv_zip = DATASET_1413495 / "csv.zip"
    csv_names = []
    with zipfile.ZipFile(csv_zip) as zf:
        for name in zf.namelist():
            if name.endswith(".csv"):
                csv_names.append(Path(name).name)

    audio_zips = {
        species_code: {
            "fg": zipfile.ZipFile(DATASET_1413495 / f"{species_code}-fg.zip"),
            "bg": zipfile.ZipFile(DATASET_1413495 / f"{species_code}-bg.zip"),
        }
        for species_code in SPECIES_CODE_TO_NAME
    }

    audio_members = {
        species_code: {
            recording_type: build_audio_zip_member_map(zf)
            for recording_type, zf in zip_group.items()
        }
        for species_code, zip_group in audio_zips.items()
    }

    try:
        for csv_name in sorted(csv_names):
            match = CSV_NAME_RE.match(csv_name)
            if match is None:
                continue

            species_code = match.group("species")
            species = SPECIES_CODE_TO_NAME[species_code]
            protocol = match.group("protocol")
            recording_type = match.group("recording_type")
            split = match.group("split")

            csv_text = read_text_from_zip(csv_zip, f"csv/{csv_name}")
            reader = csv.DictReader(io.StringIO(csv_text))
            rows = list(reader)
            label_columns = [field for field in (reader.fieldnames or []) if field != "wavfilename"]
            zf = audio_zips[species_code][recording_type]
            members = audio_members[species_code][recording_type]

            for row in rows:
                filename = Path((row.get("wavfilename") or "").strip()).name
                if not filename:
                    continue

                bird_id = resolve_one_hot_bird_id(row, label_columns, filename)
                member_name = members[filename]
                with zf.open(member_name) as handle:
                    duration_ms = wav_duration_ms_from_reader(handle)

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
    finally:
        for zip_group in audio_zips.values():
            for zf in zip_group.values():
                zf.close()


def load_3237218_species(recordings_by_species: dict[str, dict[str, dict[str, object]]]) -> None:
    bird_id_rows = {}
    with (DATASET_3237218 / "bird_IDs.txt").open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            bird_id_rows[row["individual_ID"]] = row

    for zip_path in sorted(DATASET_3237218.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            sidecars = {}
            for name in zf.namelist():
                basename = Path(name).name
                if basename.startswith("._") or "__MACOSX" in Path(name).parts:
                    continue
                if name.endswith(".csv"):
                    row = next(csv.reader(io.StringIO(zf.read(name).decode("utf-8", errors="replace"))))
                    sidecars[Path(name).stem] = {
                        "bird_id": row[0],
                        "source_wav_path": row[1],
                    }

            for name in sorted(zf.namelist()):
                basename = Path(name).name
                if basename.startswith("._") or "__MACOSX" in Path(name).parts or not name.endswith(".wav"):
                    continue

                stem = Path(basename).stem
                sidecar = sidecars.get(stem, {})
                bird_id = sidecar.get("bird_id", zip_path.stem)
                filename = f"{bird_id}__{basename}"
                with zf.open(name) as handle:
                    duration_ms = wav_duration_ms_from_reader(handle)

                add_or_merge(
                    recordings_by_species,
                    "European Starling",
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
                    },
                )


def extract_pdf_text(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def parse_dryad_solution_mapping(pdf_text: str) -> dict[int, str]:
    lines = [line.strip() for line in pdf_text.splitlines() if line.strip()]
    male_idx = lines.index("male ID")
    recording_idx = lines.index("recording nos.")
    bi_idx = lines.index("bi-syllable(s)")

    male_ids = lines[male_idx + 1 : recording_idx]
    recording_lists = lines[recording_idx + 1 : bi_idx]
    mapping: dict[int, str] = {}
    for male_id, recording_list in zip(male_ids, recording_lists):
        for item in recording_list.split(","):
            mapping[int(item.strip())] = male_id
    return mapping


def load_dryad_tree_pipit(recordings_by_species: dict[str, dict[str, dict[str, object]]]) -> None:
    mapping = parse_dryad_solution_mapping(extract_pdf_text(DATASET_DRYAD / SOLUTION_PDF))
    with zipfile.ZipFile(DATASET_DRYAD / TEST_PACKAGE_ZIP) as zf:
        for name in sorted(
            [n for n in zf.namelist() if n.lower().endswith(".wav") and not Path(n).name.startswith("._")],
            key=lambda value: int(Path(value).stem),
        ):
            filename = Path(name).name
            recording_number = int(Path(filename).stem)
            with zf.open(name) as handle:
                duration_ms = wav_duration_ms_from_reader(handle)
            add_or_merge(
                recordings_by_species,
                "Tree Pipit",
                {
                    "recording": {
                        "filename": filename,
                        "bird_id": mapping[recording_number],
                        "detected_vocalizations": 1,
                    },
                    "detected_events": [
                        {
                            "onset_ms": 0.0,
                            "offset_ms": duration_ms,
                            "units": [],
                        }
                    ],
                },
            )


def main() -> None:
    recordings_by_species: dict[str, dict[str, dict[str, object]]] = {}
    load_1413495_species(recordings_by_species)
    load_3237218_species(recordings_by_species)
    load_dryad_tree_pipit(recordings_by_species)

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
        with dst_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"Wrote {len(recordings)} recordings to {dst_path}")


if __name__ == "__main__":
    main()
