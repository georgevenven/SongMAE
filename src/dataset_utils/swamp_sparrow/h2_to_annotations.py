#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC_DIR = Path("/media/george-vengrovski/disk2/raw_data/swamp_sparrow")
DEFAULT_DST_PATH = ROOT / "files" / "annotation jsons" / "swamp_sparrow_annotations.json"
H2_CANDIDATES = [
    ROOT / "tmp" / "h2" / "h2-1.3.176.jar",
    Path("/usr/local/MATLAB/R2021b/java/jarext/distcomp/h2.jar"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Swamp sparrow Luscinia H2 labels to TinyBird annotations.")
    parser.add_argument("--src_dir", type=Path, default=DEFAULT_SRC_DIR)
    parser.add_argument("--dst_path", type=Path, default=DEFAULT_DST_PATH)
    parser.add_argument("--h2_jar", type=Path)
    return parser.parse_args()


def h2_jar(path: Path | None) -> Path:
    paths = [path] if path else H2_CANDIDATES
    for candidate in paths:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError("Pass --h2_jar pointing to a H2 jar that can read the Luscinia database.")


def db_path(src_dir: Path) -> Path:
    path = src_dir / "data" / "figshare_5625310" / "SwampSparrow.luscdb" / "SwampSparrow.h2.db"
    assert path.exists(), f"Missing H2 database: {path}"
    return path.with_suffix("").with_suffix("")


def quote_sql(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def export_csvs(src_dir: Path, jar: Path, tmp_dir: Path) -> tuple[Path, Path]:
    songs_path = tmp_dir / "songs.csv"
    syllables_path = tmp_dir / "syllables.csv"
    songs_sql = """
        SELECT S.ID SONG_ID, W.FILENAME, I.NAME BIRD_ID
        FROM SONGDATA S
        JOIN INDIVIDUAL I ON I.ID = S.INDIVIDUALID
        JOIN WAVS W ON W.SONGID = S.ID
        ORDER BY W.FILENAME, S.ID
    """
    syllables_sql = """
        SELECT SONGID SONG_ID, STARTTIME ONSET_MS, ENDTIME OFFSET_MS
        FROM SYLLABLE
        ORDER BY SONGID, STARTTIME, ENDTIME
    """
    sql = (
        f"CALL CSVWRITE({quote_sql(songs_path)}, {quote_sql(songs_sql)}); "
        f"CALL CSVWRITE({quote_sql(syllables_path)}, {quote_sql(syllables_sql)});"
    )
    subprocess.run(
        [
            "java",
            "-cp",
            str(jar),
            "org.h2.tools.Shell",
            "-url",
            f"jdbc:h2:{db_path(src_dir)}",
            "-user",
            "sa",
            "-password",
            "",
            "-sql",
            sql,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return songs_path, syllables_path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def is_parent(interval: tuple[int, int], intervals: list[tuple[int, int]]) -> bool:
    onset, offset = interval
    return any(
        (onset, offset) != other and onset <= other[0] and other[1] <= offset
        for other in intervals
    )


def unit(interval: tuple[int, int]) -> dict:
    return {"onset_ms": float(interval[0]), "offset_ms": float(interval[1]), "id": 0}


def events(intervals: list[tuple[int, int]]) -> list[dict]:
    parents = [interval for interval in intervals if is_parent(interval, intervals)]
    leaves = [interval for interval in intervals if interval not in parents]
    used = set()
    output = []

    for parent in parents:
        children = [leaf for leaf in leaves if parent[0] <= leaf[0] and leaf[1] <= parent[1]]
        if children:
            used.update(children)
            output.append({"onset_ms": float(parent[0]), "offset_ms": float(parent[1]), "units": [unit(child) for child in children]})

    for leaf in leaves:
        if leaf not in used:
            output.append({"onset_ms": float(leaf[0]), "offset_ms": float(leaf[1]), "units": [unit(leaf)]})

    return sorted(output, key=lambda event: (event["onset_ms"], event["offset_ms"]))


def build(songs_path: Path, syllables_path: Path) -> dict:
    songs = rows(songs_path)
    song_ids = {int(row["SONG_ID"]) for row in songs}
    by_song = defaultdict(list)
    for row in rows(syllables_path):
        song_id = int(row["SONG_ID"])
        if song_id in song_ids:
            by_song[song_id].append((int(row["ONSET_MS"]), int(row["OFFSET_MS"])))

    recordings = []
    for song in songs:
        song_id = int(song["SONG_ID"])
        detected_events = events(by_song[song_id])
        recordings.append(
            {
                "recording": {
                    "filename": f"{song_id}__{song['FILENAME']}",
                    "bird_id": song["BIRD_ID"],
                    "detected_vocalizations": len(detected_events),
                },
                "detected_events": detected_events,
            }
        )
    return {"metadata": {"units": "ms"}, "recordings": recordings}


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        songs_path, syllables_path = export_csvs(args.src_dir.expanduser(), h2_jar(args.h2_jar), Path(tmp))
        payload = build(songs_path, syllables_path)

    dst_path = args.dst_path.expanduser()
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {len(payload['recordings'])} recordings to {dst_path}")


if __name__ == "__main__":
    main()
