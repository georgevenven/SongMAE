#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio2spec import AudioEvent, process_audio_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build prefixed European starling specs so duplicate wav basenames do not collide."
    )
    parser.add_argument(
        "--annotation_json",
        type=Path,
        default=ROOT / "files" / "european_starling_annotations.json",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=ROOT / "files" / "european_starling_annotations_fixed.json",
    )
    parser.add_argument(
        "--source_spec_dir",
        type=Path,
        default=Path("/media/george-vengrovski/disk2/specs/european_starling_64hop_32khz"),
    )
    parser.add_argument(
        "--output_spec_dir",
        type=Path,
        default=Path("/media/george-vengrovski/disk2/specs/european_starling_64hop_32khz_prefixed"),
    )
    parser.add_argument(
        "--raw_audio_root",
        type=Path,
        default=Path("/media/george-vengrovski/disk2/raw_data/european_starling/audio_files/3237218"),
    )
    return parser.parse_args()


def unprefixed_name(filename: str) -> str:
    name = Path(filename).name
    if "__" not in name:
        return name
    return name.split("__", 1)[1]


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_audio_params(output_spec_dir: Path, source_spec_dir: Path) -> None:
    params = json.loads((source_spec_dir / "audio_params.json").read_text())
    total = 0
    total_sq = 0.0
    count = 0
    for path in tqdm(sorted(output_spec_dir.glob("*.npy")), desc="stats"):
        arr = np.load(path, mmap_mode="r")
        total += float(np.sum(arr, dtype=np.float64))
        total_sq += float(np.sum(np.square(arr, dtype=np.float64), dtype=np.float64))
        count += int(arr.size)
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    params["mean"] = mean
    params["std"] = variance**0.5
    (output_spec_dir / "audio_params.json").write_text(json.dumps(params, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    payload = json.loads(args.annotation_json.read_text())
    recordings = payload["recordings"]

    by_unprefixed: dict[str, list[dict]] = {}
    for recording in recordings:
        filename = recording["recording"]["filename"]
        by_unprefixed.setdefault(unprefixed_name(filename), []).append(recording)
    colliding = {name for name, rows in by_unprefixed.items() if len(rows) > 1}

    args.output_spec_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source_spec_dir / "audio_params.json", args.output_spec_dir / "audio_params.json")

    copied = 0
    recomputed = 0
    skipped = SimpleNamespace(value=0)
    params = json.loads((args.source_spec_dir / "audio_params.json").read_text())
    worker_args = (
        args.output_spec_dir,
        int(params["sr"]),
        int(params["fft"]),
        int(params["hop_size"]),
        True,
        int(params["mels"]),
        25,
        25,
        skipped,
    )

    for recording in tqdm(recordings, desc="prefixed specs"):
        filename = recording["recording"]["filename"]
        bird_id = recording["recording"]["bird_id"]
        source_name = unprefixed_name(filename)
        output_stem = Path(filename).stem
        output_path = args.output_spec_dir / f"{output_stem}.npy"
        if source_name not in colliding:
            link_or_copy(args.source_spec_dir / f"{Path(source_name).stem}.npy", output_path)
            copied += 1
            continue

        raw_path = args.raw_audio_root / bird_id / source_name
        event = recording["detected_events"][0]
        clip = AudioEvent(
            raw_path,
            float(event["onset_ms"]) / 1000.0,
            float(event["offset_ms"]) / 1000.0,
            output_stem,
        )
        error = process_audio_file(clip, *worker_args)
        assert error is None, error
        recomputed += 1

    missing = [
        recording["recording"]["filename"]
        for recording in recordings
        if not (args.output_spec_dir / f"{Path(recording['recording']['filename']).stem}.npy").exists()
    ]
    assert not missing, f"Missing {len(missing)} specs, first: {missing[:5]}"

    write_audio_params(args.output_spec_dir, args.source_spec_dir)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {len(recordings)} specs to {args.output_spec_dir}")
    print(f"Copied or linked {copied}; recomputed {recomputed}; skipped {skipped.value}")


if __name__ == "__main__":
    main()
