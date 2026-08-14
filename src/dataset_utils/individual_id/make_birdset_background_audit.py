#!/usr/bin/env python3
"""Create a visual audit set of human-empty BirdSet soundscapes."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.core.audio2spec import compute_spectrogram


BIRDSET = Path("/media/george-vengrovski/disk1/data/BirdSet_bambird")
OUTPUT = Path("/media/george-vengrovski/disk2/raw_data/birdset_nonvocal_backgrounds_32khz/manual_audit_100")
DATASETS = ("HSN", "NES", "PER", "POW", "SNE", "SSW", "UHH")
SAMPLE_RATE = 32_000
OFFSETS = (-10, -5, 0, 5, 10)


def candidates(dataset, rng):
    metadata = BIRDSET / dataset / "metadata" / f"{dataset}_metadata_test_5s.parquet"
    frame = pd.read_parquet(metadata)
    frame = frame[frame.ebird_code_multilabel.map(len).eq(0)]
    recordings = {}
    for name in frame.index:
        recording, start, _ = Path(name).stem.rsplit("_", 2)
        recordings.setdefault(recording, {})[int(start)] = name

    rows = []
    for recording, clips in recordings.items():
        centers = [start for start in clips if all(start + offset in clips for offset in OFFSETS)]
        if not centers:
            continue
        center = int(rng.choice(centers))
        rows.append({
            "dataset": dataset,
            "recording": recording,
            "start_s": center - 10,
            "end_s": center + 15,
            "source_files": [clips[center + offset] for offset in OFFSETS],
        })
    rng.shuffle(rows)
    return rows


def select(count, seed):
    rng = np.random.default_rng(seed)
    available = {dataset: candidates(dataset, rng) for dataset in DATASETS}
    selected = []
    while len(selected) < count:
        added = False
        for dataset in DATASETS:
            if available[dataset] and len(selected) < count:
                selected.append(available[dataset].pop())
                added = True
        assert added, f"Only found {len(selected)} eligible source recordings"
    return selected


def load_window(row):
    paths = [BIRDSET / row["dataset"] / "audio/test_5s" / name for name in row["source_files"]]
    parts = [librosa.load(path, sr=SAMPLE_RATE, mono=True)[0] for path in paths]
    assert all(len(part) == 5 * SAMPLE_RATE for part in parts)
    return np.concatenate(parts).astype(np.float32)


def draw(ax, spec, title):
    ax.imshow(spec, origin="lower", aspect="auto", vmin=-80, vmax=0, cmap="magma", extent=(0, 25, 0, 16))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Seconds")
    ax.set_ylabel("kHz")


def write_individual(spec, row, path):
    fig, ax = plt.subplots(figsize=(12, 3))
    draw(ax, spec, f'{row["id"]} | {row["dataset"]} | {row["recording"]}')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_contact_sheet(rows, specs, path):
    fig, axes = plt.subplots(5, 2, figsize=(14, 12), squeeze=False)
    for ax, row, spec in zip(axes.flat, rows, specs):
        draw(ax, spec, f'{row["id"]} | {row["dataset"]} | {row["recording"]}')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert not args.output.exists(), args.output

    wav_dir = args.output / "wav"
    image_dir = args.output / "spectrograms"
    sheet_dir = args.output / "contact_sheets"
    for path in (wav_dir, image_dir, sheet_dir):
        path.mkdir(parents=True)

    rows = select(args.count, args.seed)
    for batch_start in range(0, len(rows), 10):
        batch = rows[batch_start : batch_start + 10]
        specs = []
        for index, row in enumerate(batch, batch_start + 1):
            row["id"] = f"background_{index:03d}"
            wav = load_window(row)
            spec = compute_spectrogram(wav)
            sf.write(wav_dir / f'{row["id"]}.wav', wav, SAMPLE_RATE, subtype="FLOAT")
            write_individual(spec, row, image_dir / f'{row["id"]}.png')
            specs.append(spec)
        write_contact_sheet(batch, specs, sheet_dir / f"backgrounds_{batch_start + 1:03d}_{batch_start + len(batch):03d}.png")
        print(f"wrote {batch_start + len(batch)}/{len(rows)}", flush=True)

    manifest = {
        "sample_rate": SAMPLE_RATE,
        "duration_s": 25,
        "seed": args.seed,
        "selection": "five consecutive test_5s segments with empty human ebird_code_multilabel",
        "source_recording_disjoint": True,
        "automatic_bird_screening": None,
        "dataset_counts": Counter(row["dataset"] for row in rows),
        "backgrounds": rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
