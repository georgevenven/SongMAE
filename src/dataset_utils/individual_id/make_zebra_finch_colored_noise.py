#!/usr/bin/env python3
"""Make clean and colored-noise zebra-finch probe data."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.core.audio2spec import compute_spectrogram
from src.core.utils import load_spec, write_spec
from src.dataset_utils.individual_id.background_augmentation import add_colored_noise_0db, load_audio


ANNOTATIONS = ROOT / "files/annotation jsons/zf_annotations.json"
WAVS = Path("/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae/zf")
SPECS = Path("/media/george-vengrovski/disk2/specs/zebra_finch_5ms")
OUTPUT = Path("/media/george-vengrovski/disk2/zebra_finch_pink_noise_event_probe")
RESULTS = ROOT / "results/individual_id/zebra_finch_pink_noise_event_probe"
SAMPLE_RATE = 32_000


def event_mask(wav, events):
    mask = np.zeros(len(wav), dtype=bool)
    for event in events:
        start = max(0, round(event["onset_ms"] * SAMPLE_RATE / 1000))
        end = min(len(wav), round(event["offset_ms"] * SAMPLE_RATE / 1000))
        mask[start:end] = True
    return mask


def select_recordings(count, seed):
    rows = json.loads(ANNOTATIONS.read_text())["recordings"]
    by_bird = {}
    for row in rows:
        stem = Path(row["recording"]["filename"]).stem
        wav = WAVS / f"{stem}.wav"
        if not wav.exists() or not (SPECS / f"{stem}.npy").exists():
            continue
        duration_ms = 1000 * sf.info(wav).duration
        event_ms = sum(event["offset_ms"] - event["onset_ms"] for event in row["detected_events"])
        if event_ms >= 500 and duration_ms - event_ms >= 500:
            by_bird.setdefault(str(row["recording"]["bird_id"]), []).append(row)

    rng = np.random.default_rng(seed)
    for rows in by_bird.values():
        rng.shuffle(rows)
    selected = []
    while len(selected) < count:
        for bird in sorted(by_bird):
            if by_bird[bird] and len(selected) < count:
                selected.append(by_bird[bird].pop())
    rng.shuffle(selected)
    return selected


def draw(ax, spec, duration, title):
    ax.imshow(spec, origin="lower", aspect="auto", vmin=-80, vmax=0, cmap="magma", extent=(0, duration, 0, 16))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Seconds")
    ax.set_ylabel("kHz")


def write_pair(clean, noisy, row, label, output):
    duration = clean.shape[1] * 0.005
    fig, axes = plt.subplots(2, 1, figsize=(12, 5))
    draw(axes[0], clean, duration, f'{row["stem"]} | clean')
    draw(axes[1], noisy, duration, f'{row["stem"]} | {label}')
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def write_sheet(rows, specs, label, output):
    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 2.2 * len(rows)), squeeze=False)
    for row, (clean, noisy), pair in zip(rows, specs, axes):
        duration = clean.shape[1] * 0.005
        draw(pair[0], clean, duration, f'{row["stem"]} | clean')
        draw(pair[1], noisy, duration, label)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition", default="pink_0db")
    parser.add_argument("--noise_color", choices=("white", "pink", "brown", "green"), default="pink")
    parser.add_argument("--noise_scale", type=float, default=1.0)
    args = parser.parse_args()
    assert args.noise_scale > 0
    snr_db = -20 * np.log10(args.noise_scale)
    label = f"{args.noise_color} noise | {snr_db:.2f} dB event SNR"
    clean_specs = OUTPUT / "specs/clean"
    noisy_specs = OUTPUT / "specs" / args.condition
    noisy_wavs = OUTPUT / "wav" / args.condition
    images = RESULTS / ("spectrograms" if args.condition == "pink_0db" else f"spectrograms_{args.condition}")
    sheets = RESULTS / ("contact_sheets" if args.condition == "pink_0db" else f"contact_sheets_{args.condition}")
    for path in (clean_specs, noisy_specs, noisy_wavs, images, sheets):
        path.mkdir(parents=True, exist_ok=True)
    for path in (clean_specs, noisy_specs):
        shutil.copy2(SPECS / "audio_params.json", path / "audio_params.json")

    selected = select_recordings(args.count, args.seed)
    split_rng = np.random.default_rng(args.seed + 1)
    test_stems = set(split_rng.choice([Path(row["recording"]["filename"]).stem for row in selected], args.count // 5, replace=False))
    manifest_rows = []
    for batch_start in range(0, len(selected), 5):
        batch = selected[batch_start : batch_start + 5]
        batch_specs = []
        batch_rows = []
        for index, row in enumerate(batch, batch_start):
            stem = Path(row["recording"]["filename"]).stem
            wav = load_audio(WAVS / f"{stem}.wav")
            mask = event_mask(wav, row["detected_events"])
            assert mask.any() and (~mask).any()
            noisy, noise_gain, event_rms, _ = add_colored_noise_0db(
                wav, mask, args.seed + index, args.noise_color
            )
            noisy = wav + args.noise_scale * (noisy - wav)
            noise_gain *= args.noise_scale
            peak = float(np.max(np.abs(noisy)))
            clean_spec = load_spec(SPECS / f"{stem}.npy")
            noisy_spec = compute_spectrogram(noisy)
            assert clean_spec.shape == noisy_spec.shape
            clean_path = clean_specs / f"{stem}.npy"
            if not clean_path.exists():
                clean_path.symlink_to(SPECS / f"{stem}.npy")
            write_spec(noisy_specs / f"{stem}.npy", noisy_spec)
            sf.write(noisy_wavs / f"{stem}.wav", noisy, SAMPLE_RATE, subtype="FLOAT")
            manifest_row = {
                "stem": stem,
                "bird_id": str(row["recording"]["bird_id"]),
                "split": "test" if stem in test_stems else "train",
                "condition": args.condition,
                "snr_db": snr_db,
                "noise_seed": args.seed + index,
                "noise_gain": noise_gain,
                "event_rms": event_rms,
                "mixed_peak": peak,
            }
            write_pair(clean_spec, noisy_spec, manifest_row, label, images / f"{stem}.png")
            manifest_rows.append(manifest_row)
            batch_rows.append(manifest_row)
            batch_specs.append((clean_spec, noisy_spec))
        write_sheet(batch_rows, batch_specs, label, sheets / f"recordings_{batch_start + 1:03d}_{batch_start + len(batch):03d}.png")
        print(f"wrote {batch_start + len(batch)}/{len(selected)}", flush=True)

    selected_stems = {row["stem"] for row in manifest_rows}
    annotations = json.loads(ANNOTATIONS.read_text())
    annotations["recordings"] = [
        row for row in annotations["recordings"]
        if Path(row["recording"]["filename"]).stem in selected_stems
    ]
    (RESULTS / "annotations.json").write_text(json.dumps(annotations, indent=2) + "\n")
    manifest_path = RESULTS / ("manifest.json" if args.condition == "pink_0db" else f"manifest_{args.condition}.json")
    manifest_path.write_text(json.dumps({
        "foreground_species": "zebra_finch",
        "sample_rate": SAMPLE_RATE,
        "condition": args.condition,
        "snr_db": snr_db,
        "noise_scale": args.noise_scale,
        "noise_color": args.noise_color,
        "green_noise_definition": "Gaussian frequency weighting centered at 500 Hz with one octave log-frequency standard deviation" if args.noise_color == "green" else None,
        "noise_reference": "foreground and noise RMS measured inside outer detected_events",
        "split": "80 train and 20 test source recordings",
        "recordings": manifest_rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
