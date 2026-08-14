#!/usr/bin/env python3
"""Add the pink-noise sweep to a 30-event-per-bird species bundle."""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.core.audio2spec import compute_spectrogram
from src.core.utils import write_spec
from src.dataset_utils.individual_id.background_augmentation import add_pink_noise_0db, load_audio


WAV_ROOT = Path("/media/george-vengrovski/disk2/raw_data/individual_id_multispecies_background_robustness")
SPEC_ROOT = Path("/media/george-vengrovski/disk2/specs/individual_id_multispecies_background_robustness_5ms")
OUTPUT_ROOT = Path("/media/george-vengrovski/disk2/individual_id_pink_noise_few_shot")
SCALES = {
    "pink_p12db": 0.25,
    "pink_p6db": 0.5,
    "pink_0db": 1.0,
    "pink_m6db": 2.0,
    "pink_m12db": 4.0,
    "pink_m18db": 8.0,
}


def convert(item):
    index, path, output = item
    wav = load_audio(path)
    mixed, gain, event_rms, _ = add_pink_noise_0db(
        wav, np.ones(len(wav), dtype=bool), 42 + index
    )
    noise = mixed - wav
    for condition, scale in SCALES.items():
        write_spec(output / condition / f"{path.stem}.npy", compute_spectrogram(wav + scale * noise))
    return {
        "stem": path.stem,
        "noise_seed": 42 + index,
        "event_rms": event_rms,
        "pink_0db_gain": gain,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    wavs = WAV_ROOT / args.species / "clean"
    specs = SPEC_ROOT / args.species / "clean"
    output = OUTPUT_ROOT / args.species / "specs"
    clean = output / "clean"
    clean.parent.mkdir(parents=True, exist_ok=True)
    if not clean.exists():
        clean.symlink_to(specs, target_is_directory=True)
    params = json.loads((specs / "audio_params.json").read_text())
    for condition in SCALES:
        path = output / condition
        path.mkdir(parents=True, exist_ok=True)
        (path / "audio_params.json").write_text(json.dumps(params, indent=2) + "\n")
    files = sorted(wavs.glob("*.wav"))
    with mp.Pool(args.workers) as pool:
        rows = list(pool.imap(convert, ((index, path, output) for index, path in enumerate(files))))
    (output.parent / "manifest.json").write_text(json.dumps({
        "sample_rate": 32_000,
        "noise": "pink",
        "noise_reference": "event clip RMS",
        "conditions": {condition: {"scale": scale, "snr_db": -20 * np.log10(scale)} for condition, scale in SCALES.items()},
        "recordings": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
