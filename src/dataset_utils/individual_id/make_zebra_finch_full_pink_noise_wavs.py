#!/usr/bin/env python3
"""Materialize the deterministic 0 dB pink-noise waveforms used by SongMAE."""
import json
from pathlib import Path

import soundfile as sf

from src.dataset_utils.individual_id.background_augmentation import add_colored_noise_0db, load_audio
from src.dataset_utils.individual_id.make_zebra_finch_colored_noise import event_mask


ROOT = Path(__file__).resolve().parents[3]
ANNOTATIONS = ROOT / "results/individual_id/zebra_finch_pink_noise_event_probe_full/annotations.json"
SOURCE = Path("/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae/zf")
OUTPUT = Path("/media/george-vengrovski/disk2/zebra_finch_pink_noise_event_probe_full/wavs/pink_0db")


def main():
    rows = json.loads(ANNOTATIONS.read_text())["recordings"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        name = row["recording"]["filename"]
        destination = OUTPUT / name
        if not destination.exists():
            wav = load_audio(SOURCE / name)
            noisy, *_ = add_colored_noise_0db(wav, event_mask(wav, row["detected_events"]), 42 + index, "pink")
            sf.write(destination, noisy, 32_000, subtype="FLOAT")
        if (index + 1) % 100 == 0:
            print(f"wrote {index + 1}/{len(rows)}", flush=True)
    assert len(list(OUTPUT.glob("*.wav"))) == len(rows)


if __name__ == "__main__":
    main()
