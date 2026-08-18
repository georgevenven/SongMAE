#!/usr/bin/env python3
"""Make 0 dB pink-noise spectrograms for the full zebra-finch corpus."""
import json
import shutil
import sys
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.core.audio2spec import compute_spectrogram
from src.core.utils import write_spec
from src.dataset_utils.individual_id.background_augmentation import add_colored_noise_0db, load_audio
from src.dataset_utils.individual_id.make_zebra_finch_colored_noise import event_mask


ANNOTATIONS = ROOT / "files/annotation jsons/zf_annotations.json"
WAVS = Path("/media/george-vengrovski/disk2/raw_data/wav_files_canary_zf_bf_songmae/zf")
CLEAN_SPECS = Path("/media/george-vengrovski/disk2/specs/zebra_finch_5ms")
OUTPUT = Path("/media/george-vengrovski/disk2/zebra_finch_pink_noise_event_probe_full/specs/pink_0db")
RESULTS = ROOT / "results/individual_id/zebra_finch_pink_noise_event_probe_full"
SEED = 42


def main():
    rows = []
    for row in json.loads(ANNOTATIONS.read_text())["recordings"]:
        stem = Path(row["recording"]["filename"]).stem
        duration_ms = sf.info(WAVS / f"{stem}.wav").duration * 1000
        event_ms = sum(event["offset_ms"] - event["onset_ms"] for event in row["detected_events"])
        if event_ms >= 500 and duration_ms - event_ms >= 500:
            rows.append(row)
    rows.sort(key=lambda row: row["recording"]["filename"])
    assert len(rows) == 1033
    OUTPUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CLEAN_SPECS / "audio_params.json", OUTPUT / "audio_params.json")

    manifest = []
    for index, row in enumerate(rows):
        stem = Path(row["recording"]["filename"]).stem
        destination = OUTPUT / f"{stem}.npy"
        if not destination.exists():
            wav = load_audio(WAVS / f"{stem}.wav")
            mask = event_mask(wav, row["detected_events"])
            noisy, gain, event_rms, _ = add_colored_noise_0db(wav, mask, SEED + index, "pink")
            write_spec(destination, compute_spectrogram(noisy))
        else:
            gain = event_rms = None
        manifest.append({
            "stem": stem, "bird_id": str(row["recording"]["bird_id"]),
            "noise_seed": SEED + index, "noise_gain": gain, "event_rms": event_rms,
        })
        if (index + 1) % 100 == 0:
            print(f"wrote {index + 1}/{len(rows)}", flush=True)

    (RESULTS / "annotations.json").write_text(json.dumps({"metadata": {"units": "ms"}, "recordings": rows}, indent=2) + "\n")
    (RESULTS / "manifest.json").write_text(json.dumps({
        "condition": "pink_0db", "snr_db": 0, "noise_color": "pink",
        "noise_reference": "foreground and noise RMS measured inside outer detected events",
        "selection": "at least 500 ms inside and 500 ms outside outer detected events",
        "recordings": manifest,
    }, indent=2) + "\n")
    assert len(list(OUTPUT.glob("*.npy"))) == len(rows)
    (OUTPUT.parent / ".complete").write_text(f"{len(rows)} recordings\n")


if __name__ == "__main__":
    main()
