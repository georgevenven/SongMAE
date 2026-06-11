#!/usr/bin/env python3
"""BirdSet -> spectrogram wrapper."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from audio2spec import write_audio_params, write_waveform_spectrogram

def merge_intervals(intervals):
    intervals = sorted(intervals)
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def annotation_events(sample, mode):
    if mode == "none":
        return []
    if mode == "human":
        start = sample.get("start_time")
        end = sample.get("end_time")
        if start is None or end is None:
            return []
        return [{"onset_ms": float(start) * 1000.0, "offset_ms": float(end) * 1000.0}]

    events = sample.get("detected_events", [])
    return [
        {"onset_ms": float(start) * 1000.0, "offset_ms": float(end) * 1000.0}
        for start, end in merge_intervals(events)
    ]


def recording_payload(sample, filename, detection_mode):
    return {
        "recording": {
            "filename": filename,
            "ebird_code": sample.get("ebird_code"),
            "ebird_code_multilabel": sample.get("ebird_code_multilabel", []),
            "lat": sample.get("lat"),
            "long": sample.get("long"),
            "source": sample.get("source", "xenocanto"),
            "quality": sample.get("quality"),
            "recordist": sample.get("recordist"),
            "license": sample.get("license"),
        },
        "detected_events": annotation_events(sample, detection_mode),
    }


def sample_name(sample, index):
    audio = sample.get("audio", {})
    filepath = sample.get("filepath") or audio.get("path")
    if filepath:
        return Path(filepath).stem
    return f"sample_{index:06d}"


def birdset_to_specs(
    birdset,
    split,
    out_dir,
    detection_mode="human",
    sr=32_000,
    hop_size=64,
    n_fft=1024,
    n_mels=128,
    take_n=None,
):
    from datasets import Audio, load_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_audio_params(out_dir, sr, n_fft, hop_size, n_mels)

    dataset = load_dataset("DBD-research-group/BirdSet", birdset, split=split, streaming=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sr))

    records = []
    for index, sample in enumerate(dataset, start=1):
        if take_n is not None and index > take_n:
            break

        audio = sample["audio"]
        name = sample_name(sample, index)
        write_waveform_spectrogram(
            audio["array"],
            out_dir / f"{name}.npy",
            sr,
            n_fft,
            hop_size,
            n_mels,
        )
        if detection_mode != "none":
            records.append(recording_payload(sample, f"{name}.wav", detection_mode))

        if index % 250 == 0:
            print(f"processed {index} samples")

    if detection_mode != "none":
        annotations = {"metadata": {"units": "ms"}, "recordings": records}
        path = out_dir / f"{birdset}_{split}_annotations.json"
        path.write_text(json.dumps(annotations, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Convert BirdSet audio to spectrogram .npy files.")
    parser.add_argument("--birdset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--detections", choices=["none", "human", "bambird"], default="human")
    parser.add_argument("--sr", type=int, default=32_000)
    parser.add_argument("--hop_size", type=int, default=64)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--take_n", type=int)
    args = parser.parse_args()

    birdset_to_specs(
        args.birdset,
        args.split,
        args.out_dir,
        args.detections,
        args.sr,
        args.hop_size,
        args.n_fft,
        args.n_mels,
        args.take_n,
    )


if __name__ == "__main__":
    main()
