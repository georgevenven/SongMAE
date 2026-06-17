#!/usr/bin/env python3
"""BirdSet -> spectrogram wrapper."""

import argparse
import concurrent.futures as futures
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.core.audio2spec import DEFAULT_SHARD_SIZE, compute_spectrogram, write_audio_params, write_spec_shard, write_waveform_spectrogram
from src.core.utils import write_spec


def append_shard_index(path, rows):
    with Path(path).open("a") as f:
        for name, shard, start, end in rows:
            f.write(f"{name}\t{shard}\t{start}\t{end}\n")


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
        # We do not know what BirdSet's start_time/end_time fields represent.
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


def recording_payload(sample, filename, events):
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
        "detected_events": events,
    }


def ms_to_bin(ms, sr, hop_size):
    return int(ms * sr / (1000.0 * hop_size))


def write_event_spectrograms(wav, out_dir, name, events, sr, n_fft, hop_size, n_mels, storage_dtype):
    spec = compute_spectrogram(wav, sr, n_fft, hop_size, n_mels)
    for event in events:
        start_ms = int(round(event["onset_ms"]))
        end_ms = int(round(event["offset_ms"]))
        start_bin = ms_to_bin(event["onset_ms"], sr, hop_size)
        end_bin = ms_to_bin(event["offset_ms"], sr, hop_size)
        start_bin = max(0, min(start_bin, spec.shape[1]))
        end_bin = max(0, min(end_bin, spec.shape[1]))
        if start_bin >= end_bin:
            continue
        write_spec(out_dir / f"{name}__ms_{start_ms}_{end_ms}.npy", spec[:, start_bin:end_bin], storage_dtype)


def write_sample_spectrogram(task):
    index, wav, out_dir, name, detection_mode, events, record, sr, n_fft, hop_size, n_mels, storage_dtype, shard_size = task
    if shard_size:
        spec = compute_spectrogram(wav, sr, n_fft, hop_size, n_mels)
        return index, record, name, spec
    if detection_mode == "bambird":
        write_event_spectrograms(wav, out_dir, name, events, sr, n_fft, hop_size, n_mels, storage_dtype)
    else:
        write_waveform_spectrogram(wav, out_dir / f"{name}.npy", sr, n_fft, hop_size, n_mels, storage_dtype)
    return index, record, None, None


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
    detection_mode="none",
    sr=32_000,
    hop_size=160,
    n_fft=1024,
    n_mels=128,
    take_n=None,
    workers=1,
    storage_dtype="float32",
    shard_size=DEFAULT_SHARD_SIZE,
):
    from datasets import Audio, load_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    success_path = out_dir / "_SUCCESS"
    success_path.unlink(missing_ok=True)
    write_audio_params(out_dir, sr, n_fft, hop_size, n_mels, storage_dtype)
    assert shard_size >= 0
    assert detection_mode != "bambird" or shard_size == 0

    if shard_size:
        partial_index = out_dir / "shards" / "index.partial.tsv"
        final_index = out_dir / "shards" / "index.tsv"
        partial_index.parent.mkdir(parents=True, exist_ok=True)
        partial_index.write_text("name\tshard\tstart\tend\n")
        final_index.unlink(missing_ok=True)

    dataset = load_dataset("DBD-research-group/BirdSet", birdset, split=split, trust_remote_code=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sr))

    assert workers > 0
    records = []
    shard_rows = []
    shard_index = 0
    pending = set()

    def collect(done):
        nonlocal shard_index, shard_rows
        for future in done:
            index, record, name, spec = future.result()
            if record is not None:
                records.append((index, record))
            if not shard_size:
                continue
            shard_rows.append((name, spec))
            if len(shard_rows) == shard_size:
                append_shard_index(partial_index, write_spec_shard(out_dir, shard_index, shard_rows, storage_dtype))
                shard_index += 1
                shard_rows = []

    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for index, sample in enumerate(dataset, start=1):
            if take_n is not None and index > take_n:
                break

            audio = sample["audio"]
            name = sample_name(sample, index)
            events = annotation_events(sample, detection_mode)
            record = recording_payload(sample, f"{name}.wav", events)

            task = (
                index,
                audio["array"],
                out_dir,
                name,
                detection_mode,
                events,
                record,
                sr,
                n_fft,
                hop_size,
                n_mels,
                storage_dtype,
                shard_size,
            )
            pending.add(pool.submit(write_sample_spectrogram, task))
            if len(pending) >= workers * 2:
                done, pending = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
                collect(done)

            if index % 250 == 0:
                print(f"processed {index} samples")

        collect(pending)

    if shard_rows:
        append_shard_index(partial_index, write_spec_shard(out_dir, shard_index, shard_rows, storage_dtype))
    if shard_size:
        partial_index.replace(final_index)

    records = [record for _, record in sorted(records)]
    annotations = {"metadata": {"units": "ms"}, "recordings": records}
    path = out_dir / f"{birdset}_{split}_annotations.json"
    path.write_text(json.dumps(annotations, indent=2) + "\n")
    success_path.write_text("ok\n")


def main():
    parser = argparse.ArgumentParser(description="Convert BirdSet audio to spectrogram .npy files.")
    parser.add_argument("--birdset", required=True)  # like XCM or XCL
    parser.add_argument("--split", default="train")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--detections", choices=["none", "human", "bambird"], default="none") # not sure what human is, prolly can nuke this 
    parser.add_argument("--sr", type=int, default=32_000)
    parser.add_argument("--hop_size", type=int, default=160)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--take_n", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--storage_dtype", choices=["float32", "int8_affine"], default="float32")
    parser.add_argument("--shard_size", type=int, default=DEFAULT_SHARD_SIZE)
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
        args.workers,
        args.storage_dtype,
        args.shard_size,
    )


if __name__ == "__main__":
    main()
