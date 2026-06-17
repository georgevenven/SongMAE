"""Core audio -> spectrogram utilities.

compute_spectrogram turns a waveform into a log-mel spectrogram array.
write_audio_params writes the spectrogram shape/config metadata JSON.
write_waveform_spectrogram saves one waveform as one spectrogram .npy file.
audio_file_to_spec loads one audio file and writes its spectrogram.
audio_dir_to_specs converts every supported audio file under a directory.
compute_statistics computes mean/std over existing spectrogram .npy files.
write_statistics writes those mean/std values into audio_params.json.
"""

import argparse
import json
import multiprocessing as mp
import random
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

try:
    from .utils import list_spec_items, load_spec_item, write_spec
except ImportError:
    from src.core.utils import list_spec_items, load_spec_item, write_spec

AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac")
DEFAULT_SHARD_SIZE = 2048

def compute_spectrogram(wav, sr=32_000, n_fft=1024, hop_size=160, n_mels=128):
    spec = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_size,
        power=2.0,
        n_mels=n_mels,
        fmin=20,
        fmax=sr // 2,
    )
    return librosa.power_to_db(spec, ref=np.max, top_db=None).astype(np.float32)


def write_audio_params(out_dir, sr=32_000, n_fft=1024, hop_size=160, n_mels=128, storage_dtype="float32"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = {"sr": sr, "mels": n_mels, "hop_size": hop_size, "fft": n_fft}
    if storage_dtype != "float32":
        params["storage_dtype"] = storage_dtype
    (out_dir / "audio_params.json").write_text(json.dumps(params, indent=2) + "\n")


def write_waveform_spectrogram(wav, out_path, sr=32_000, n_fft=1024, hop_size=160, n_mels=128, storage_dtype="float32"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spec = compute_spectrogram(wav, sr, n_fft, hop_size, n_mels)
    write_spec(out_path, spec, storage_dtype)
    return out_path


def audio_file_to_spec(path, out_dir, sr=32_000, n_fft=1024, hop_size=160, n_mels=128, storage_dtype="float32"):
    path = Path(path)
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return write_waveform_spectrogram(
        wav,
        Path(out_dir) / f"{path.stem}.npy",
        sr,
        n_fft,
        hop_size,
        n_mels,
        storage_dtype,
    )


def audio_file_to_spec_from_args(args):
    return audio_file_to_spec(*args)


def audio_file_to_spectrogram(path, sr=32_000, n_fft=1024, hop_size=160, n_mels=128):
    path = Path(path)
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return path.stem, compute_spectrogram(wav, sr, n_fft, hop_size, n_mels)


def audio_file_to_spectrogram_from_args(args):
    return audio_file_to_spectrogram(*args)


def write_shard_index(out_dir, rows):
    shard_dir = Path(out_dir) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    lines = ["name\tshard\tstart\tend"]
    lines += [f"{name}\t{shard}\t{start}\t{end}" for name, shard, start, end in rows]
    (shard_dir / "index.tsv").write_text("\n".join(lines) + "\n")


def write_spec_shard(out_dir, shard_index, rows, storage_dtype="float32"):
    assert rows
    assert storage_dtype in ("float32", "int8_affine")
    shard = f"shard_{shard_index:06d}.npy"
    path = Path(out_dir) / "shards" / shard
    path.parent.mkdir(parents=True, exist_ok=True)

    mels = rows[0][1].shape[0]
    total = sum(spec.shape[1] for _, spec in rows)
    index_rows = []

    if storage_dtype == "float32":
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(total, mels))
        start = 0
        for name, spec in rows:
            end = start + spec.shape[1]
            arr[start:end] = spec.T
            index_rows.append((name, shard, start, end))
            start = end
        del arr
        return index_rows

    low = np.full(mels, np.inf, dtype=np.float32)
    high = np.full(mels, -np.inf, dtype=np.float32)
    for _, spec in rows:
        low = np.minimum(low, spec.min(axis=1))
        high = np.maximum(high, spec.max(axis=1))
    scale = np.maximum((high - low) / 255.0, np.float32(1e-6))
    offset = low + 128.0 * scale

    codes = np.lib.format.open_memmap(path, mode="w+", dtype=np.int8, shape=(total, mels))
    start = 0
    for name, spec in rows:
        end = start + spec.shape[1]
        quantized = np.clip(np.rint((spec - offset[:, None]) / scale[:, None]), -128, 127)
        codes[start:end] = quantized.T.astype(np.int8)
        index_rows.append((name, shard, start, end))
        start = end
    del codes
    np.savetxt(path.with_suffix(".txt"), np.column_stack([scale, offset]), fmt="%.9g")
    return index_rows


def audio_dir_to_specs(src_dir, out_dir, sr=32_000, n_fft=1024, hop_size=160, n_mels=128, workers=1, storage_dtype="float32", shard_size=0):
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    write_audio_params(out_dir, sr, n_fft, hop_size, n_mels, storage_dtype)
    paths = sorted(path for path in src_dir.rglob("*") if path.suffix.lower() in AUDIO_EXTS)
    assert shard_size >= 0
    if shard_size:
        tasks = [(path, sr, n_fft, hop_size, n_mels) for path in paths]
        rows = []
        index_rows = []
        shard_index = 0
        assert workers > 0
        if workers == 1:
            iterator = map(audio_file_to_spectrogram_from_args, tasks)
        else:
            pool = mp.Pool(workers)
            iterator = pool.imap_unordered(audio_file_to_spectrogram_from_args, tasks)
        for name, spec in tqdm(iterator, total=len(tasks), desc="audio2spec"):
            rows.append((name, spec))
            if len(rows) == shard_size:
                index_rows.extend(write_spec_shard(out_dir, shard_index, rows, storage_dtype))
                shard_index += 1
                rows = []
        if workers != 1:
            pool.close()
            pool.join()
        if rows:
            index_rows.extend(write_spec_shard(out_dir, shard_index, rows, storage_dtype))
        write_shard_index(out_dir, index_rows)
        return paths

    tasks = [(path, out_dir, sr, n_fft, hop_size, n_mels, storage_dtype) for path in paths]
    assert workers > 0
    if workers == 1:
        for task in tqdm(tasks, desc="audio2spec"):
            audio_file_to_spec_from_args(task)
    else:
        with mp.Pool(workers) as pool:
            iterator = pool.imap_unordered(audio_file_to_spec_from_args, tasks)
            list(tqdm(iterator, total=len(tasks), desc="audio2spec"))
    return paths


def compute_statistics(spec_dir, sample_fraction=0.1, seed=0):
    rng = random.Random(seed)
    all_items = list_spec_items(spec_dir)
    files = [item for item in all_items if rng.random() < sample_fraction]
    if not files:
        files = all_items

    total_sum = 0.0
    total_sq_sum = 0.0
    total_values = 0
    for item in tqdm(files, desc="computing stats"):
        spec, _ = load_spec_item(item)
        total_sum += spec.sum().item()
        total_sq_sum += np.square(spec).sum().item()
        total_values += spec.size

    mean = total_sum / total_values
    variance = total_sq_sum / total_values - mean * mean
    return mean, max(variance, 0.0) ** 0.5, len(files)


def write_statistics(spec_dir, sample_fraction=0.1, seed=0):
    spec_dir = Path(spec_dir)
    mean, std, num_files = compute_statistics(spec_dir, sample_fraction, seed)
    path = spec_dir / "audio_params.json"
    params = json.loads(path.read_text()) if path.exists() else {}
    params["mean"] = float(mean)
    params["std"] = float(std)
    path.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")
    return mean, std, num_files


def main():
    parser = argparse.ArgumentParser(description="Convert audio files to spectrogram .npy files.")
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--dst_dir", required=True)
    parser.add_argument("--sr", type=int, default=32_000)
    parser.add_argument("--hop_size", type=int, default=160)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--storage_dtype", choices=["float32", "int8_affine"], default="float32")
    parser.add_argument("--shard_size", type=int, default=0)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    audio_dir_to_specs(
        args.src_dir,
        args.dst_dir,
        args.sr,
        args.n_fft,
        args.hop_size,
        args.n_mels,
        args.workers,
        args.storage_dtype,
        args.shard_size,
    )
    if args.stats:
        write_statistics(args.dst_dir)


if __name__ == "__main__":
    main()
