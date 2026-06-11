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

AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac")

def compute_spectrogram(wav, sr=32_000, n_fft=1024, hop_size=64, n_mels=128):
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


def write_audio_params(out_dir, sr=32_000, n_fft=1024, hop_size=64, n_mels=128):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = {"sr": sr, "mels": n_mels, "hop_size": hop_size, "fft": n_fft}
    (out_dir / "audio_params.json").write_text(json.dumps(params, indent=2) + "\n")


def write_waveform_spectrogram(wav, out_path, sr=32_000, n_fft=1024, hop_size=64, n_mels=128):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spec = compute_spectrogram(wav, sr, n_fft, hop_size, n_mels)
    np.save(out_path, spec)
    return out_path


def audio_file_to_spec(path, out_dir, sr=32_000, n_fft=1024, hop_size=64, n_mels=128):
    path = Path(path)
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return write_waveform_spectrogram(
        wav,
        Path(out_dir) / f"{path.stem}.npy",
        sr,
        n_fft,
        hop_size,
        n_mels,
    )


def audio_file_to_spec_from_args(args):
    return audio_file_to_spec(*args)


def audio_dir_to_specs(src_dir, out_dir, sr=32_000, n_fft=1024, hop_size=64, n_mels=128, workers=1):
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    write_audio_params(out_dir, sr, n_fft, hop_size, n_mels)
    paths = sorted(path for path in src_dir.rglob("*") if path.suffix.lower() in AUDIO_EXTS)
    tasks = [(path, out_dir, sr, n_fft, hop_size, n_mels) for path in paths]
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
    files = [path for path in Path(spec_dir).glob("*.npy") if rng.random() < sample_fraction]
    if not files:
        files = list(Path(spec_dir).glob("*.npy"))

    total_sum = 0.0
    total_sq_sum = 0.0
    total_values = 0
    for path in tqdm(files, desc="computing stats"):
        spec = np.load(path).astype(np.float32, copy=False)
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
    parser.add_argument("--hop_size", type=int, default=64)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    audio_dir_to_specs(args.src_dir, args.dst_dir, args.sr, args.n_fft, args.hop_size, args.n_mels, args.workers)
    if args.stats:
        write_statistics(args.dst_dir)


if __name__ == "__main__":
    main()
