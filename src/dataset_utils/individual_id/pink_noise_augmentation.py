#!/usr/bin/env python3
"""Create pink-noise audio and spectrograms projected to reference statistics."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import librosa
import numpy as np
import soundfile as sf

from src.core.audio2spec import AUDIO_EXTS, compute_spectrogram
from src.core.data_structures import AudioParams
from src.core.utils import write_spec


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def add_pink_noise(wav, snr_db, seed):
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(len(wav)))
    frequencies = np.fft.rfftfreq(len(wav))
    spectrum[0] = 0
    spectrum[1:] /= np.sqrt(frequencies[1:])
    noise = np.fft.irfft(spectrum, n=len(wav)).astype(np.float32)
    noise /= rms(noise)
    signal_rms = rms(wav)
    assert signal_rms > 0
    return wav + noise * signal_rms * 10 ** (-snr_db / 20)


def project_statistics(spec, mean, std):
    source_std = float(spec.std(dtype=np.float64))
    assert source_std > 0
    normalized = (spec - spec.mean(dtype=np.float64)) / source_std
    return (normalized * std + mean).astype(np.float32)


def convert(path, source, audio_output, spec_output, params, snr_db, seed):
    wav, _ = librosa.load(path, sr=params.sr, mono=True)
    noisy = add_pink_noise(wav.astype(np.float32), snr_db, seed)
    relative = path.relative_to(source)
    audio_path = (audio_output / relative).with_suffix(".wav")
    spec_path = (spec_output / relative).with_suffix(".npy")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(audio_path, noisy, params.sr, subtype="FLOAT")
    spec = compute_spectrogram(noisy, params.sr, params.fft, params.hop_size, params.mels)
    write_spec(spec_path, project_statistics(spec, params.mean, params.std))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--audio_output_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--stats_dir", type=Path, required=True)
    parser.add_argument("--snr_db", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    params = AudioParams.from_dir(args.stats_dir)
    paths = sorted(path for path in args.audio_dir.rglob("*") if path.suffix.lower() in AUDIO_EXTS)
    assert paths, f"no audio files in {args.audio_dir}"
    args.audio_output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audio_params.json").write_text(json.dumps(params.__dict__, indent=2) + "\n")
    for index, path in enumerate(paths):
        convert(
            path,
            args.audio_dir,
            args.audio_output_dir,
            args.output_dir,
            params,
            args.snr_db,
            args.seed + index,
        )
    print(f"wrote {len(paths)} pink-noise audio files and spectrograms")


if __name__ == "__main__":
    main()
