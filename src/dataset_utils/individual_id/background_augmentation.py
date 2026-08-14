#!/usr/bin/env python3
"""Mix one foreground recording with one background recording at 0 dB SNR."""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SAMPLE_RATE = 32_000


def load_audio(path):
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return wav.astype(np.float32)


def rms(wav):
    return float(np.sqrt(np.mean(np.square(wav, dtype=np.float64))))


def colored_noise(length, seed, color):
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(length))
    frequencies = np.fft.rfftfreq(length)
    spectrum[0] = 0
    if color == "pink":
        spectrum[1:] /= np.sqrt(frequencies[1:])
    elif color == "brown":
        spectrum[1:] /= frequencies[1:]
    elif color == "green":
        center = 500 / SAMPLE_RATE
        spectrum[1:] *= np.exp(-0.5 * np.square(np.log2(frequencies[1:] / center)))
    else:
        assert color == "white"
    noise = np.fft.irfft(spectrum, n=length).astype(np.float32)
    return noise / rms(noise)


def add_colored_noise_0db(wav, mask, seed, color):
    assert mask.dtype == bool and mask.shape == wav.shape and mask.any()
    noise = colored_noise(len(wav), seed, color)
    event_rms = rms(wav[mask])
    gain = event_rms / rms(noise[mask])
    mixed = wav + gain * noise
    return mixed.astype(np.float32), gain, event_rms, float(np.max(np.abs(mixed)))


def add_pink_noise_0db(wav, mask, seed):
    return add_colored_noise_0db(wav, mask, seed, "pink")


def mix_0db(foreground, background, seed):
    assert len(background) >= len(foreground)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(len(background) - len(foreground) + 1))
    background = background[start : start + len(foreground)]
    foreground_rms = rms(foreground)
    background_rms = rms(background)
    assert foreground_rms > 0 and background_rms > 0
    gain = foreground_rms / background_rms
    mixed = foreground + gain * background
    return mixed.astype(np.float32), start, gain, foreground_rms, background_rms


def augment(foreground_path, background_path, output_path, seed):
    foreground = load_audio(foreground_path)
    background = load_audio(background_path)
    mixed, start, gain, foreground_rms, background_rms = mix_0db(
        foreground, background, seed
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, mixed, SAMPLE_RATE, subtype="FLOAT")
    return {
        "foreground": str(foreground_path),
        "background": str(background_path),
        "output": str(output_path),
        "sample_rate": SAMPLE_RATE,
        "snr_db": 0,
        "seed": seed,
        "background_start_sample": start,
        "background_start_s": start / SAMPLE_RATE,
        "foreground_rms": foreground_rms,
        "background_rms": background_rms,
        "background_gain": gain,
        "mixed_peak": float(np.max(np.abs(mixed))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("foreground")
    parser.add_argument("background")
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(augment(args.foreground, args.background, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
