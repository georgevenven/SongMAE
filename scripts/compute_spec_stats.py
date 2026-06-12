#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.audio2spec import write_statistics


def main():
    parser = argparse.ArgumentParser(description="Compute mean/std for spectrogram .npy files.")
    parser.add_argument("--spec_dir", type=Path, required=True)
    parser.add_argument("--sample_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mean, std, num_files = write_statistics(args.spec_dir, args.sample_fraction, args.seed)
    print(f"Processed: {num_files} files")
    print(f"Mean: {mean:.6f}")
    print(f"Std: {std:.6f}")
    print(f"Saved to {args.spec_dir / 'audio_params.json'}")


if __name__ == "__main__":
    main()
