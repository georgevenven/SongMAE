#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_npz", nargs="+", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args()

    paths = [Path(path) for path in args.feature_npz]
    arrays = [np.load(path, allow_pickle=True) for path in paths]

    features = np.vstack([array["features"] for array in arrays]).astype(np.float32, copy=False)
    bird_labels = np.concatenate([array["bird_labels"] for array in arrays]).astype(object, copy=False)
    syllable_labels = np.concatenate([array["syllable_labels"] for array in arrays]).astype(np.int64, copy=False)
    recording_labels = np.concatenate([array["recording_labels"] for array in arrays]).astype(object, copy=False)
    pairs = {}
    for bird, recording in zip(bird_labels.tolist(), recording_labels.tolist()):
        pairs.setdefault(str(recording), set()).add(str(bird))
    if any(len(birds) > 1 for birds in pairs.values()):
        recording_labels = np.asarray(
            [f"{bird}:{recording}" for bird, recording in zip(bird_labels.tolist(), recording_labels.tolist())],
            dtype=object,
        )

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        features=features,
        bird_labels=bird_labels,
        syllable_labels=syllable_labels,
        recording_labels=recording_labels,
    )

    summary = {
        "feature_path": str(out_npz),
        "points": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "birds": int(len(set(bird_labels.tolist()))),
        "recordings": int(len(set(recording_labels.tolist()))),
        "chunks": [str(path) for path in paths],
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
