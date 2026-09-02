import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from src.core.utils import load_spec_slice, normalize_spectrogram


def read_rows(path, maximum=None, seed=0):
    rows = {}
    for line in path.open():
        row = json.loads(line)
        if row.get("status") != "ok":
            continue
        tile = row["tile"]
        key = (row["recording"], row["source"]["shard"], tile.get("ownership_start_timebin", tile["start_timebin"]),
            tile.get("ownership_end_timebin", tile["end_timebin"]))
        if "events" in row:
            row = {**row, "boxes": [event for event in row["events"]
                if event["label"] in ("target_vocalization", "uncertain_vocalization", "chorus")]}
        rows[key] = row
    rows = list(rows.values())
    random.Random(seed).shuffle(rows)
    return rows[:maximum] if maximum else rows


def split_rows(rows, fraction, seed):
    recordings = sorted({row["recording"] for row in rows})
    assert len(recordings) > 1, "need at least two recordings for train/validation"
    random.Random(seed).shuffle(recordings)
    count = min(len(recordings) - 1, max(1, round(len(recordings) * fraction)))
    validation = set(recordings[:count])
    return [row for row in rows if row["recording"] not in validation], [row for row in rows if row["recording"] in validation]


class BoxTokens(Dataset):
    def __init__(self, rows, audio, width, patch_height):
        self.audio, self.width, self.patch_height = audio, width, patch_height
        self.height = audio.mels // patch_height
        self.windows = []
        for row in rows:
            tile = row["tile"]
            for start in range(tile["start_timebin"], tile["end_timebin"], width):
                valid = min(width, tile["end_timebin"] - start)
                target = np.zeros((self.height, width), dtype=np.float32)
                for box in row["boxes"]:
                    low, high = max(start, box["start_timebin"]), min(start + valid, box["end_timebin"])
                    if low >= high:
                        continue
                    y0 = box["low_mel_bin"] // patch_height
                    y1 = (box["high_mel_bin"] + patch_height - 1) // patch_height
                    target[y0:y1, low - start:high - start] = 1
                self.windows.append((row, start, valid, target.reshape(-1)))

    def counts(self):
        positive = sum(target.reshape(self.height, self.width)[:, :valid].sum() for _, _, valid, target in self.windows)
        total = sum(self.height * valid for _, _, valid, _ in self.windows)
        return int(positive), int(total - positive)

    def __getitem__(self, index):
        row, start, valid, target = self.windows[index]
        source = row["source"]
        raw = np.zeros((self.audio.mels, self.width), dtype=np.float32)
        raw[:, :valid] = load_spec_slice(source["shard"], source["start"] + start, source["start"] + start + valid)
        spec = normalize_spectrogram(raw, self.audio.mean, self.audio.std)
        return torch.from_numpy(spec).unsqueeze(0), torch.from_numpy(target), valid

    def __len__(self):
        return len(self.windows)
