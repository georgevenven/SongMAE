#!/usr/bin/env python3
"""Plot the longest annotated non-song interval in each zebra-finch recording."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.core.utils import load_spec


RESULTS = ROOT / "results/individual_id/zebra_finch_pink_noise_event_probe"
SPECS = Path("/media/george-vengrovski/disk2/zebra_finch_pink_noise_event_probe/specs")
OUTPUT = RESULTS / "non_song_audit_pink_m18db"
HOP_MS = 5
MAX_SECONDS = 5


def longest_gap(events, duration_ms):
    intervals = sorted((event["onset_ms"], event["offset_ms"]) for event in events)
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    gaps = [(0, merged[0][0])]
    gaps += [(left[1], right[0]) for left, right in zip(merged, merged[1:])]
    gaps.append((merged[-1][1], duration_ms))
    start, end = max(gaps, key=lambda gap: gap[1] - gap[0])
    if end - start > MAX_SECONDS * 1000:
        middle = (start + end) / 2
        start = middle - MAX_SECONDS * 500
        end = middle + MAX_SECONDS * 500
    return start, end


def draw(ax, spec, title):
    duration = spec.shape[1] * HOP_MS / 1000
    ax.imshow(spec, origin="lower", aspect="auto", vmin=-80, vmax=0, cmap="magma", extent=(0, duration, 0, 16))
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Seconds within crop")
    ax.set_ylabel("kHz")


def main():
    assert not OUTPUT.exists(), OUTPUT
    images = OUTPUT / "spectrograms"
    sheets = OUTPUT / "contact_sheets"
    images.mkdir(parents=True)
    sheets.mkdir()

    annotations = {
        Path(row["recording"]["filename"]).stem: row
        for row in json.loads((RESULTS / "annotations.json").read_text())["recordings"]
    }
    rows = []
    for stem in sorted(annotations):
        clean = load_spec(SPECS / "clean" / f"{stem}.npy")
        noisy = load_spec(SPECS / "pink_m18db" / f"{stem}.npy")
        start_ms, end_ms = longest_gap(annotations[stem]["detected_events"], clean.shape[1] * HOP_MS)
        start = round(start_ms / HOP_MS)
        end = round(end_ms / HOP_MS)
        row = {
            "stem": stem,
            "bird_id": str(annotations[stem]["recording"]["bird_id"]),
            "start_ms": start * HOP_MS,
            "end_ms": end * HOP_MS,
        }
        rows.append((row, clean[:, start:end], noisy[:, start:end]))

    for index in range(0, len(rows), 5):
        batch = rows[index : index + 5]
        fig, axes = plt.subplots(len(batch), 2, figsize=(14, 2.2 * len(batch)), squeeze=False)
        for (row, clean, noisy), pair in zip(batch, axes):
            title = f'{row["bird_id"]} | {row["stem"]} | {row["start_ms"] / 1000:.2f}-{row["end_ms"] / 1000:.2f} s'
            draw(pair[0], clean, f"clean non-song | {title}")
            draw(pair[1], noisy, "same interval | 8x pink noise")
            one, one_axes = plt.subplots(2, 1, figsize=(12, 5))
            draw(one_axes[0], clean, f"clean non-song | {title}")
            draw(one_axes[1], noisy, "same interval | 8x pink noise")
            one.tight_layout()
            one.savefig(images / f'{row["stem"]}.png', dpi=150)
            plt.close(one)
        fig.tight_layout()
        fig.savefig(sheets / f"non_song_{index + 1:03d}_{index + len(batch):03d}.png", dpi=150)
        plt.close(fig)

    (OUTPUT / "manifest.json").write_text(json.dumps({
        "selection": "longest interval outside all merged outer detected_events, centered crop capped at five seconds",
        "condition": "clean versus 8x pink noise at -18.06 dB event SNR",
        "recordings": [row for row, _, _ in rows],
    }, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
