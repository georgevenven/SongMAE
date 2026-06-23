#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.colors import BoundaryNorm, ListedColormap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plotting_utils.plotting_utils import SPEC_DPI, SPEC_IMSHOW_KW, SPEC_TITLE_KW, save_fig


DETECT_CMAP = ListedColormap([[0.05, 0.05, 0.05], [0.90, 0.20, 0.20]])
BACKGROUND = [0.05, 0.05, 0.05]
CROP_SECONDS = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot full spectrograms with detection and unit label bars.")
    parser.add_argument("spec_dir", type=Path)
    parser.add_argument("annotation_json", type=Path)
    parser.add_argument("--extra_spec_dir", action="append", type=Path, default=[])
    parser.add_argument("--out_dir", type=Path, default=Path("tmp"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def audio_params(spec_dir: Path) -> tuple[int, int, int]:
    data = json.loads((spec_dir / "audio_params.json").read_text())
    return int(data["sr"]), int(data["hop_size"]), int(data["mels"])


def ms_to_bin(ms: float, sr: int, hop: int, limit: int) -> int:
    return max(0, min(limit, int(ms / 1000.0 * sr / hop)))


def stem_aliases(stem: str) -> list[str]:
    aliases = [stem]
    if "__" in stem:
        aliases.append(stem.split("__", 1)[1])
    if stem.startswith("XC") and "_" in stem:
        aliases.append(stem.split("_", 1)[0])
    return aliases


def annotation_map(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    annotations = {}
    for item in data["recordings"]:
        for alias in stem_aliases(Path(item["recording"]["filename"]).stem):
            annotations.setdefault(alias, item)
    return annotations


def recording_for_path(path: Path, annotations: dict[str, dict]) -> dict | None:
    for alias in stem_aliases(path.stem):
        if alias in annotations:
            return annotations[alias]
    return None


def unit_id_map(annotations: dict[str, dict]) -> dict[int, int]:
    ids = sorted({
        int(unit["id"])
        for item in annotations.values()
        for event in item.get("detected_events", [])
        for unit in event.get("units", [])
    })
    return {unit_id: index + 1 for index, unit_id in enumerate(ids)}


def unit_cmap(unit_count: int) -> ListedColormap:
    names = ["tab20", "tab20b", "tab20c"]
    colors = [mcolors.to_rgb(color) for name in names for color in plt.get_cmap(name).colors]
    if unit_count > len(colors):
        colors = [
            mcolors.hsv_to_rgb(((index * 0.61803398875) % 1.0, 0.70, 0.95))
            for index in range(unit_count)
        ]
    return ListedColormap([BACKGROUND, *colors[:unit_count]])


def unit_norm(unit_count: int) -> BoundaryNorm:
    return BoundaryNorm(np.arange(unit_count + 2) - 0.5, unit_count + 1)


def spec_paths(spec_dirs: list[Path], annotations: dict[str, dict]) -> list[Path]:
    paths = [
        path
        for spec_dir in spec_dirs
        for path in spec_dir.glob("*.npy")
        if recording_for_path(path, annotations)
    ]
    assert paths, f"No matching .npy files for {spec_dirs} and annotation JSON."
    return sorted(paths)


def display_spec(path: Path, mels: int) -> np.ndarray:
    spec = np.load(path)
    if spec.shape[1] == mels:
        return spec.T
    assert spec.shape[0] == mels, f"Expected one spectrogram axis to equal {mels}: {path} {spec.shape}"
    return spec


def tracks(recording: dict, unit_ids: dict[int, int], width: int, sr: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    detection = np.zeros(width, dtype=np.int64)
    units = np.zeros(width, dtype=np.int64)
    for event in recording.get("detected_events", []):
        start = ms_to_bin(event["onset_ms"], sr, hop, width)
        stop = ms_to_bin(event["offset_ms"], sr, hop, width)
        stop = max(start + 1, stop)
        detection[start:stop] = 1
        for unit in event.get("units", []):
            unit_start = ms_to_bin(unit["onset_ms"], sr, hop, width)
            unit_stop = ms_to_bin(unit["offset_ms"], sr, hop, width)
            unit_stop = max(unit_start + 1, unit_stop)
            units[unit_start:unit_stop] = unit_ids[int(unit["id"])]
    return detection, units


def label_ax(ax, label: str) -> None:
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_one(path: Path, recording: dict, unit_ids: dict[int, int], out_path: Path, sr: int, hop: int, mels: int) -> None:
    spec = display_spec(path, mels)
    crop = ms_to_bin(CROP_SECONDS * 1000.0, sr, hop, spec.shape[1])
    spec = spec[:, :crop]
    detection, units = tracks(recording, unit_ids, spec.shape[1], sr, hop)
    seconds = spec.shape[1] * hop / sr

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 4.0),
        dpi=SPEC_DPI,
        sharex=True,
        gridspec_kw={"height_ratios": [12, 1, 1], "hspace": 0.05},
    )
    axes[0].imshow(spec, extent=(0, seconds, 0, spec.shape[0]), cmap="viridis", **SPEC_IMSHOW_KW)
    axes[0].set_yticks([])
    axes[0].set_title(path.stem, **SPEC_TITLE_KW)
    for spine in axes[0].spines.values():
        spine.set_visible(False)

    axes[1].imshow(
        detection[None, :],
        extent=(0, seconds, 0, 1),
        aspect="auto",
        cmap=DETECT_CMAP,
        norm=BoundaryNorm([-0.5, 0.5, 1.5], 2),
        interpolation="none",
    )
    label_ax(axes[1], "detect")
    axes[2].imshow(
        units[None, :],
        extent=(0, seconds, 0, 1),
        aspect="auto",
        cmap=unit_cmap(len(unit_ids)),
        norm=unit_norm(len(unit_ids)),
        interpolation="none",
    )
    label_ax(axes[2], "units")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_xlim(0, seconds)

    save_fig(fig, out_path)


def main() -> None:
    args = parse_args()
    spec_dirs = [args.spec_dir, *args.extra_spec_dir]
    sr, hop, mels = audio_params(args.spec_dir)
    assert all(audio_params(path) == (sr, hop, mels) for path in spec_dirs), "spec dirs must share audio_params"
    annotations = annotation_map(args.annotation_json)
    unit_ids = unit_id_map(annotations)
    paths = spec_paths(spec_dirs, annotations)
    random.Random(args.seed).shuffle(paths)

    out_dir = args.out_dir / args.annotation_json.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths[: args.n]:
        recording = recording_for_path(path, annotations)
        assert recording is not None, path
        plot_one(path, recording, unit_ids, out_dir / f"{path.stem}.png", sr, hop, mels)
    print(f"Wrote {min(args.n, len(paths))} plots to {out_dir}")


if __name__ == "__main__":
    main()
