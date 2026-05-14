import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Animate 10 individuals in one saved 2D latent space.")
    parser.add_argument("--trajectory_npz", required=True)
    parser.add_argument("--out_gif", required=True)
    parser.add_argument("--individuals", type=int, default=10)
    parser.add_argument("--songs", type=int, default=10)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--duration_ms", type=int, default=70)
    return parser.parse_args()


def pick_indices(birds, individuals, songs):
    selected = {}
    for bird in sorted(set(birds), key=str):
        indices = np.flatnonzero(birds == bird)
        if len(indices) >= songs:
            selected[str(bird)] = indices[:songs]
        if len(selected) == individuals:
            return selected
    assert False, (len(selected), individuals)


def frame_image(paths_by_bird, lo, hi, step, frames):
    cols = 5
    rows = 2
    colors = plt.get_cmap("turbo", 10)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 6), dpi=130)
    for ax, (bird, paths) in zip(axes.ravel(), paths_by_bird.items()):
        for song, points in enumerate(paths):
            end = 2 + int((len(points) - 2) * step / max(1, frames - 1))
            color = colors(song)
            ax.plot(points[:, 0], points[:, 1], color="0.82", linewidth=0.8, alpha=0.45)
            ax.plot(points[:end, 0], points[:end, 1], color=color, linewidth=1.7, alpha=0.95)
            ax.scatter(points[end - 1, 0], points[end - 1, 1], color=color, s=14, linewidths=0)
        ax.set_title(f"individual {bird}", fontsize=9)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("0.82")

    fig.suptitle("Ovenbird: 10 individuals embedded together, 10 songs each", fontsize=13)
    fig.tight_layout()
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return image


def main():
    args = parse_args()
    data = np.load(args.trajectory_npz, allow_pickle=True)
    xy = data["xy"].astype(np.float32)
    slices = data["slices"].astype(np.int64)
    birds = data["birds"].astype(str)
    recordings = data["recordings"].astype(str)

    selected = pick_indices(birds, args.individuals, args.songs)
    paths_by_bird = {}
    recordings_by_bird = {}
    for bird, indices in selected.items():
        paths_by_bird[bird] = [xy[start:end] for start, end in slices[indices]]
        recordings_by_bird[bird] = recordings[indices].tolist()

    all_xy = np.vstack([path for paths in paths_by_bird.values() for path in paths])
    pad = (all_xy.max(axis=0) - all_xy.min(axis=0)) * 0.06
    lo = all_xy.min(axis=0) - pad
    hi = all_xy.max(axis=0) + pad
    frames = [
        frame_image(paths_by_bird, lo, hi, step, args.frames)
        for step in range(args.frames)
    ]

    out_gif = Path(args.out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=args.duration_ms, loop=0)

    summary = {
        "trajectory_npz": args.trajectory_npz,
        "out_gif": str(out_gif),
        "individuals": list(paths_by_bird),
        "recordings_by_individual": recordings_by_bird,
        "songs_per_individual": args.songs,
        "frames": args.frames,
        "duration_ms": args.duration_ms,
    }
    out_gif.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
