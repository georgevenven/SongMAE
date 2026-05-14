import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def load_grid(path, rows, cols):
    data = np.load(path, allow_pickle=True)
    xy = data["xy"].astype(np.float32)
    slices = data["slices"].astype(np.int64)
    birds = data["birds"].astype(str)
    selected = []
    for bird in birds:
        if bird not in selected:
            selected.append(bird)
        if len(selected) == rows:
            break
    assert len(selected) == rows

    grid = []
    for bird in selected:
        bird_slices = [pair for pair, row_bird in zip(slices, birds) if row_bird == bird]
        assert len(bird_slices) >= cols, bird
        grid.append(bird_slices[:cols])
    return xy, np.asarray(grid, dtype=np.int64), selected


def draw_frame(xy, grid, birds, frame_index, frame_count, title):
    rows, cols = grid.shape[:2]
    progress = (frame_index + 1) / frame_count
    fig, axes = plt.subplots(rows, cols, figsize=(2.15 * cols, 2.0 * rows), squeeze=False)
    xpad = (xy[:, 0].max() - xy[:, 0].min()) * 0.05
    ypad = (xy[:, 1].max() - xy[:, 1].min()) * 0.05
    xlim = (xy[:, 0].min() - xpad, xy[:, 0].max() + xpad)
    ylim = (xy[:, 1].min() - ypad, xy[:, 1].max() + ypad)
    cmap = plt.get_cmap("viridis")

    for r, bird in enumerate(birds):
        for c in range(cols):
            ax = axes[r, c]
            start, end = grid[r, c]
            pts = xy[start:end]
            count = max(2, int(np.ceil(progress * pts.shape[0])))
            shown = pts[:count]
            if shown.shape[0] > 1:
                segs = np.stack([shown[:-1], shown[1:]], axis=1)
                t = np.linspace(0.0, 1.0, segs.shape[0])
                colors = cmap(t)
                colors[:, 3] = np.linspace(0.05, 0.85, segs.shape[0])
                ax.add_collection(LineCollection(segs, colors=colors, linewidths=1.0))
            ax.scatter(shown[-1, 0], shown[-1, 1], c=[cmap(progress)], s=12, linewidths=0)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{c + 1}", fontsize=8)
            if c == 0:
                ax.set_ylabel(str(bird), fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def parse_args():
    parser = argparse.ArgumentParser(description="Animate selected feature-token UMAP trajectories in a row-by-individual grid.")
    parser.add_argument("--trajectory_npz", required=True)
    parser.add_argument("--out_gif", required=True)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--title", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    xy, grid, birds = load_grid(args.trajectory_npz, args.rows, args.cols)
    frames = [draw_frame(xy, grid, birds, i, args.frames, args.title) for i in range(args.frames)]
    out = Path(args.out_gif)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=args.fps)
    print(out)


if __name__ == "__main__":
    main()
