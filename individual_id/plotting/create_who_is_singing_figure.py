#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial import ConvexHull


ROOT = Path(__file__).resolve().parents[2]
OUT_BASE = ROOT / "results/individual_id_latent_space/who_is_singing"
# Same palette as the 3D latent-space figure: purple, green, orange, blue
COLORS = ["#6a1fb1", "#0aa02a", "#ff5a00", "#0967d8"]

# Quadrant centers per individual (top-down Z1/Z2 plane).
CENTERS = {
    2: np.array([-1.30, 1.30]),   # orange  — top left
    0: np.array([1.30, 1.30]),    # purple  — top right
    3: np.array([-1.30, -1.30]),  # blue    — bottom left
    1: np.array([1.30, -1.30]),   # green   — bottom right
}


def _local_path(label):
    """A wiggly 2D trajectory, centered on the origin, for one individual."""
    t = np.linspace(0.0, 1.0, 60)
    if label == 2:        # orange — wavy horizontal sweep
        return np.column_stack([-0.55 + 1.1 * t, 0.24 * np.sin(2.0 * np.pi * t)])
    if label == 0:        # purple — open C arc
        a = np.linspace(1.15 * np.pi, -0.15 * np.pi, 60)
        return np.column_stack([0.60 * np.cos(a), 0.46 * np.sin(a)])
    if label == 3:        # blue — vertical S
        return np.column_stack([0.44 * np.sin(2.0 * np.pi * t), 1.1 * (t - 0.5)])
    # green — mirrored vertical S
    return np.column_stack([-0.44 * np.sin(2.0 * np.pi * t), 1.1 * (t - 0.5)])


def make_clusters(seed=7, points_per_individual=150):
    rng = np.random.default_rng(seed)
    points, labels, paths = [], [], {}
    for label, center in CENTERS.items():
        path = _local_path(label) + center
        paths[label] = path
        picks = rng.integers(0, path.shape[0], size=points_per_individual)
        cloud = path[picks] + rng.normal(0.0, 0.23, size=(points_per_individual, 2))
        points.append(cloud)
        labels.extend([label] * points_per_individual)
    return np.vstack(points), np.asarray(labels), paths


def blob_outline(pts, pad=0.30, smooth_s=0.4):
    """Smooth, rounded outline that comfortably encloses a cluster's points.

    Built from the convex hull (so no points fall outside), expanded radially
    outward by ``pad`` and rounded with a periodic smoothing spline.
    """
    hull = ConvexHull(pts)
    verts = pts[hull.vertices]              # ordered counter-clockwise
    c = verts.mean(axis=0)

    # Push each hull vertex outward from the centroid by a fixed margin.
    d = verts - c
    dist = np.hypot(d[:, 0], d[:, 1])
    verts = c + d * ((dist + pad) / dist)[:, None]

    # Periodic spline through the expanded hull for a soft, blobby boundary.
    tck, _ = splprep([verts[:, 0], verts[:, 1]], s=smooth_s, per=True)
    u = np.linspace(0.0, 1.0, 400)
    xs, ys = splev(u, tck)
    return xs, ys


def style_panel(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() * 0.62
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal")

    # Light dashed grid behind everything.
    ax.set_xticks(np.linspace(*ax.get_xlim(), 5))
    ax.set_yticks(np.linspace(*ax.get_ylim(), 5))
    ax.grid(True, linestyle=(0, (4, 4)), color=(0.78, 0.78, 0.78), linewidth=0.7, zorder=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_color((0.45, 0.45, 0.45))
        spine.set_linewidth(1.1)


def plot_who_is_singing(out_base):
    points, labels, paths = make_clusters()

    fig, ax = plt.subplots(figsize=(6.6, 6.6), dpi=450)
    style_panel(ax, points)

    for label, color in enumerate(COLORS):
        mask = labels == label
        cloud = points[mask]

        # Cluster blob (soft fill + colored outline).
        bx, by = blob_outline(cloud)
        ax.fill(bx, by, color=color, alpha=0.10, zorder=1, linewidth=0)
        ax.plot(bx, by, color=color, alpha=0.55, linewidth=1.3, zorder=2)

        # Faint point cloud.
        rng = np.random.default_rng(label)
        sizes = rng.uniform(8.0, 38.0, size=cloud.shape[0])
        ax.scatter(cloud[:, 0], cloud[:, 1], s=sizes, color=color, alpha=0.18,
                   edgecolors="none", zorder=3)

        # Trajectory line + white-edged nodes.
        path = paths[label]
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.2, alpha=0.95, zorder=4)
        step = max(1, path.shape[0] // 8)
        nodes = path[::step]
        ax.scatter(nodes[:, 0], nodes[:, 1], s=46.0, color=color,
                   edgecolors="white", linewidths=0.7, zorder=5)

    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(out_base.with_suffix(f".{suffix}"), bbox_inches="tight",
                    pad_inches=0.04, dpi=450)
    plt.close(fig)


def main():
    plot_who_is_singing(OUT_BASE)
    print(OUT_BASE)


if __name__ == "__main__":
    main()
