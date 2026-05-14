#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#6a1fb1", "#0aa02a", "#ff5a00", "#0967d8", "#d33682", "#008b8b"]


def make_conceptual_points(seed, points_per_individual):
    rng = np.random.default_rng(seed)
    specs = [
        (np.linspace(0.0, 2.0 * np.pi, 70), np.array([-2.0, 1.1, 0.6]), "ring"),
        (np.linspace(0.2, 1.8 * np.pi, 72), np.array([1.4, 1.1, 0.7]), "arc"),
        (np.linspace(0.2, 4.8 * np.pi, 90), np.array([-1.7, -1.3, -0.6]), "spiral"),
        (np.linspace(0.0, 1.0, 85), np.array([1.0, -1.5, -0.7]), "wave"),
    ]

    paths = []
    points = []
    labels = []
    for label, (t, center, shape) in enumerate(specs):
        if shape == "ring":
            path = np.column_stack([0.78 * np.cos(t), 0.45 * np.sin(t), 0.24 * np.sin(t + 0.7)])
            path = path @ np.array([[0.86, -0.20, 0.18], [0.25, 0.96, 0.12], [-0.09, 0.16, 0.98]])
        elif shape == "arc":
            path = np.column_stack([0.55 * np.cos(t), 0.34 * np.sin(t), np.linspace(-0.75, 0.95, t.size)])
            path = path @ np.array([[0.74, 0.28, -0.12], [-0.18, 0.90, 0.18], [0.26, 0.05, 1.0]])
        elif shape == "spiral":
            r = np.linspace(0.12, 0.86, t.size)
            path = np.column_stack([r * np.cos(t), r * np.sin(t), 0.16 * np.sin(0.65 * t)])
        else:
            path = np.column_stack([1.55 * t - 0.75, 0.30 * np.sin(10.0 * t), 0.44 * np.cos(6.2 * t)])
            path = path @ np.array([[0.96, 0.10, 0.20], [-0.12, 0.92, 0.15], [-0.14, 0.08, 0.98]])

        path = path + center
        picks = rng.integers(0, path.shape[0], size=points_per_individual)
        cloud = path[picks] + rng.normal(0.0, 0.13, size=(points_per_individual, 3))
        paths.append(path)
        points.append(cloud)
        labels.extend([label] * cloud.shape[0])

    return np.vstack(points), np.asarray(labels), paths


def load_npz_points(path, coords_key, features_key, labels_key):
    data = np.load(path, allow_pickle=True)
    if coords_key in data:
        points = np.asarray(data[coords_key])
    else:
        assert features_key in data, f"Expected '{coords_key}' or '{features_key}' in {path}"
        features = np.asarray(data[features_key], dtype=np.float32)
        n_components = min(3, features.shape[0], features.shape[1])
        assert n_components > 0, features.shape
        points = PCA(n_components=n_components, random_state=0).fit_transform(features)

    assert points.ndim == 2, points.shape
    if points.shape[1] < 3:
        pad = np.zeros((points.shape[0], 3 - points.shape[1]))
        points = np.hstack([points, pad])
    assert points.shape[1] >= 3, points.shape
    points = points[:, :3].astype(np.float32, copy=False)

    assert labels_key in data, f"Expected '{labels_key}' in {path}"
    labels = np.asarray(data[labels_key])
    assert labels.shape[0] == points.shape[0], (labels.shape, points.shape)
    return points, labels, []


def equalize_axes(ax, points):
    center = points.mean(axis=0)
    radius = np.ptp(points, axis=0).max() * 0.52
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def style_axes(ax, show_box, grid_alpha, grid_linewidth):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.tick_params(length=0, pad=-2)
    ax.grid(show_box)
    if not show_box:
        ax.set_axis_off()
        return
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((0.82, 0.82, 0.82, 0.42))
        axis.line.set_color((0.0, 0.0, 0.0, 0.82))
        axis.line.set_linewidth(0.9)
        axis._axinfo["grid"]["color"] = (0.70, 0.70, 0.70, grid_alpha)
        axis._axinfo["grid"]["linewidth"] = grid_linewidth
        axis._axinfo["axisline"]["color"] = (0.65, 0.65, 0.65, 0.22)
        axis._axinfo["tick"]["color"] = (0, 0, 0, 0)


def draw_latent_dimension_scaffold(ax, args):
    if not args.latent_scaffold:
        return

    assert args.latent_scaffold_n_dirs >= 3
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    span = min(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    length = span * args.latent_scaffold_scale
    hub = np.asarray(
        [
            xlim[0] + span * 0.10,
            ylim[1] - span * 0.10,
            zlim[0] + span * 0.12,
        ]
    )

    main_dirs = [
        ("z1", np.asarray([1.0, 0.0, 0.0]), 0.86, 1.25),
        ("z2", np.asarray([0.0, -1.0, 0.0]), 0.86, 1.25),
        ("z3", np.asarray([0.0, 0.0, 1.0]), 0.86, 1.25),
    ]
    color = args.latent_scaffold_color
    ax.scatter(
        hub[0],
        hub[1],
        hub[2],
        s=14,
        color=color,
        alpha=min(0.55, args.latent_scaffold_alpha * 1.8),
        edgecolors="none",
        depthshade=False,
    )

    for label, direction, scale, linewidth in main_dirs:
        draw_scaffold_ray(ax, hub, direction, length * scale, label, color, args.latent_scaffold_alpha * 1.7, linewidth, args)

    n_extra = args.latent_scaffold_n_dirs - 3
    for index, direction in enumerate(extra_scaffold_dirs(n_extra), start=4):
        scale = 0.50 + 0.035 * ((index - 4) % 3)
        draw_scaffold_ray(
            ax,
            hub,
            direction,
            length * scale,
            f"L{index}",
            color,
            args.latent_scaffold_alpha,
            0.65,
            args,
        )


def extra_scaffold_dirs(count):
    dirs = []
    for index in range(count):
        angle = 0.9 * index + 0.35
        z = 0.62 * np.sin(1.35 * index + 0.4)
        x = 0.62 + 0.28 * np.cos(angle)
        y = -0.56 + 0.28 * np.sin(angle)
        dirs.append(np.asarray([x, y, z]))
    return dirs


def draw_scaffold_ray(ax, hub, direction, length, label, color, alpha, linewidth, args):
    direction = direction / np.linalg.norm(direction)
    tip = hub + direction * length
    ax.quiver(
        hub[0],
        hub[1],
        hub[2],
        direction[0] * length,
        direction[1] * length,
        direction[2] * length,
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        arrow_length_ratio=0.10,
        normalize=False,
    )
    ax.text(
        tip[0],
        tip[1],
        tip[2],
        label,
        color=color,
        alpha=min(0.75, alpha * 1.35),
        fontsize=args.latent_scaffold_label_size,
        ha="center",
        va="center",
    )


def draw_axis_arrows(ax, color, corner):
    assert len(corner) == 3
    assert set(corner) <= {"0", "1"}
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    limits = [xlim, ylim, zlim]
    span = min(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    inset = span * 0.025
    length = span * 0.145
    origin = []
    directions = []
    for dim, bit in enumerate(corner):
        low, high = limits[dim]
        sign = 1.0 if bit == "0" else -1.0
        origin.append((low if bit == "0" else high) + sign * inset)
        direction = np.zeros(3)
        direction[dim] = sign * length
        directions.append(direction)
    origin = np.asarray(origin)
    for direction in directions:
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            color=color,
            linewidth=1.15,
            arrow_length_ratio=0.18,
            normalize=False,
        )


def plot_latent_space(points, labels, paths, out_base, args):
    fig = plt.figure(figsize=(args.width, args.height), dpi=args.dpi)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    unique_labels = sorted(set(labels.tolist()))

    equalize_axes(ax, points)
    draw_latent_dimension_scaffold(ax, args)

    for index, label in enumerate(unique_labels):
        color = COLORS[index % len(COLORS)]
        mask = labels == label
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            points[mask, 2],
            s=args.point_size,
            alpha=args.point_alpha,
            color=color,
            edgecolors="none",
            depthshade=False,
        )

    for index, path in enumerate(paths):
        color = COLORS[index % len(COLORS)]
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=args.line_width, alpha=0.95)
        stride = max(1, path.shape[0] // 14)
        nodes = path[::stride]
        ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            nodes[:, 2],
            s=args.node_size,
            color=color,
            edgecolors="white",
            linewidths=0.45,
            depthshade=False,
        )

    if args.axis_arrows:
        draw_axis_arrows(ax, args.arrow_color, args.axis_corner)
    style_axes(ax, args.show_box, args.grid_alpha, args.grid_linewidth)
    ax.view_init(elev=args.elev, azim=args.azim, roll=args.roll)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in args.formats:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if suffix == "png":
            kwargs["transparent"] = args.transparent
            kwargs["dpi"] = args.dpi
        fig.savefig(out_base.with_suffix(f".{suffix}"), **kwargs)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create a clean 3D latent-space panel for individual-ID figures.")
    parser.add_argument("--input_npz", type=Path, default=None)
    parser.add_argument("--coords_key", default="xyz")
    parser.add_argument("--features_key", default="features")
    parser.add_argument("--labels_key", default="bird_labels")
    parser.add_argument("--out", type=Path, default=ROOT / "results/individual_id_latent_space/latent_space")
    parser.add_argument("--points_per_individual", type=int, default=420)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--width", type=float, default=7.2)
    parser.add_argument("--height", type=float, default=6.2)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--elev", type=float, default=19.0)
    parser.add_argument("--azim", type=float, default=-52.0)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--point_size", type=float, default=10.0)
    parser.add_argument("--node_size", type=float, default=34.0)
    parser.add_argument("--line_width", type=float, default=2.2)
    parser.add_argument("--point_alpha", type=float, default=0.14)
    parser.add_argument("--show_box", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--axis_arrows", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--axis_corner", default="010")
    parser.add_argument("--arrow_color", default="#333333")
    parser.add_argument("--grid_alpha", type=float, default=0.38)
    parser.add_argument("--grid_linewidth", type=float, default=0.55)
    parser.add_argument("--latent_scaffold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--latent_scaffold_alpha", type=float, default=0.22)
    parser.add_argument("--latent_scaffold_color", default="#666666")
    parser.add_argument("--latent_scaffold_label_size", type=float, default=8)
    parser.add_argument("--latent_scaffold_scale", type=float, default=0.32)
    parser.add_argument("--latent_scaffold_n_dirs", type=int, default=8)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    if args.input_npz is None:
        points, labels, paths = make_conceptual_points(args.seed, args.points_per_individual)
    else:
        points, labels, paths = load_npz_points(args.input_npz, args.coords_key, args.features_key, args.labels_key)
    plot_latent_space(points, labels, paths, args.out, args)
    print(args.out)


if __name__ == "__main__":
    main()
