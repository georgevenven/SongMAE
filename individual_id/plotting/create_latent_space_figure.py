#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from scipy.spatial import KDTree


ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "results/individual_id_latent_space/latent_space"
COLORS = ["#6a1fb1", "#0aa02a", "#ff5a00", "#0967d8"]


def make_points(seed=9, points_per_individual=420):
    rng = np.random.default_rng(seed)
    specs = [
        (np.linspace(0.0, 2.0 * np.pi, 70), np.array([-1.20, 0.76, 0.44]), "ring"),
        (np.linspace(0.2, 1.8 * np.pi, 72), np.array([1.08, 0.84, 0.48]), "arc"),
        (np.linspace(0.2, 4.8 * np.pi, 90), np.array([-1.10, -0.88, -0.34]), "spiral"),
        (np.linspace(0.0, 1.0, 85), np.array([0.84, -1.04, -0.44]), "wave"),
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
            radius = np.linspace(0.12, 0.86, t.size)
            path = np.column_stack([radius * np.cos(t), radius * np.sin(t), 0.16 * np.sin(0.65 * t)])
        else:
            path = np.column_stack([1.55 * t - 0.75, 0.30 * np.sin(10.0 * t), 0.44 * np.cos(6.2 * t)])
            path = path @ np.array([[0.96, 0.10, 0.20], [-0.12, 0.92, 0.15], [-0.14, 0.08, 0.98]])

        path = path + center
        picks = rng.integers(0, path.shape[0], size=points_per_individual)
        points.append(path[picks] + rng.normal(0.0, 0.13, size=(points_per_individual, 3)))
        paths.append(path)
        labels.extend([label] * points_per_individual)

    return np.vstack(points), np.asarray(labels), paths


def equalize_axes(ax, points):
    mins = np.percentile(points, 0.5, axis=0)
    maxs = np.percentile(points, 99.5, axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() * 0.53
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xticks(np.linspace(*ax.get_xlim3d(), 7))
    ax.set_yticks(np.linspace(*ax.get_ylim3d(), 7))
    ax.set_zticks(np.linspace(*ax.get_zlim3d(), 7))


def style_axes(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.tick_params(length=0, pad=-2)
    ax.grid(False)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color((0.0, 0.0, 0.0, 0.0))
        axis.line.set_linewidth(0.9)
        axis._axinfo["grid"]["color"] = (0, 0, 0, 0)
        axis._axinfo["grid"]["linewidth"] = 0
        axis._axinfo["axisline"]["color"] = (0, 0, 0, 0)
        axis._axinfo["tick"]["inward_factor"] = 0
        axis._axinfo["tick"]["outward_factor"] = 0
        axis._axinfo["tick"]["linewidth"] = {True: 0, False: 0}
        axis._axinfo["tick"]["color"] = (0, 0, 0, 0)


def draw_grid(ax):
    xmin, xmax = ax.get_xlim3d()
    ymin, ymax = ax.get_ylim3d()
    zmin, zmax = ax.get_zlim3d()
    color = (0.70, 0.70, 0.70, 0.32)

    for x in ax.get_xticks():
        ax.plot([x, x], [ymin, ymax], [zmin, zmin], color=color, linewidth=0.55)
        ax.plot([x, x], [ymax, ymax], [zmin, zmax], color=color, linewidth=0.55)
    for y in ax.get_yticks():
        ax.plot([xmin, xmax], [y, y], [zmin, zmin], color=color, linewidth=0.55)
        ax.plot([xmin, xmin], [y, y], [zmin, zmax], color=color, linewidth=0.55)
    for z in ax.get_zticks():
        ax.plot([xmin, xmax], [ymax, ymax], [z, z], color=color, linewidth=0.55)
        ax.plot([xmin, xmin], [ymin, ymax], [z, z], color=color, linewidth=0.55)


def draw_origin_axes(ax):
    xmin, xmax = ax.get_xlim3d()
    ymin, ymax = ax.get_ylim3d()
    zmin, zmax = ax.get_zlim3d()
    origin = np.array([xmin, ymax, zmin])
    axes = [
        np.array([xmax, ymax, zmin]),
        np.array([xmin, ymin, zmin]),
        np.array([xmin, ymax, zmax]),
    ]
    color = (0, 0, 0, 0.82)

    for end in axes:
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            color=color,
            linewidth=0.9,
            zorder=0,
        )


def draw_path_arrow(ax, path, color):
    pass


def draw_neighborhood_inset(fig, ax, points, labels, anchor_idx, k,
                            inset_center, inset_radius, seed=42,
                            forced_mix=None):
    """Draw a circular inset showing the k-nearest neighbors of points[anchor_idx].

    forced_mix: optional dict {label: count} to override the actual neighbor composition.
    """
    rng = np.random.default_rng(seed)

    if forced_mix is not None:
        neighbor_labels = np.concatenate(
            [np.full(count, lbl, dtype=int) for lbl, count in forced_mix.items()]
        )
        rng.shuffle(neighbor_labels)
        k = len(neighbor_labels)
    else:
        tree = KDTree(points)
        _, neighbor_idxs = tree.query(points[anchor_idx], k=k)
        neighbor_labels = labels[neighbor_idxs]

    # Project the anchor point from 3D to 2D display coordinates
    x3, y3, z3 = points[anchor_idx]
    x2d, y2d, _ = proj3d.proj_transform(x3, y3, z3, ax.get_proj())
    # Convert data coords to figure fraction
    disp = ax.transData.transform((x2d, y2d))
    anchor_fig = fig.transFigure.inverted().transform(disp)

    # Create the inset axes (circular region)
    inset_size = inset_radius * 2
    inset_ax = fig.add_axes(
        [inset_center[0] - inset_radius, inset_center[1] - inset_radius,
         inset_size, inset_size],
    )
    inset_ax.set_xlim(-1.15, 1.15)
    inset_ax.set_ylim(-1.15, 1.15)
    inset_ax.set_aspect("equal")
    inset_ax.axis("off")

    # Draw the circle boundary
    circle = mpatches.Circle((0, 0), 1.0, fill=True, facecolor="white",
                              edgecolor=(0.45, 0.45, 0.45, 0.9), linewidth=1.2,
                              zorder=0)
    inset_ax.add_patch(circle)
    inset_ax.set_clip_on(False)

    # Place neighbor dots inside the circle
    angles = rng.uniform(0, 2 * np.pi, size=k)
    radii = np.sqrt(rng.uniform(0.0, 0.82**2, size=k))
    dot_x = radii * np.cos(angles)
    dot_y = radii * np.sin(angles)

    for i in range(k):
        c = COLORS[neighbor_labels[i]]
        inset_ax.scatter(dot_x[i], dot_y[i], s=55, color=c, alpha=0.55,
                         edgecolors="white", linewidths=0.3, zorder=2)

    # Highlight the center point
    inset_ax.scatter(0, 0, s=90, color=COLORS[labels[anchor_idx]],
                     edgecolors="black", linewidths=0.7, zorder=3)

    # Draw connector line from anchor to inset edge
    # Find the point on the inset circle edge closest to the anchor
    inset_fig_center = np.array(inset_center)
    direction = anchor_fig - inset_fig_center
    dist = np.linalg.norm(direction)
    if dist > 0:
        edge_point = inset_fig_center + direction / dist * inset_radius
    else:
        edge_point = inset_fig_center

    fig.add_artist(plt.Line2D(
        [anchor_fig[0], edge_point[0]],
        [anchor_fig[1], edge_point[1]],
        transform=fig.transFigure,
        color=(0.4, 0.4, 0.4, 0.7),
        linewidth=0.8,
        linestyle="--",
        zorder=5,
    ))


def plot_latent_space(out_base):
    points, labels, paths = make_points()

    fig = plt.figure(figsize=(7.2, 6.2), dpi=450)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    equalize_axes(ax, np.vstack([points, *paths]))
    style_axes(ax)
    ax.view_init(elev=19.0, azim=-52.0, roll=0.0)
    draw_grid(ax)
    draw_origin_axes(ax)

    for label, color in enumerate(COLORS):
        mask = labels == label
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            points[mask, 2],
            s=10.0,
            alpha=0.14,
            color=color,
            edgecolors="none",
            depthshade=False,
        )

    for path, color in zip(paths, COLORS):
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=2.2, alpha=0.95)
        step = max(1, path.shape[0] // 14)
        node_idxs = list(range(0, path.shape[0], step))
        nodes = path[node_idxs]
        ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            nodes[:, 2],
            s=34.0,
            color=color,
            edgecolors="white",
            linewidths=0.45,
            depthshade=False,
        )
        draw_path_arrow(ax, path, color)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Force a draw so 3D projections are computed before we place insets
    fig.canvas.draw()

    # Pick 3 anchor points from different individuals for the insets
    # individual 0=purple, 1=green, 2=orange, 3=blue
    points_per_ind = 420

    # Find an orange point closest to the blue cloud (orange-blue boundary)
    orange_pts = points[labels == 2]
    blue_pts = points[labels == 3]
    blue_center = blue_pts.mean(axis=0)
    orange_dists = np.linalg.norm(orange_pts - blue_center, axis=1)
    boundary_orange_idx = 2 * points_per_ind + np.argmin(orange_dists)

    inset_configs = [
        # Purple — top right, ~90% pure (with a bit of green mixed in)
        {"anchor": 0 * points_per_ind + 200, "inset_center": (0.72, 0.75),
         "seed": 10, "forced_mix": {0: 27, 1: 3}},
        # Green — bottom right, ~90% pure (with a bit of blue mixed in)
        {"anchor": 1 * points_per_ind + 250, "inset_center": (0.75, 0.22),
         "seed": 20, "forced_mix": {1: 27, 3: 3}},
        # Orange — top left, ~50% purity (orange/blue mix), anchored at boundary
        {"anchor": boundary_orange_idx, "inset_center": (0.18, 0.75),
         "seed": 30, "forced_mix": {2: 15, 3: 15}},
    ]
    for cfg in inset_configs:
        draw_neighborhood_inset(
            fig, ax, points, labels,
            anchor_idx=cfg["anchor"],
            k=30,
            inset_center=cfg["inset_center"],
            inset_radius=0.09,
            seed=cfg["seed"],
            forced_mix=cfg.get("forced_mix"),
        )

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(out_base.with_suffix(f".{suffix}"), bbox_inches="tight", pad_inches=0.02, dpi=450)
    plt.close(fig)


def _make_affinity_matrix(n_per_singer=12, n_singers=4, within_strength=0.28,
                          across_strength=0.22, seed=42):
    """Create a synthetic affinity matrix with block structure."""
    rng = np.random.default_rng(seed)
    n = n_per_singer * n_singers
    labels = np.repeat(np.arange(n_singers), n_per_singer)
    # Start with uniform noise everywhere
    A = rng.uniform(0.05, 0.55, size=(n, n))
    # Add a modest boost for same-singer pairs
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                A[i, j] += rng.uniform(within_strength, within_strength + 0.35)
            else:
                A[i, j] += rng.uniform(0, across_strength)
    # Make symmetric and clip
    A = (A + A.T) / 2
    A = np.clip(A, 0, 1)
    np.fill_diagonal(A, 0.55 + rng.uniform(0, 0.35, size=n))
    return A, labels


def _colorize_affinity(A, labels):
    """Create an RGB image where each cell is tinted by its singer color."""
    from matplotlib.colors import to_rgb
    n = A.shape[0]
    img = np.ones((n, n, 3))
    singer_rgbs = [np.array(to_rgb(c)) for c in COLORS]
    off_diag_rgb = np.array([0.85, 0.65, 0.30])  # warm tan for cross-singer

    for i in range(n):
        for j in range(n):
            val = A[i, j]
            if labels[i] == labels[j]:
                rgb = singer_rgbs[labels[i]]
            else:
                rgb = off_diag_rgb
            # Blend white -> color based on affinity value
            img[i, j] = 1.0 - val * (1.0 - rgb)
    return img


def _style_matrix_ax(ax, n, xlabel, ylabel):
    """Apply clean styling to a matrix axes."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=7, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=7, labelpad=4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color((0.3, 0.3, 0.3))


def plot_affinity_unordered(out_base):
    """Unordered affinity matrix — rows/cols in random order, no identity info."""
    from matplotlib.colors import LinearSegmentedColormap
    A, labels = _make_affinity_matrix()
    n = A.shape[0]
    rng = np.random.default_rng(99)
    perm = rng.permutation(n)
    A_shuffled = A[np.ix_(perm, perm)]

    cmap = LinearSegmentedColormap.from_list(
        "warmcool", ["#ffffff", "#fff3e0", "#ffcc80", "#ffab40", "#e68a00"], N=256
    )

    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=450)
    ax.imshow(A_shuffled, cmap=cmap, vmin=0, vmax=1, aspect="equal", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color((0.3, 0.3, 0.3))

    fig.tight_layout()
    out_path = out_base.parent / "affinity_unordered"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(out_path.with_suffix(f".{suffix}"), bbox_inches="tight",
                    pad_inches=0.04, dpi=450)
    plt.close(fig)
    return out_path


def plot_affinity_sorted(out_base):
    """Sorted-by-singer affinity matrix with block-diagonal structure, no ID colors."""
    from matplotlib.colors import LinearSegmentedColormap
    A, labels = _make_affinity_matrix()
    n = A.shape[0]
    sort_idx = np.argsort(labels, kind="stable")
    A_sorted = A[np.ix_(sort_idx, sort_idx)]

    cmap = LinearSegmentedColormap.from_list(
        "warmcool", ["#ffffff", "#fff3e0", "#ffcc80", "#ffab40", "#e68a00"], N=256
    )

    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=450)
    ax.imshow(A_sorted, cmap=cmap, vmin=0, vmax=1, aspect="equal", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color((0.3, 0.3, 0.3))

    fig.tight_layout()
    out_path = out_base.parent / "affinity_sorted"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(out_path.with_suffix(f".{suffix}"), bbox_inches="tight",
                    pad_inches=0.04, dpi=450)
    plt.close(fig)
    return out_path


def plot_stable_rank(out_base):
    """Singular-value / stable-rank view: 1, 2, 4 singers side by side."""

    fig = plt.figure(figsize=(8.5, 4.2), dpi=450)

    configs = [
        {"n_singers": 2, "title": "2 singers"},
        {"n_singers": 3, "title": "3 singers"},
        {"n_singers": 4, "title": "4 singers"},
    ]

    # Fixed total matrix size — same grid for all panels
    n_total = 48
    for col_idx, cfg in enumerate(configs):
        ns = cfg["n_singers"]
        n_per = n_total // ns
        A, labels = _make_affinity_matrix(n_per_singer=n_per, n_singers=ns, seed=42 + col_idx)

        # Matrix subplot (top)
        ax_mat = fig.add_axes([
            0.04 + col_idx * 0.33,  # left
            0.38,                    # bottom
            0.22,                    # width
            0.52,                    # height
        ])
        # Sort by label for block-diagonal structure
        sort_idx = np.argsort(labels, kind="stable")
        A_sorted = A[np.ix_(sort_idx, sort_idx)]
        labels_sorted = labels[sort_idx]
        img = _colorize_affinity(A_sorted, labels_sorted)
        ax_mat.imshow(img, aspect="equal", interpolation="nearest")

        ax_mat.set_xticks([])
        ax_mat.set_yticks([])
        for spine in ax_mat.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color((0.3, 0.3, 0.3))
        ax_mat.set_title(cfg["title"], fontsize=8, fontweight="bold", pad=5)

        # Vector diagram (bottom row)
        ax_vec = fig.add_axes([
            0.04 + col_idx * 0.33,  # left
            0.02,                    # bottom
            0.22,                    # width
            0.30,                    # height
        ])
        ax_vec.set_xlim(-1.4, 1.4)
        ax_vec.set_ylim(-1.4, 1.4)
        ax_vec.set_aspect("equal")
        ax_vec.axis("off")

        # Draw a small dot at origin
        ax_vec.scatter(0, 0, s=25, color="black", zorder=5)

        # Spread arrows evenly; angles chosen to look like the reference
        base_angles = {
            2: [140, -20],
            3: [150, -10, -80],
            4: [120, -10, 160, -70],
        }
        arrow_len = 1.0
        angles = base_angles[ns]
        for ai in range(ns):
            angle_rad = np.radians(angles[ai])
            dx = arrow_len * np.cos(angle_rad)
            dy = arrow_len * np.sin(angle_rad)
            ax_vec.annotate(
                "", xy=(dx, dy), xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.25,head_length=0.15",
                    color=COLORS[ai], lw=2.0,
                ),
            )
            # Label
            label_offset = 0.22
            lx = (arrow_len + label_offset) * np.cos(angle_rad)
            ly = (arrow_len + label_offset) * np.sin(angle_rad)
            ax_vec.text(lx, ly, f"$u_{{{ai+1}}}$", fontsize=9, fontweight="bold",
                        color=COLORS[ai], ha="center", va="center")

        # Arrow between columns
        if col_idx < 2:
            arrow_x = 0.04 + (col_idx + 1) * 0.33 - 0.04
            fig.text(arrow_x, 0.62, "\u25B6", fontsize=14, ha="center", va="center",
                     color=(0.55, 0.55, 0.55))

    # Bottom label
    fig.text(0.5, -0.02, "stable rank  ≈  # singers", fontsize=9, ha="center",
             fontweight="bold", style="italic", color=(0.25, 0.25, 0.25))

    out_path = out_base.parent / "stable_rank"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(out_path.with_suffix(f".{suffix}"), bbox_inches="tight",
                    pad_inches=0.04, dpi=450)
    plt.close(fig)
    return out_path


def main():
    plot_latent_space(OUT_BASE)
    print(OUT_BASE)

    p = plot_affinity_unordered(OUT_BASE)
    print(p)

    p = plot_affinity_sorted(OUT_BASE)
    print(p)

    p = plot_stable_rank(OUT_BASE)
    print(p)


if __name__ == "__main__":
    main()
