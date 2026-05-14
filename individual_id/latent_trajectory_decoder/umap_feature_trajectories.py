import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def load_rows(path):
    data = np.load(path, allow_pickle=True)
    assert {"features", "bird_labels", "recording_labels"} <= set(data.files)
    features = data["features"].astype(np.float32)
    birds = data["bird_labels"].astype(str)
    recordings = data["recording_labels"].astype(str)

    by_recording = defaultdict(list)
    for i, recording in enumerate(recordings):
        by_recording[recording].append(i)

    rows = []
    for recording, indices in by_recording.items():
        labels = np.unique(birds[indices])
        assert len(labels) == 1, recording
        rows.append((recording, labels[0], features[indices]))
    return rows, features.shape[1]


def select_rows(rows, individuals, songs_per_individual, seed):
    rng = np.random.default_rng(seed)
    by_bird = defaultdict(list)
    for row in rows:
        by_bird[row[1]].append(row)
    eligible = [bird for bird, bird_rows in by_bird.items() if len(bird_rows) >= songs_per_individual]
    assert len(eligible) >= individuals, (len(eligible), individuals)
    birds = sorted(rng.choice(sorted(eligible), size=individuals, replace=False).tolist())

    selected = []
    for bird in birds:
        bird_rows = sorted(by_bird[bird], key=lambda row: row[0])
        keep = rng.choice(len(bird_rows), size=songs_per_individual, replace=False)
        selected.extend([bird_rows[i] for i in sorted(keep)])
    return selected, birds


def fit_umap(x, neighbors, min_dist, metric, seed):
    import umap

    reducer = umap.UMAP(
        n_neighbors=min(neighbors, max(2, x.shape[0] - 1)),
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(x).astype(np.float32)


def plot_trajectories(xy, slices, birds, out_path, title):
    unique_birds = sorted(set(birds))
    colors = {bird: plt.get_cmap("turbo", len(unique_birds))(i) for i, bird in enumerate(unique_birds)}
    fig, ax = plt.subplots(figsize=(12, 10))
    for (start, end), bird in zip(slices, birds):
        pts = xy[start:end]
        ax.plot(pts[:, 0], pts[:, 1], color=colors[bird], alpha=0.22, linewidth=0.8)
        ax.scatter(pts[0, 0], pts[0, 1], color=colors[bird], s=8, alpha=0.45, linewidths=0)
    for bird in unique_birds:
        ax.scatter([], [], color=colors[bird], label=bird, s=20)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_time_panels(xy, slices, birds, out_path, title):
    unique_birds = sorted(set(birds))
    cols = min(5, len(unique_birds))
    rows = int(np.ceil(len(unique_birds) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    margin_x = (xy[:, 0].max() - xy[:, 0].min()) * 0.05
    margin_y = (xy[:, 1].max() - xy[:, 1].min()) * 0.05
    limits = (
        xy[:, 0].min() - margin_x,
        xy[:, 0].max() + margin_x,
        xy[:, 1].min() - margin_y,
        xy[:, 1].max() + margin_y,
    )
    cmap = plt.get_cmap("viridis")
    for ax, bird in zip(axes.ravel(), unique_birds):
        for (start, end), row_bird in zip(slices, birds):
            if row_bird != bird:
                continue
            pts = xy[start:end]
            if pts.shape[0] > 1:
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                t = np.linspace(0.0, 1.0, segs.shape[0])
                lines = LineCollection(segs, colors=cmap(t), linewidths=0.8, alpha=0.28)
                ax.add_collection(lines)
            t_pts = np.linspace(0.0, 1.0, pts.shape[0])
            ax.scatter(pts[:, 0], pts[:, 1], c=t_pts, cmap=cmap, s=4, alpha=0.38, linewidths=0)
        ax.set_title(str(bird), fontsize=10)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(unique_birds) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="UMAP raw latent token vectors as connected song trajectories.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--individuals", type=int, default=30)
    parser.add_argument("--songs_per_individual", type=int, default=10)
    parser.add_argument("--umap_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.05)
    parser.add_argument("--umap_metric", default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, input_dim = load_rows(args.features_npz)
    selected, selected_birds = select_rows(rows, args.individuals, args.songs_per_individual, args.seed)

    features = []
    slices = []
    birds = []
    recordings = []
    offset = 0
    for recording, bird, x in selected:
        features.append(x)
        slices.append((offset, offset + x.shape[0]))
        birds.append(bird)
        recordings.append(recording)
        offset += x.shape[0]

    features = np.vstack(features).astype(np.float32)
    xy = fit_umap(features, args.umap_neighbors, args.umap_min_dist, args.umap_metric, args.seed)
    png_path = out_dir / "feature_token_trajectories_umap.png"
    title = args.title or f"Encoded latent trajectories: {args.individuals} individuals x {args.songs_per_individual} songs"
    plot_trajectories(
        xy,
        slices,
        birds,
        png_path,
        title,
    )
    panel_path = out_dir / "feature_token_trajectories_umap_panels_timecolor.png"
    plot_time_panels(xy, slices, birds, panel_path, title)
    np.savez_compressed(
        out_dir / "feature_token_trajectories_umap.npz",
        xy=xy,
        slices=np.asarray(slices, dtype=np.int64),
        birds=np.asarray(birds, dtype=object),
        recordings=np.asarray(recordings, dtype=object),
        selected_birds=np.asarray(selected_birds, dtype=object),
    )
    summary = {
        "features_npz": args.features_npz,
        "individuals": args.individuals,
        "songs_per_individual": args.songs_per_individual,
        "selected_birds": selected_birds,
        "songs": len(selected),
        "token_points": int(xy.shape[0]),
        "input_dim": int(input_dim),
        "trajectory_png": str(png_path),
        "panel_timecolor_png": str(panel_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
