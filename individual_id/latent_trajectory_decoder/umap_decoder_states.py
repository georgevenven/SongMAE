import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from individual_id.latent_trajectory_decoder.train_probe import TinyTrajectoryGPT


def load_feature_rows(path, seq_len):
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
        x = features[indices]
        if x.shape[0] > seq_len:
            rows.append((recording, labels[0], x))
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


def make_windows(x, seq_len, stride, max_windows_per_recording, seed):
    starts = np.arange(0, x.shape[0] - seq_len + 1, stride, dtype=np.int64)
    assert starts.size > 0
    if max_windows_per_recording and starts.size > max_windows_per_recording:
        rng = np.random.default_rng(seed)
        starts = np.sort(rng.choice(starts, size=max_windows_per_recording, replace=False))
    return np.stack([x[start : start + seq_len] for start in starts]).astype(np.float32)


@torch.no_grad()
def decoder_states(model, windows, batch_size, device):
    model.eval()
    states = []
    for start in range(0, windows.shape[0], batch_size):
        x = torch.from_numpy(windows[start : start + batch_size]).to(device)
        _, h = model(x)
        states.append(h.mean(dim=1).cpu().numpy())
    return np.vstack(states).astype(np.float32)


def fit_umap(x, neighbors, min_dist, metric, seed):
    import umap

    reducer = umap.UMAP(
        n_neighbors=min(neighbors, max(2, x.shape[0] - 1)),
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(x).astype(np.float32)


def plot_umap(xy, labels, title, path):
    birds = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("turbo", len(birds))
    color_by_bird = {bird: cmap(i) for i, bird in enumerate(birds)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for bird in birds:
        mask = labels == bird
        ax.scatter(xy[mask, 0], xy[mask, 1], s=18, color=color_by_bird[bird], label=bird, alpha=0.82, linewidths=0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, markerscale=1.5, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_model(path, input_dim, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    model = TinyTrajectoryGPT(
        input_dim,
        int(args["seq_len"]),
        int(args["d_model"]),
        int(args["heads"]),
        int(args["layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["decoder"])
    model.eval()
    mean = np.asarray(checkpoint["coord_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["coord_std"], dtype=np.float32)
    return model, mean, std, args


def score(xy, labels):
    if len(set(labels.tolist())) < 2:
        return None
    return float(silhouette_score(xy, labels, metric="euclidean"))


def parse_args():
    parser = argparse.ArgumentParser(description="UMAP decoder hidden states for a balanced recording subset.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--model_pt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--individuals", type=int, default=30)
    parser.add_argument("--songs_per_individual", type=int, default=10)
    parser.add_argument("--window_stride", type=int, default=32)
    parser.add_argument("--max_windows_per_recording", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--umap_neighbors", type=int, default=20)
    parser.add_argument("--umap_min_dist", type=float, default=0.05)
    parser.add_argument("--umap_metric", default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model_pt, map_location="cpu", weights_only=False)
    seq_len = int(checkpoint["args"]["seq_len"])
    rows, input_dim = load_feature_rows(args.features_npz, seq_len)
    model, mean, std, model_args = load_model(args.model_pt, input_dim, device)
    selected, selected_birds = select_rows(rows, args.individuals, args.songs_per_individual, args.seed)

    window_states = []
    window_birds = []
    window_recordings = []
    recording_states = []
    recording_birds = []
    recording_names = []
    for index, (recording, bird, x) in enumerate(selected):
        x = (x - mean) / std
        windows = make_windows(x, seq_len, args.window_stride, args.max_windows_per_recording, args.seed + index)
        states = decoder_states(model, windows, args.batch_size, device)
        window_states.append(states)
        window_birds.extend([bird] * states.shape[0])
        window_recordings.extend([recording] * states.shape[0])
        recording_states.append(states.mean(axis=0))
        recording_birds.append(bird)
        recording_names.append(recording)

    window_states = np.vstack(window_states)
    recording_states = np.vstack(recording_states)
    window_birds = np.asarray(window_birds, dtype=object)
    recording_birds = np.asarray(recording_birds, dtype=object)
    window_recordings = np.asarray(window_recordings, dtype=object)
    recording_names = np.asarray(recording_names, dtype=object)

    recording_xy = fit_umap(recording_states, args.umap_neighbors, args.umap_min_dist, args.umap_metric, args.seed)
    window_xy = fit_umap(window_states, args.umap_neighbors, args.umap_min_dist, args.umap_metric, args.seed)

    recording_png = out_dir / "decoder_recording_state_umap.png"
    window_png = out_dir / "decoder_window_state_umap.png"
    plot_umap(recording_xy, recording_birds, "Decoder state UMAP: 30 zf individuals x 10 recordings", recording_png)
    plot_umap(window_xy, window_birds, "Decoder window-state UMAP: 30 zf individuals x 10 recordings", window_png)

    np.savez_compressed(
        out_dir / "decoder_state_umaps.npz",
        recording_xy=recording_xy,
        recording_birds=recording_birds,
        recording_names=recording_names,
        window_xy=window_xy,
        window_birds=window_birds,
        window_recordings=window_recordings,
        selected_birds=np.asarray(selected_birds, dtype=object),
    )
    summary = {
        "features_npz": args.features_npz,
        "model_pt": args.model_pt,
        "device": str(device),
        "individuals": args.individuals,
        "songs_per_individual": args.songs_per_individual,
        "selected_birds": selected_birds,
        "recording_points": int(recording_xy.shape[0]),
        "window_points": int(window_xy.shape[0]),
        "decoder_state_dim": int(recording_states.shape[1]),
        "input_dim": int(input_dim),
        "seq_len": int(seq_len),
        "model_d_model": int(model_args["d_model"]),
        "recording_silhouette": score(recording_xy, recording_birds),
        "window_silhouette": score(window_xy, window_birds),
        "recording_png": str(recording_png),
        "window_png": str(window_png),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
