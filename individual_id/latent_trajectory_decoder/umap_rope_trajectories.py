import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from individual_id.latent_trajectory_decoder.train_rope_sequence import RopeSequenceDecoder


def load_rows(path):
    data = np.load(path, allow_pickle=True)
    x = data["features"].astype(np.float32)
    birds = data["bird_labels"].astype(str)
    recordings = data["recording_labels"].astype(str)
    by_recording = defaultdict(list)
    for i, recording in enumerate(recordings):
        by_recording[recording].append(i)

    rows = []
    for recording, indices in by_recording.items():
        labels = np.unique(birds[indices])
        assert len(labels) == 1, recording
        if len(indices) > 2:
            rows.append((recording, labels[0], x[indices]))
    return rows, x.shape[1]


def select_rows(rows, individuals, songs_per_individual, seed):
    rng = np.random.default_rng(seed)
    by_bird = defaultdict(list)
    for row in rows:
        by_bird[row[1]].append(row)
    eligible = [bird for bird, bird_rows in by_bird.items() if len(bird_rows) >= songs_per_individual]
    assert len(eligible) >= individuals
    birds = sorted(rng.choice(sorted(eligible), size=individuals, replace=False).tolist())
    selected = []
    for bird in birds:
        bird_rows = sorted(by_bird[bird], key=lambda row: row[0])
        keep = rng.choice(len(bird_rows), size=songs_per_individual, replace=False)
        selected.extend([bird_rows[i] for i in sorted(keep)])
    return selected, birds


def load_model(path, input_dim, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    model = RopeSequenceDecoder(input_dim, int(args["d_model"]), int(args["heads"]), int(args["layers"])).to(device)
    model.load_state_dict(checkpoint["decoder"])
    model.eval()
    return model, args


@torch.no_grad()
def states_for_sequence(model, seq, device, target_mode):
    x = seq if target_mode == "self" else seq[:-1]
    x = torch.from_numpy(x).to(device=device, dtype=torch.float32)[None]
    valid = torch.ones((1, x.shape[1]), device=device, dtype=torch.bool)
    _, h = model(x, valid)
    return h.squeeze(0).cpu().numpy().astype(np.float32)


def fit_umap(x, neighbors, min_dist, metric, seed):
    import umap

    reducer = umap.UMAP(
        n_neighbors=min(neighbors, max(2, x.shape[0] - 1)),
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(x).astype(np.float32)


def plot_trajectories(xy, slices, birds, recordings, out_path, title):
    unique_birds = sorted(set(birds))
    colors = {bird: plt.get_cmap("turbo", len(unique_birds))(i) for i, bird in enumerate(unique_birds)}
    fig, ax = plt.subplots(figsize=(12, 10))
    for (start, end), bird, _ in zip(slices, birds, recordings):
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


def parse_args():
    parser = argparse.ArgumentParser(description="UMAP full per-token RoPE decoder states as connected song trajectories.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--model_pt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--individuals", type=int, default=30)
    parser.add_argument("--songs_per_individual", type=int, default=10)
    parser.add_argument("--title", default=None)
    parser.add_argument("--umap_neighbors", type=int, default=30)
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
    rows, input_dim = load_rows(args.features_npz)
    selected, selected_birds = select_rows(rows, args.individuals, args.songs_per_individual, args.seed)
    model, model_args = load_model(args.model_pt, input_dim, device)
    target_mode = model_args.get("target_mode", "next")

    states = []
    slices = []
    birds = []
    recordings = []
    offset = 0
    for recording, bird, seq in selected:
        h = states_for_sequence(model, seq, device, target_mode)
        states.append(h)
        slices.append((offset, offset + h.shape[0]))
        birds.append(bird)
        recordings.append(recording)
        offset += h.shape[0]

    states = np.vstack(states)
    xy = fit_umap(states, args.umap_neighbors, args.umap_min_dist, args.umap_metric, args.seed)
    png_path = out_dir / "rope_decoder_token_trajectories_umap.png"
    title = args.title or f"RoPE decoder hidden-state trajectories: {args.individuals} individuals x {args.songs_per_individual} songs"
    plot_trajectories(xy, slices, birds, recordings, png_path, title)
    np.savez_compressed(
        out_dir / "rope_decoder_token_trajectories_umap.npz",
        xy=xy,
        slices=np.asarray(slices, dtype=np.int64),
        birds=np.asarray(birds, dtype=object),
        recordings=np.asarray(recordings, dtype=object),
        selected_birds=np.asarray(selected_birds, dtype=object),
    )
    summary = {
        "features_npz": args.features_npz,
        "model_pt": args.model_pt,
        "device": str(device),
        "individuals": args.individuals,
        "songs_per_individual": args.songs_per_individual,
        "selected_birds": selected_birds,
        "songs": len(selected),
        "token_points": int(xy.shape[0]),
        "decoder_state_dim": int(states.shape[1]),
        "input_dim": int(input_dim),
        "model_d_model": int(model_args["d_model"]),
        "target_mode": target_mode,
        "trajectory_png": str(png_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
