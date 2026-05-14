import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from individual_id.latent_trajectory_decoder.train_rope_sequence import RopeSequenceDecoder


def load_rows(path):
    data = np.load(path, allow_pickle=True)
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
        if len(indices) > 2:
            rows.append((recording, labels[0], features[indices]))
    return rows, features.shape[1]


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
def states_for_sequence(model, seq, device):
    x = torch.from_numpy(seq[:-1]).to(device=device, dtype=torch.float32)[None]
    valid = torch.ones((1, x.shape[1]), device=device, dtype=torch.bool)
    _, h = model(x, valid)
    return h.squeeze(0).cpu().numpy().astype(np.float32)


def pooled_state(h, mode):
    if mode == "mean":
        return h.mean(axis=0)
    assert mode == "mean_std"
    return np.concatenate([h.mean(axis=0), h.std(axis=0)], axis=0)


def plot_pca(xy, birds, out_path, title):
    unique_birds = sorted(set(birds.tolist()))
    colors = {bird: plt.get_cmap("turbo", len(unique_birds))(i) for i, bird in enumerate(unique_birds)}
    fig, ax = plt.subplots(figsize=(10, 8))
    for bird in unique_birds:
        mask = birds == bird
        ax.scatter(xy[mask, 0], xy[mask, 1], color=colors[bird], s=22, alpha=0.85, linewidths=0, label=bird)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def score(xy, birds):
    return float(silhouette_score(xy, birds, metric="euclidean"))


def parse_args():
    parser = argparse.ArgumentParser(description="Pool RoPE decoder token states per song and PCA the song vectors.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--model_pt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--individuals", type=int, default=30)
    parser.add_argument("--songs_per_individual", type=int, default=10)
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

    pooled = {"mean": [], "mean_std": []}
    birds = []
    recordings = []
    lengths = []
    for recording, bird, seq in selected:
        h = states_for_sequence(model, seq, device)
        pooled["mean"].append(pooled_state(h, "mean"))
        pooled["mean_std"].append(pooled_state(h, "mean_std"))
        birds.append(bird)
        recordings.append(recording)
        lengths.append(int(h.shape[0]))

    birds = np.asarray(birds, dtype=object)
    recordings = np.asarray(recordings, dtype=object)
    outputs = {}
    for mode, rows_for_mode in pooled.items():
        x = np.vstack(rows_for_mode).astype(np.float32)
        pca = PCA(n_components=2, random_state=args.seed)
        xy = pca.fit_transform(x).astype(np.float32)
        png_path = out_dir / f"pooled_decoder_{mode}_pca.png"
        plot_pca(xy, birds, png_path, f"PCA of pooled RoPE decoder states ({mode})")
        outputs[mode] = {
            "xy": xy,
            "explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
            "silhouette": score(xy, birds),
            "png": str(png_path),
            "pooled_dim": int(x.shape[1]),
        }

    np.savez_compressed(
        out_dir / "pooled_decoder_state_pca.npz",
        mean_xy=outputs["mean"]["xy"],
        mean_std_xy=outputs["mean_std"]["xy"],
        birds=birds,
        recordings=recordings,
        lengths=np.asarray(lengths, dtype=np.int64),
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
        "decoder_state_dim": int(model_args["d_model"]),
        "input_dim": int(input_dim),
        "mean": {k: v for k, v in outputs["mean"].items() if k != "xy"},
        "mean_std": {k: v for k, v in outputs["mean_std"].items() if k != "xy"},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
