import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class UmapTrajectoryGRU(nn.Module):
    def __init__(self, hidden_dim, layers):
        super().__init__()
        self.gru = nn.GRU(2, hidden_dim, num_layers=layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x, state=None):
        h, state = self.gru(x, state)
        h = self.norm(h)
        return self.head(h), h, state


def load_trajectories(path):
    data = np.load(path, allow_pickle=True)
    xy = data["xy"].astype(np.float32)
    slices = data["slices"].astype(np.int64)
    birds = data["birds"].astype(str)
    recordings = data["recordings"].astype(str)
    return [(recording, bird, xy[start:end]) for (start, end), bird, recording in zip(slices, birds, recordings)]


def split_rows(rows, val_fraction, seed):
    rng = np.random.default_rng(seed)
    train = []
    val = []
    for bird in sorted({row[1] for row in rows}):
        bird_rows = [row for row in rows if row[1] == bird]
        rng.shuffle(bird_rows)
        n_val = max(1, int(round(len(bird_rows) * val_fraction)))
        val.extend(bird_rows[:n_val])
        train.extend(bird_rows[n_val:])
    return train, val


def make_windows(rows, seq_len, stride):
    xs = []
    ys = []
    bird_labels = []
    for _, bird, xy in rows:
        for start in range(0, xy.shape[0] - seq_len, stride):
            chunk = xy[start : start + seq_len + 1]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            bird_labels.append(bird)
    assert xs
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32), np.asarray(bird_labels, dtype=object)


def standardize(train_x, train_y, val_x, val_y):
    points = np.concatenate([train_x.reshape(-1, 2), train_y.reshape(-1, 2)], axis=0)
    mean = points.mean(axis=0)
    std = points.std(axis=0)
    assert np.all(std > 0)
    return (train_x - mean) / std, (train_y - mean) / std, (val_x - mean) / std, (val_y - mean) / std, mean, std


def loader(x, y, batch_size, shuffle):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def mse(model, batches, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x, y in batches:
            pred, _, _ = model(x.to(device))
            total += (pred - y.to(device)).square().sum().item()
            count += int(np.prod(y.shape))
    return total / count


def train(model, train_loader, val_loader, epochs, lr, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []
    best = None
    best_state = None
    for epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for x, y in train_loader:
            pred, _, _ = model(x.to(device))
            loss = (pred - y.to(device)).square().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * int(np.prod(y.shape))
            count += int(np.prod(y.shape))
        row = {"epoch": epoch + 1, "train_mse": total / count, "val_mse": mse(model, val_loader, device)}
        history.append(row)
        if best is None or row["val_mse"] < best["val_mse"]:
            best = row
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        print(json.dumps(row), flush=True)
    model.load_state_dict(best_state)
    return history, best


@torch.no_grad()
def rollout(model, prefix, total_len, device):
    points = torch.from_numpy(prefix).to(device=device, dtype=torch.float32)
    state = None
    if points.shape[0] > 1:
        _, _, state = model(points[:-1][None], None)
    while points.shape[0] < total_len:
        pred, _, state = model(points[-1:][None], state)
        points = torch.cat([points, pred[0, -1:]], dim=0)
    return points.cpu().numpy()


def plot_predictions(model, rows, mean, std, out_dir, prefix_len, device):
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), dpi=150)
    axes = axes.ravel()
    frames = []
    rows = rows[:10]
    for ax, (recording, bird, xy) in zip(axes, rows):
        true = (xy - mean) / std
        pred = rollout(model, true[:prefix_len], true.shape[0], device) * std + mean
        true = xy
        ax.plot(true[:, 0], true[:, 1], color="#2f6fdd", linewidth=1.3, alpha=0.75)
        ax.plot(pred[:, 0], pred[:, 1], color="#d64b3c", linewidth=1.0, alpha=0.75)
        ax.scatter(true[0, 0], true[0, 1], color="#2f6fdd", s=12)
        ax.set_title(f"{bird} {recording}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(rows):]:
        ax.axis("off")
    fig.suptitle("GRU on 2-D UMAP coordinates: blue=true, red=rollout")
    fig.tight_layout()
    fig.savefig(out_dir / "prediction_rollouts.png")
    plt.close(fig)

    recording, bird, xy = rows[0]
    true = (xy - mean) / std
    pred = rollout(model, true[:prefix_len], true.shape[0], device) * std + mean
    lo = np.minimum(xy.min(axis=0), pred.min(axis=0))
    hi = np.maximum(xy.max(axis=0), pred.max(axis=0))
    for end in range(prefix_len + 1, xy.shape[0] + 1):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        ax.plot(xy[:end, 0], xy[:end, 1], color="#2f6fdd", linewidth=2.0, alpha=0.8)
        ax.plot(pred[:end, 0], pred[:end, 1], color="#d64b3c", linewidth=1.6, alpha=0.8)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{bird}: blue=true, red=GRU rollout")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("P", palette=Image.Palette.ADAPTIVE))
        plt.close(fig)
    imageio.mimsave(out_dir / "prediction_rollout.gif", frames, duration=0.08, loop=0)


@torch.no_grad()
def hidden_summary(model, rows, mean, std, device):
    states = []
    birds = []
    for _, bird, xy in rows:
        x = torch.from_numpy(((xy - mean) / std).astype(np.float32)).to(device)[None]
        _, h, _ = model(x)
        states.append(h.mean(dim=1).squeeze(0).cpu().numpy())
        birds.append(bird)
    return np.vstack(states).astype(np.float32), np.asarray(birds, dtype=object)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GRU next-step model on 2-D UMAP coordinate trajectories.")
    parser.add_argument("--trajectory_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--prefix_len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_trajectories(args.trajectory_npz)
    train_rows, val_rows = split_rows(rows, args.val_fraction, args.seed)
    train_x, train_y, _ = make_windows(train_rows, args.seq_len, args.stride)
    val_x, val_y, _ = make_windows(val_rows, args.seq_len, args.stride)
    train_x, train_y, val_x, val_y, mean, std = standardize(train_x, train_y, val_x, val_y)

    model = UmapTrajectoryGRU(args.hidden_dim, args.layers).to(device)
    history, best = train(
        model,
        loader(train_x, train_y, args.batch_size, True),
        loader(val_x, val_y, args.batch_size, False),
        args.epochs,
        args.lr,
        device,
    )
    plot_predictions(model, val_rows, mean, std, out_dir, args.prefix_len, device)
    hidden, hidden_birds = hidden_summary(model, rows, mean, std, device)
    np.savez_compressed(out_dir / "gru_hidden_recording_states.npz", hidden=hidden, birds=hidden_birds)

    metrics = {
        "trajectory_npz": args.trajectory_npz,
        "device": str(device),
        "recordings": len(rows),
        "train_recordings": len(train_rows),
        "val_recordings": len(val_rows),
        "seq_len": int(args.seq_len),
        "stride": int(args.stride),
        "hidden_dim": int(args.hidden_dim),
        "layers": int(args.layers),
        "train_windows": int(train_x.shape[0]),
        "val_windows": int(val_x.shape[0]),
        "history": history,
        "best_epoch": best,
        "final_train_mse": history[-1]["train_mse"],
        "final_val_mse": history[-1]["val_mse"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "args": vars(args), "metrics": metrics, "mean": mean, "std": std}, out_dir / "model.pt")
    print(json.dumps({key: metrics[key] for key in ["best_epoch", "final_val_mse", "train_windows", "val_windows"]}, indent=2))


if __name__ == "__main__":
    main()
