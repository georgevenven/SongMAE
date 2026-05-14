import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image, ImageDraw


class TinyTrajectoryGPT(nn.Module):
    def __init__(self, input_dim, seq_len, d_model, heads, layers):
        super().__init__()
        self.coord = nn.Linear(input_dim, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Linear(d_model, input_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device)
        h = self.coord(x) + self.pos(positions)[None, :, :]
        h = self.blocks(h, mask=self.mask[: x.shape[1], : x.shape[1]])
        return self.head(h), h


def load_recordings(path):
    data = np.load(path, allow_pickle=True)
    if "features" in data.files:
        x = data["features"].astype(np.float32)
        source_space = "latent"
    else:
        assert "xy" in data.files
        x = data["xy"].astype(np.float32)
        source_space = "umap"
    assert {"bird_labels", "recording_labels"} <= set(data.files)
    birds = data["bird_labels"].astype(str)
    recordings = data["recording_labels"].astype(str)

    by_recording = defaultdict(list)
    for i, recording in enumerate(recordings):
        by_recording[recording].append(i)

    rows = []
    for recording, indices in by_recording.items():
        labels = np.unique(birds[indices])
        assert len(labels) == 1, recording
        rows.append((recording, labels[0], x[indices]))
    return rows, source_space, x.shape[1]


def split_recordings(rows, val_fraction, seed):
    rng = np.random.default_rng(seed)
    by_bird = defaultdict(list)
    for row in rows:
        by_bird[row[1]].append(row)

    train = []
    val = []
    for bird_rows in by_bird.values():
        shuffled = list(bird_rows)
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * val_fraction))
        if len(shuffled) > 1:
            n_val = max(1, min(n_val, len(shuffled) - 1))
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val


def make_windows(rows, seq_len, stride, max_windows, seed):
    xs = []
    ys = []
    birds = []
    recordings = []
    for recording, bird, xy in rows:
        for start in range(0, xy.shape[0] - seq_len, stride):
            chunk = xy[start : start + seq_len + 1]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            birds.append(bird)
            recordings.append(recording)

    assert xs, "no coordinate windows; lower --seq_len or --stride"
    if max_windows and len(xs) > max_windows:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(xs), size=max_windows, replace=False)
        xs = [xs[i] for i in keep]
        ys = [ys[i] for i in keep]
        birds = [birds[i] for i in keep]
        recordings = [recordings[i] for i in keep]

    return (
        np.stack(xs).astype(np.float32),
        np.stack(ys).astype(np.float32),
        np.array(birds),
        np.array(recordings),
    )


def standardize(train_x, train_y, val_x, val_y):
    dim = train_x.shape[-1]
    points = np.concatenate([train_x.reshape(-1, dim), train_y.reshape(-1, dim)], axis=0)
    mean = points.mean(axis=0)
    std = points.std(axis=0)
    assert np.all(std > 0), std
    return (train_x - mean) / std, (train_y - mean) / std, (val_x - mean) / std, (val_y - mean) / std, mean, std


def loader(x, y, labels, batch_size, shuffle):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_decoder(model, train_loader, val_loader, epochs, lr, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.MSELoss()
    history = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred, _ = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.shape[0]
            train_count += x.shape[0]

        val_loss = evaluate_decoder(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_mse": train_loss / train_count, "val_mse": val_loss})
    return history


@torch.no_grad()
def evaluate_decoder(model, batches, device):
    model.eval()
    loss_fn = nn.MSELoss(reduction="sum")
    total = 0.0
    count = 0
    for x, y, _ in batches:
        x = x.to(device)
        y = y.to(device)
        pred, _ = model(x)
        total += loss_fn(pred, y).item()
        count += int(np.prod(y.shape))
    return total / count


@torch.no_grad()
def states(model, batches, device):
    model.eval()
    xs = []
    ys = []
    for x, _, label in batches:
        _, h = model(x.to(device))
        xs.append(h.mean(dim=1).cpu())
        ys.append(label)
    return torch.cat(xs), torch.cat(ys)


def train_probe(train_x, train_y, val_x, val_y, classes, epochs, lr, device):
    probe = nn.Linear(train_x.shape[1], classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    for _ in range(epochs):
        logits = probe(train_x)
        loss = loss_fn(logits, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_acc = (probe(train_x).argmax(dim=1) == train_y).float().mean().item()
        val_acc = (probe(val_x).argmax(dim=1) == val_y).float().mean().item()
    return probe, train_acc, val_acc


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def encode_labels(train_birds, val_birds):
    labels = sorted(set(train_birds.tolist()))
    index = {label: i for i, label in enumerate(labels)}
    train = np.array([index[label] for label in train_birds], dtype=np.int64)
    val = np.array([index[label] for label in val_birds], dtype=np.int64)
    return train, val, labels


@torch.no_grad()
def rollout(model, prompt, total_len, device):
    model.eval()
    points = torch.from_numpy(prompt).to(device=device, dtype=torch.float32)
    while points.shape[0] < total_len:
        _, hidden = model(points[None, :, :])
        pred = model.head(hidden[:, -1]).squeeze(0)
        points = torch.cat([points, pred[None]], dim=0)
    return points.cpu().numpy()


def to_pixels(points, lo, hi, size, pad):
    span = np.maximum(hi - lo, 1e-6)
    xy = (points - lo) / span
    x = pad + xy[:, 0] * (size - 2 * pad)
    y = size - pad - xy[:, 1] * (size - 2 * pad)
    return list(map(tuple, np.stack([x, y], axis=1)))


def pca_2d(points):
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def save_prediction_gif(model, val_x, val_y, mean, std, path, prefix_len, device, size=640):
    prefix_len = max(1, min(prefix_len, val_x.shape[1]))
    true = np.concatenate([val_x[0, :1], val_y[0]], axis=0)
    pred = rollout(model, val_x[0, :prefix_len], true.shape[0], device)
    true = true * std + mean
    pred = pred * std + mean
    if true.shape[1] > 2:
        projected = pca_2d(np.vstack([true, pred]))
        true = projected[: true.shape[0]]
        pred = projected[true.shape[0] :]

    lo = np.minimum(true.min(axis=0), pred.min(axis=0))
    hi = np.maximum(true.max(axis=0), pred.max(axis=0))
    pad = 54
    true_px = to_pixels(true, lo, hi, size, pad)
    pred_px = to_pixels(pred, lo, hi, size, pad)
    frames = []
    for end in range(prefix_len + 1, true.shape[0] + 1):
        frame = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(frame)
        draw.rectangle((pad, pad, size - pad, size - pad), outline=(210, 210, 210))
        draw.text((pad, 18), "blue=true trajectory, red=decoder rollout", fill=(20, 20, 20))
        draw.text((pad, size - 34), f"prefix={prefix_len}, step={end - 1}/{true.shape[0] - 1}", fill=(40, 40, 40))
        if end > 1:
            draw.line(true_px[:end], fill=(40, 95, 200), width=4)
            draw.line(pred_px[:end], fill=(210, 60, 55), width=4)
        for point in true_px[:prefix_len]:
            x, y = point
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(20, 60, 170))
        x, y = true_px[end - 1]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(40, 95, 200))
        x, y = pred_px[end - 1]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(210, 60, 55))
        frames.append(frame)

    frames[0].save(path, save_all=True, append_images=frames[1:], duration=160, loop=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tiny causal decoder on UMAP coordinate trajectories, then probe bird ID.")
    parser.add_argument("--umap_points_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max_windows", type=int, default=20000)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--probe_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=144)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--probe_lr", type=float, default=1e-2)
    parser.add_argument("--gif_prefix_len", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    raw_rows, source_space, input_dim = load_recordings(args.umap_points_npz)
    rows = [row for row in raw_rows if row[2].shape[0] > args.seq_len]
    assert rows, "no recordings are long enough for --seq_len"
    train_rows, val_rows = split_recordings(rows, args.val_fraction, args.seed)
    assert train_rows and val_rows

    train_x, train_y, train_birds, train_recordings = make_windows(train_rows, args.seq_len, args.stride, args.max_windows, args.seed)
    val_x, val_y, val_birds, val_recordings = make_windows(val_rows, args.seq_len, args.stride, args.max_windows, args.seed + 1)
    known_val = np.isin(val_birds, np.unique(train_birds))
    assert known_val.any(), "no validation windows match train probe labels"
    val_x = val_x[known_val]
    val_y = val_y[known_val]
    val_birds = val_birds[known_val]
    val_recordings = val_recordings[known_val]
    train_x, train_y, val_x, val_y, mean, std = standardize(train_x, train_y, val_x, val_y)
    train_labels, val_labels, classes = encode_labels(train_birds, val_birds)

    model = TinyTrajectoryGPT(input_dim, args.seq_len, args.d_model, args.heads, args.layers).to(device)
    trainable_parameters = parameter_count(model)
    train_batches = loader(train_x, train_y, train_labels, args.batch_size, True)
    val_batches = loader(val_x, val_y, val_labels, args.batch_size, False)
    history = train_decoder(model, train_batches, val_batches, args.epochs, args.lr, device)

    train_state, train_label = states(model, loader(train_x, train_y, train_labels, args.batch_size, False), device)
    val_state, val_label = states(model, val_batches, device)
    probe, train_acc, val_acc = train_probe(
        train_state,
        train_label,
        val_state,
        val_label,
        len(classes),
        args.probe_epochs,
        args.probe_lr,
        device,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / "prediction_rollout.gif"
    save_prediction_gif(model, val_x, val_y, mean, std, gif_path, args.gif_prefix_len, device)
    metrics = {
        "umap_points_npz": args.umap_points_npz,
        "device": str(device),
        "decoder_objective": "next_coordinate_mse",
        "source_space": source_space,
        "input_dim": input_dim,
        "decoder_uses_bird_labels": False,
        "probe_uses_bird_labels": True,
        "recordings": len(raw_rows),
        "eligible_recordings": len(rows),
        "train_recordings": len(train_rows),
        "val_recordings": len(val_rows),
        "train_windows": int(train_x.shape[0]),
        "val_windows": int(val_x.shape[0]),
        "classes": classes,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "d_model": args.d_model,
        "heads": args.heads,
        "layers": args.layers,
        "trainable_parameters": trainable_parameters,
        "coord_mean_shape": list(mean.shape),
        "coord_std_shape": list(std.shape),
        "coord_mean_first8": mean[:8].tolist(),
        "coord_std_first8": std[:8].tolist(),
        "decoder_history": history,
        "final_train_mse": history[-1]["train_mse"],
        "final_val_mse": history[-1]["val_mse"],
        "probe_train_accuracy": train_acc,
        "probe_val_accuracy": val_acc,
        "prediction_gif": str(gif_path),
        "train_recording_examples": train_recordings[:5].tolist(),
        "val_recording_examples": val_recordings[:5].tolist(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(
        {
            "decoder": model.state_dict(),
            "probe": probe.state_dict(),
            "classes": classes,
            "coord_mean": mean,
            "coord_std": std,
            "args": vars(args),
        },
        out_dir / "model.pt",
    )
    print(
        json.dumps(
            {
                "source_space": metrics["source_space"],
                "input_dim": metrics["input_dim"],
                "trainable_parameters": metrics["trainable_parameters"],
                "train_windows": metrics["train_windows"],
                "val_windows": metrics["val_windows"],
                "final_val_mse": metrics["final_val_mse"],
                "probe_val_accuracy": metrics["probe_val_accuracy"],
                "prediction_gif": metrics["prediction_gif"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
