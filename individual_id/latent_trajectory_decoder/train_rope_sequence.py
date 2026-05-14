import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def load_rows(path):
    data = np.load(path, allow_pickle=True)
    assert {"features", "bird_labels", "recording_labels"} <= set(data.files)
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


def split_rows(rows, val_fraction, seed):
    rng = np.random.default_rng(seed)
    by_bird = defaultdict(list)
    for row in rows:
        by_bird[row[1]].append(row)

    train = []
    val = []
    for bird_rows in by_bird.values():
        shuffled = list(bird_rows)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        n_val = min(n_val, len(shuffled) - 1)
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val


def collate(rows):
    lengths = torch.tensor([row[2].shape[0] - 1 for row in rows], dtype=torch.long)
    max_len = int(lengths.max().item())
    input_dim = rows[0][2].shape[1]
    x = torch.zeros((len(rows), max_len, input_dim), dtype=torch.float32)
    y = torch.zeros((len(rows), max_len, input_dim), dtype=torch.float32)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    birds = []
    recordings = []
    for i, (recording, bird, seq) in enumerate(rows):
        n = seq.shape[0] - 1
        x[i, :n] = torch.from_numpy(seq[:-1])
        y[i, :n] = torch.from_numpy(seq[1:])
        mask[i, :n] = True
        birds.append(bird)
        recordings.append(recording)
    return x, y, mask, birds, recordings


def rotate_half(x):
    a = x[..., 0::2]
    b = x[..., 1::2]
    return torch.stack((-b, a), dim=-1).flatten(-2)


def apply_rope(x):
    _, _, _, dim = x.shape
    assert dim % 2 == 0
    positions = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    angles = positions[:, None] * freqs[None, :]
    cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)[None, :, None, :]
    sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)[None, :, None, :]
    return x * cos + rotate_half(x) * sin


def sinusoidal_positions(length, dim, device):
    assert dim % 2 == 0
    positions = torch.arange(length, device=device, dtype=torch.float32)
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    angles = positions[:, None] * freqs[None, :]
    pe = torch.empty((length, dim), device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


class RopeBlock(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.head_dim = d_model // heads
        assert self.head_dim % 2 == 0
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, valid, return_parts=False):
        b, t, d = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.head_dim)
        k = k.view(b, t, self.heads, self.head_dim)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q).transpose(1, 2)
        k = apply_rope(k).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / np.sqrt(self.head_dim)
        causal = torch.triu(torch.ones((t, t), device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal[None, None], -torch.inf)
        scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)
        y = (attn @ v).transpose(1, 2).reshape(b, t, d)
        attn_out = self.out(y)
        x = x + attn_out
        ffn_out = self.ff(self.norm2(x))
        x = x + ffn_out
        if return_parts:
            return x, attn_out, ffn_out
        return x


class SinusoidalBlock(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.head_dim = d_model // heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, valid, return_parts=False):
        b, t, d = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / np.sqrt(self.head_dim)
        causal = torch.triu(torch.ones((t, t), device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal[None, None], -torch.inf)
        scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)
        y = (attn @ v).transpose(1, 2).reshape(b, t, d)
        attn_out = self.out(y)
        x = x + attn_out
        ffn_out = self.ff(self.norm2(x))
        x = x + ffn_out
        if return_parts:
            return x, attn_out, ffn_out
        return x


class RopeSequenceDecoder(nn.Module):
    def __init__(self, input_dim, d_model, heads, layers):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([RopeBlock(d_model, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, input_dim)

    def forward(self, x, valid):
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, valid)
        h = self.norm(h)
        return self.head(h), h

    def forward_layers(self, x, valid):
        h = self.in_proj(x)
        parts = {"attn": [], "ffn": [], "hidden": []}
        for block in self.blocks:
            h, attn_out, ffn_out = block(h, valid, return_parts=True)
            parts["attn"].append(attn_out)
            parts["ffn"].append(ffn_out)
            parts["hidden"].append(h)
        h = self.norm(h)
        return self.head(h), h, parts


class SinusoidalSequenceDecoder(nn.Module):
    def __init__(self, input_dim, d_model, heads, layers):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([SinusoidalBlock(d_model, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, input_dim)

    def add_positions(self, h):
        return h + sinusoidal_positions(h.shape[1], h.shape[2], h.device)[None]

    def forward(self, x, valid):
        h = self.add_positions(self.in_proj(x))
        for block in self.blocks:
            h = block(h, valid)
        h = self.norm(h)
        return self.head(h), h

    def forward_layers(self, x, valid):
        h = self.add_positions(self.in_proj(x))
        parts = {"attn": [], "ffn": [], "hidden": []}
        for block in self.blocks:
            h, attn_out, ffn_out = block(h, valid, return_parts=True)
            parts["attn"].append(attn_out)
            parts["ffn"].append(ffn_out)
            parts["hidden"].append(h)
        h = self.norm(h)
        return self.head(h), h, parts


def loss_for_batch(model, batch, device):
    x, y, valid, _, _ = batch
    x = x.to(device)
    y = y.to(device)
    valid = valid.to(device)
    pred, _ = model(x, valid)
    diff = (pred - y).square().mean(dim=-1)
    return diff[valid].mean()


def train(model, train_loader, val_loader, epochs, lr, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            loss = loss_for_batch(model, batch, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            n = int(batch[2].sum().item())
            total += loss.item() * n
            count += n
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_mse": total / count, "val_mse": val}
        history.append(row)
        print(json.dumps(row), flush=True)
    return history


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        loss = loss_for_batch(model, batch, device)
        n = int(batch[2].sum().item())
        total += loss.item() * n
        count += n
    return total / count


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a RoPE causal decoder on full variable-length latent sequences.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    rows, input_dim = load_rows(args.features_npz)
    train_rows, val_rows = split_rows(rows, args.val_fraction, args.seed)

    train_loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_rows, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = RopeSequenceDecoder(input_dim, args.d_model, args.heads, args.layers).to(device)
    history = train(model, train_loader, val_loader, args.epochs, args.lr, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "features_npz": args.features_npz,
        "device": str(device),
        "input_dim": input_dim,
        "d_model": args.d_model,
        "heads": args.heads,
        "layers": args.layers,
        "trainable_parameters": parameter_count(model),
        "recordings": len(rows),
        "train_recordings": len(train_rows),
        "val_recordings": len(val_rows),
        "train_tokens": int(sum(row[2].shape[0] - 1 for row in train_rows)),
        "val_tokens": int(sum(row[2].shape[0] - 1 for row in val_rows)),
        "decoder_objective": "next_latent_mse_full_variable_sequences_rope",
        "feature_source": "patch_embeddings_before_pos_encoding_whiten_l2",
        "history": history,
        "final_train_mse": history[-1]["train_mse"],
        "final_val_mse": history[-1]["val_mse"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"decoder": model.state_dict(), "args": vars(args), "metrics": metrics}, out_dir / "model.pt")
    print(json.dumps({k: metrics[k] for k in ["trainable_parameters", "train_tokens", "val_tokens", "final_val_mse"]}, indent=2))


if __name__ == "__main__":
    main()
