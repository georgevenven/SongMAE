import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from individual_id.latent_trajectory_decoder.train_rope_sequence import load_rows, parameter_count, split_rows


class T2VecTokenGRU(nn.Module):
    def __init__(self, vocab_size, hidden_dim, layers, bidirectional):
        super().__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.directions = 2 if bidirectional else 1
        self.start_id = vocab_size
        self.embed = nn.Embedding(vocab_size + 1, hidden_dim)
        self.encoder = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.to_decoder = nn.Linear(hidden_dim * self.directions, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def encode(self, tokens, lengths):
        x = self.embed(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.encoder(packed)
        h = h.view(self.layers, self.directions, tokens.shape[0], self.hidden_dim)
        h = h[-1].transpose(0, 1).reshape(tokens.shape[0], self.hidden_dim * self.directions)
        return self.to_decoder(h)

    def forward(self, enc_tokens, enc_lengths, clean_tokens):
        state = self.encode(enc_tokens, enc_lengths)
        h0 = state[None].repeat(self.layers, 1, 1).contiguous()
        start = torch.full((clean_tokens.shape[0], 1), self.start_id, device=clean_tokens.device, dtype=torch.long)
        dec_tokens = torch.cat([start, clean_tokens[:, :-1]], dim=1)
        h, _ = self.decoder(self.embed(dec_tokens), h0)
        h = self.norm(h)
        return self.head(h), h, state


def fit_codebook(rows, vocab_size, seed):
    x = np.vstack([row[2] for row in rows]).astype(np.float32, copy=False)
    kmeans = MiniBatchKMeans(
        n_clusters=vocab_size,
        batch_size=4096,
        n_init=10,
        random_state=seed,
    )
    kmeans.fit(x)
    return kmeans


def tokenize_rows(rows, kmeans):
    return [(recording, bird, kmeans.predict(seq).astype(np.int64)) for recording, bird, seq in rows]


def corrupt_tokens(tokens, keep_prob, rng):
    keep = rng.random(tokens.shape[0]) < keep_prob
    keep[0] = True
    keep[-1] = True
    return tokens[keep]


def make_collate(keep_prob, seed):
    rng = np.random.default_rng(seed)

    def collate(rows):
        clean_lengths = torch.tensor([row[2].shape[0] for row in rows], dtype=torch.long)
        clean = torch.zeros((len(rows), int(clean_lengths.max().item())), dtype=torch.long)
        mask = torch.zeros_like(clean, dtype=torch.bool)
        corrupt = []
        birds = []
        recordings = []
        for i, (recording, bird, tokens) in enumerate(rows):
            clean[i, : tokens.shape[0]] = torch.from_numpy(tokens)
            mask[i, : tokens.shape[0]] = True
            corrupt.append(corrupt_tokens(tokens, keep_prob, rng))
            birds.append(bird)
            recordings.append(recording)

        corrupt_lengths = torch.tensor([tokens.shape[0] for tokens in corrupt], dtype=torch.long)
        enc = torch.zeros((len(rows), int(corrupt_lengths.max().item())), dtype=torch.long)
        for i, tokens in enumerate(corrupt):
            enc[i, : tokens.shape[0]] = torch.from_numpy(tokens)
        return enc, corrupt_lengths, clean, mask, birds, recordings

    return collate


def loss_and_stats(model, batch, device):
    enc, enc_lengths, clean, mask, _, _ = batch
    enc = enc.to(device)
    enc_lengths = enc_lengths.to(device)
    clean = clean.to(device)
    mask = mask.to(device)
    logits, _, _ = model(enc, enc_lengths, clean)
    loss = F.cross_entropy(logits[mask], clean[mask])
    pred = logits.argmax(dim=-1)
    acc = (pred[mask] == clean[mask]).float().mean()
    return loss, acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    for batch in loader:
        loss, acc = loss_and_stats(model, batch, device)
        n = int(batch[3].sum().item())
        total_loss += float(loss.item()) * n
        total_acc += float(acc.item()) * n
        count += n
    return {"loss": total_loss / count, "accuracy": total_acc / count}


def train(model, train_loader, val_loader, args, device, out_dir):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        count = 0
        for batch in train_loader:
            loss, acc = loss_and_stats(model, batch, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = int(batch[3].sum().item())
            total_loss += float(loss.item()) * n
            total_acc += float(acc.item()) * n
            count += n
        val = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / count,
            "train_accuracy": total_acc / count,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
        }
        history.append(row)
        if best is None or row["val_accuracy"] > best["val_accuracy"]:
            best = row
            torch.save({"model": model.state_dict(), "args": vars(args), "best_epoch": best}, out_dir / "model_best.pt")
        print(json.dumps(row), flush=True)
    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train a t2vec-style GRU denoising autoencoder over KMeans latent tokens.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--keep_prob", type=float, default=0.65)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, input_dim = load_rows(args.features_npz)
    train_rows_raw, val_rows_raw = split_rows(rows, args.val_fraction, args.seed)
    kmeans = fit_codebook(train_rows_raw, args.vocab_size, args.seed)
    np.savez_compressed(
        out_dir / "kmeans_codebook.npz",
        centers=kmeans.cluster_centers_.astype(np.float32),
        input_dim=np.asarray(input_dim, dtype=np.int64),
    )
    train_rows = tokenize_rows(train_rows_raw, kmeans)
    val_rows = tokenize_rows(val_rows_raw, kmeans)

    train_loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=make_collate(args.keep_prob, args.seed))
    val_loader = DataLoader(val_rows, batch_size=args.batch_size, shuffle=False, collate_fn=make_collate(args.keep_prob, args.seed + 1))
    model = T2VecTokenGRU(args.vocab_size, args.hidden_dim, args.layers, args.bidirectional).to(device)
    history = train(model, train_loader, val_loader, args, device, out_dir)
    best_checkpoint = torch.load(out_dir / "model_best.pt", map_location="cpu", weights_only=False)

    metrics = {
        "features_npz": args.features_npz,
        "device": str(device),
        "model_type": "t2vec_token_gru",
        "input_dim": int(input_dim),
        "vocab_size": int(args.vocab_size),
        "hidden_dim": int(args.hidden_dim),
        "layers": int(args.layers),
        "bidirectional": bool(args.bidirectional),
        "keep_prob": float(args.keep_prob),
        "trainable_parameters": parameter_count(model),
        "recordings": len(rows),
        "train_recordings": len(train_rows),
        "val_recordings": len(val_rows),
        "history": history,
        "best_epoch": best_checkpoint["best_epoch"],
        "final_train_loss": history[-1]["train_loss"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_val_loss": history[-1]["val_loss"],
        "final_val_accuracy": history[-1]["val_accuracy"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "args": vars(args), "metrics": metrics}, out_dir / "model.pt")
    best_checkpoint["metrics"] = metrics
    torch.save(best_checkpoint, out_dir / "model_best.pt")
    print(json.dumps({k: metrics[k] for k in ["trainable_parameters", "final_val_loss", "final_val_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
