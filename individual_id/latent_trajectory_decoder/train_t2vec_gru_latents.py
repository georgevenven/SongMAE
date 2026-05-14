import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from individual_id.latent_trajectory_decoder.train_rope_sequence import load_rows, parameter_count, split_rows


class T2VecLatentGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, bidirectional, cell="gru", vae=False, latent_dim=0):
        super().__init__()
        assert cell in {"gru", "rnn", "lstm"}
        rnn_cls = {"gru": nn.GRU, "rnn": nn.RNN, "lstm": nn.LSTM}[cell]
        self.cell = cell
        self.vae = vae
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.directions = 2 if bidirectional else 1
        encoder_dim = hidden_dim * self.directions
        latent_dim = latent_dim or hidden_dim
        self.encoder = rnn_cls(
            input_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        if vae:
            self.to_mu = nn.Linear(encoder_dim, latent_dim)
            self.to_logvar = nn.Linear(encoder_dim, latent_dim)
            self.to_decoder = nn.Linear(latent_dim, hidden_dim)
        else:
            self.to_decoder = nn.Linear(encoder_dim, hidden_dim)
        self.start = nn.Parameter(torch.zeros(input_dim))
        self.decoder = rnn_cls(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.encoder(packed)
        if self.cell == "lstm":
            h = h[0]
        h = h.view(self.layers, self.directions, x.shape[0], self.hidden_dim)
        last = h[-1].transpose(0, 1).reshape(x.shape[0], self.hidden_dim * self.directions)
        if not self.vae:
            state = self.to_decoder(last)
            return state, state, last.new_zeros(())
        mu = self.to_mu(last)
        logvar = self.to_logvar(last).clamp(-10.0, 10.0)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std if self.training else mu
        kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
        return self.to_decoder(z), mu, kl

    def forward(self, enc_x, enc_lengths, clean_x):
        state, bottleneck, kl = self.encode(enc_x, enc_lengths)
        h0 = state[None].repeat(self.layers, 1, 1).contiguous()
        start = self.start[None, None].expand(clean_x.shape[0], 1, clean_x.shape[2])
        dec_in = torch.cat([start, clean_x[:, :-1]], dim=1)
        initial = (h0, h0) if self.cell == "lstm" else h0
        h, _ = self.decoder(dec_in, initial)
        h = self.norm(h)
        return self.head(h), h, bottleneck, kl


def corrupt_sequence(seq, keep_prob, rng):
    keep = rng.random(seq.shape[0]) < keep_prob
    keep[0] = True
    keep[-1] = True
    if keep.sum() < 2:
        keep[rng.integers(0, seq.shape[0])] = True
    return seq[keep]


def make_collate(keep_prob, seed):
    rng = np.random.default_rng(seed)

    def collate(rows):
        clean_lengths = torch.tensor([row[2].shape[0] for row in rows], dtype=torch.long)
        clean_max = int(clean_lengths.max().item())
        input_dim = rows[0][2].shape[1]
        clean = torch.zeros((len(rows), clean_max, input_dim), dtype=torch.float32)
        mask = torch.zeros((len(rows), clean_max), dtype=torch.bool)
        corrupt = []
        birds = []
        recordings = []
        for i, (recording, bird, seq) in enumerate(rows):
            clean[i, : seq.shape[0]] = torch.from_numpy(seq)
            mask[i, : seq.shape[0]] = True
            corrupt.append(corrupt_sequence(seq, keep_prob, rng))
            birds.append(bird)
            recordings.append(recording)

        corrupt_lengths = torch.tensor([seq.shape[0] for seq in corrupt], dtype=torch.long)
        corrupt_max = int(corrupt_lengths.max().item())
        enc = torch.zeros((len(rows), corrupt_max, input_dim), dtype=torch.float32)
        for i, seq in enumerate(corrupt):
            enc[i, : seq.shape[0]] = torch.from_numpy(seq)
        return enc, corrupt_lengths, clean, mask, birds, recordings

    return collate


def loss_and_stats(model, batch, device, kl_beta):
    enc, enc_lengths, clean, mask, _, _ = batch
    enc = enc.to(device)
    enc_lengths = enc_lengths.to(device)
    clean = clean.to(device)
    mask = mask.to(device)
    pred, _, _, kl = model(enc, enc_lengths, clean)
    pred_unit = F.normalize(pred, dim=-1)
    clean_unit = F.normalize(clean, dim=-1)
    cosine = (pred_unit * clean_unit).sum(dim=-1)
    recon_loss = (1.0 - cosine)[mask].mean()
    loss = recon_loss + kl_beta * kl
    mse = (pred_unit - clean_unit).square().mean(dim=-1)
    return loss, mse[mask].mean(), cosine[mask].mean(), pred.norm(dim=-1)[mask].mean(), recon_loss, kl


@torch.no_grad()
def evaluate(model, loader, device, kl_beta):
    model.eval()
    totals = {"loss": 0.0, "mse": 0.0, "cos": 0.0, "pred_norm": 0.0, "recon_loss": 0.0, "kl": 0.0}
    count = 0
    for batch in loader:
        loss, mse, cos, pred_norm, recon_loss, kl = loss_and_stats(model, batch, device, kl_beta)
        n = int(batch[3].sum().item())
        totals["loss"] += float(loss.item()) * n
        totals["mse"] += float(mse.item()) * n
        totals["cos"] += float(cos.item()) * n
        totals["pred_norm"] += float(pred_norm.item()) * n
        totals["recon_loss"] += float(recon_loss.item()) * n
        totals["kl"] += float(kl.item()) * n
        count += n
    return {key: value / count for key, value in totals.items()}


def train(model, train_loader, val_loader, args, device, out_dir):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            loss, _, _, _, _, _ = loss_and_stats(model, batch, device, args.kl_beta)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = int(batch[3].sum().item())
            total += float(loss.item()) * n
            count += n
        val = evaluate(model, val_loader, device, args.kl_beta)
        row = {"epoch": epoch + 1, "train_loss": total / count, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        if best is None or row["val_cos"] > best["val_cos"]:
            best = row
            torch.save({"model": model.state_dict(), "args": vars(args), "best_epoch": best}, out_dir / "model_best.pt")
        print(json.dumps(row), flush=True)
    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train a t2vec-style denoising GRU seq2seq autoencoder on latent trajectories.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--cell", choices=["gru", "rnn", "lstm"], default="gru")
    parser.add_argument("--vae", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=0)
    parser.add_argument("--kl_beta", type=float, default=0.0)
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
    train_rows, val_rows = split_rows(rows, args.val_fraction, args.seed)
    train_loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=make_collate(args.keep_prob, args.seed))
    val_loader = DataLoader(val_rows, batch_size=args.batch_size, shuffle=False, collate_fn=make_collate(args.keep_prob, args.seed + 1))
    model = T2VecLatentGRU(input_dim, args.hidden_dim, args.layers, args.bidirectional, args.cell, args.vae, args.latent_dim).to(device)
    history = train(model, train_loader, val_loader, args, device, out_dir)
    best_checkpoint = torch.load(out_dir / "model_best.pt", map_location="cpu", weights_only=False)

    metrics = {
        "features_npz": args.features_npz,
        "device": str(device),
        "model_type": f"t2vec_latent_{args.cell}",
        "input_dim": int(input_dim),
        "hidden_dim": int(args.hidden_dim),
        "layers": int(args.layers),
        "cell": args.cell,
        "vae": bool(args.vae),
        "latent_dim": int(args.latent_dim or args.hidden_dim),
        "kl_beta": float(args.kl_beta),
        "bidirectional": bool(args.bidirectional),
        "keep_prob": float(args.keep_prob),
        "trainable_parameters": parameter_count(model),
        "recordings": len(rows),
        "train_recordings": len(train_rows),
        "val_recordings": len(val_rows),
        "history": history,
        "best_epoch": best_checkpoint["best_epoch"],
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "final_val_mse": history[-1]["val_mse"],
        "final_val_cos": history[-1]["val_cos"],
        "final_val_pred_norm": history[-1]["val_pred_norm"],
        "final_val_recon_loss": history[-1]["val_recon_loss"],
        "final_val_kl": history[-1]["val_kl"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "args": vars(args), "metrics": metrics}, out_dir / "model.pt")
    best_checkpoint["metrics"] = metrics
    torch.save(best_checkpoint, out_dir / "model_best.pt")
    print(json.dumps({k: metrics[k] for k in ["trainable_parameters", "final_val_loss", "final_val_cos"]}, indent=2))


if __name__ == "__main__":
    main()
