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
from individual_id.latent_trajectory_decoder.train_t2vec_gru_latents import make_collate


class BottleneckGPT(nn.Module):
    def __init__(self, input_dim, model_dim, bottleneck_dim, encoder_layers, decoder_layers, heads, ff_dim, max_len):
        super().__init__()
        self.model_dim = model_dim
        self.max_len = max_len
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.encoder_pos = nn.Parameter(torch.zeros(max_len, model_dim))
        self.decoder_pos = nn.Parameter(torch.zeros(max_len + 1, model_dim))
        enc_layer = nn.TransformerEncoderLayer(model_dim, heads, ff_dim, batch_first=True, norm_first=True)
        dec_layer = nn.TransformerEncoderLayer(model_dim, heads, ff_dim, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, encoder_layers)
        self.decoder = nn.TransformerEncoder(dec_layer, decoder_layers)
        self.to_bottleneck = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, bottleneck_dim), nn.LayerNorm(bottleneck_dim))
        self.to_prefix = nn.Linear(bottleneck_dim, model_dim)
        self.start = nn.Parameter(torch.zeros(input_dim))
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, input_dim)

    def encode(self, x, lengths):
        assert x.shape[1] <= self.max_len
        mask = torch.arange(x.shape[1], device=x.device)[None] >= lengths[:, None]
        h = self.input_proj(x) + self.encoder_pos[: x.shape[1]]
        h = self.encoder(h, src_key_padding_mask=mask)
        last = h[torch.arange(x.shape[0], device=x.device), lengths - 1]
        return self.to_bottleneck(last)

    def forward(self, enc_x, enc_lengths, clean_x):
        bottleneck = self.encode(enc_x, enc_lengths)
        start = self.start[None, None].expand(clean_x.shape[0], 1, clean_x.shape[2])
        dec_in = torch.cat([start, clean_x[:, :-1]], dim=1)
        prefix = self.to_prefix(bottleneck)[:, None]
        h = torch.cat([prefix, self.input_proj(dec_in)], dim=1)
        assert h.shape[1] <= self.max_len + 1
        h = h + self.decoder_pos[: h.shape[1]]
        causal = torch.triu(torch.ones((h.shape[1], h.shape[1]), device=h.device, dtype=torch.bool), diagonal=1)
        h = self.decoder(h, mask=causal)
        h = self.norm(h[:, 1:])
        return self.head(h), h, bottleneck


def loss_and_stats(model, batch, device):
    enc, enc_lengths, clean, mask, _, _ = batch
    enc = enc.to(device)
    enc_lengths = enc_lengths.to(device)
    clean = clean.to(device)
    mask = mask.to(device)
    pred, _, _ = model(enc, enc_lengths, clean)
    pred_unit = F.normalize(pred, dim=-1)
    clean_unit = F.normalize(clean, dim=-1)
    cosine = (pred_unit * clean_unit).sum(dim=-1)
    loss = (1.0 - cosine)[mask].mean()
    mse = (pred_unit - clean_unit).square().mean(dim=-1)
    return loss, mse[mask].mean(), cosine[mask].mean(), pred.norm(dim=-1)[mask].mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = {"loss": 0.0, "mse": 0.0, "cos": 0.0, "pred_norm": 0.0}
    count = 0
    for batch in loader:
        loss, mse, cos, pred_norm = loss_and_stats(model, batch, device)
        n = int(batch[3].sum().item())
        totals["loss"] += float(loss.item()) * n
        totals["mse"] += float(mse.item()) * n
        totals["cos"] += float(cos.item()) * n
        totals["pred_norm"] += float(pred_norm.item()) * n
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
            loss, _, _, _ = loss_and_stats(model, batch, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = int(batch[3].sum().item())
            total += float(loss.item()) * n
            count += n
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": total / count, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        if best is None or row["val_cos"] > best["val_cos"]:
            best = row
            torch.save({"model": model.state_dict(), "args": vars(args), "best_epoch": best}, out_dir / "model_best.pt")
        print(json.dumps(row), flush=True)
    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train a bottlenecked Transformer/GPT denoising autoencoder on latent trajectories.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--bottleneck_dim", type=int, default=512)
    parser.add_argument("--encoder_layers", type=int, default=2)
    parser.add_argument("--decoder_layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=2048)
    parser.add_argument("--max_len", type=int, default=256)
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
    model = BottleneckGPT(input_dim, args.model_dim, args.bottleneck_dim, args.encoder_layers, args.decoder_layers, args.heads, args.ff_dim, args.max_len).to(device)
    history = train(model, train_loader, val_loader, args, device, out_dir)
    best_checkpoint = torch.load(out_dir / "model_best.pt", map_location="cpu", weights_only=False)

    metrics = {
        "features_npz": args.features_npz,
        "device": str(device),
        "model_type": "t2vec_latent_bottleneck_gpt",
        "input_dim": int(input_dim),
        "model_dim": int(args.model_dim),
        "bottleneck_dim": int(args.bottleneck_dim),
        "encoder_layers": int(args.encoder_layers),
        "decoder_layers": int(args.decoder_layers),
        "heads": int(args.heads),
        "ff_dim": int(args.ff_dim),
        "max_len": int(args.max_len),
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
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "args": vars(args), "metrics": metrics}, out_dir / "model.pt")
    best_checkpoint["metrics"] = metrics
    torch.save(best_checkpoint, out_dir / "model_best.pt")
    print(json.dumps({k: metrics[k] for k in ["trainable_parameters", "final_val_loss", "final_val_cos"]}, indent=2))


if __name__ == "__main__":
    main()
