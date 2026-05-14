import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from individual_id.latent_trajectory_decoder.train_rope_sequence import load_rows, parameter_count, split_rows
from individual_id.latent_trajectory_decoder.train_t2vec_gru_latents import T2VecLatentGRU, corrupt_sequence


def group_by_bird(rows):
    by_bird = defaultdict(list)
    for row in rows:
        by_bird[row[1]].append(row)
    return {bird: bird_rows for bird, bird_rows in by_bird.items() if len(bird_rows) >= 2}


def crop_window(seq, window_size, rng):
    if window_size <= 0:
        return seq
    if seq.shape[0] <= window_size:
        return seq
    start = int(rng.integers(0, seq.shape[0] - window_size + 1))
    return seq[start : start + window_size]


def pad_sequences(seqs, input_dim):
    lengths = torch.tensor([seq.shape[0] for seq in seqs], dtype=torch.long)
    x = torch.zeros((len(seqs), int(lengths.max().item()), input_dim), dtype=torch.float32)
    mask = torch.zeros((len(seqs), x.shape[1]), dtype=torch.bool)
    for i, seq in enumerate(seqs):
        x[i, : seq.shape[0]] = torch.from_numpy(seq)
        mask[i, : seq.shape[0]] = True
    return x, lengths, mask


def sample_batch(rows, by_bird, birds, bird_to_id, args, rng, input_dim):
    clean = []
    enc = []
    labels = []
    if args.contrastive_weight <= 0.0:
        for row_index in rng.choice(len(rows), size=args.batch_pairs * 2, replace=True):
            window = crop_window(rows[int(row_index)][2], args.window_size, rng)
            clean.append(window)
            enc.append(corrupt_sequence(window, args.keep_prob, rng))
            labels.append(0)
    elif args.positive_scope == "recording":
        for pair_id, row_index in enumerate(rng.choice(len(rows), size=args.batch_pairs, replace=True)):
            for _ in range(2):
                window = crop_window(rows[int(row_index)][2], args.window_size, rng)
                clean.append(window)
                enc.append(corrupt_sequence(window, args.keep_prob, rng))
                labels.append(pair_id)
    elif args.positive_scope == "bird":
        for bird in rng.choice(birds, size=args.batch_pairs, replace=True):
            picks = rng.choice(len(by_bird[bird]), size=2, replace=False)
            for pick in picks:
                window = crop_window(by_bird[bird][int(pick)][2], args.window_size, rng)
                clean.append(window)
                enc.append(corrupt_sequence(window, args.keep_prob, rng))
                labels.append(bird_to_id[bird])
    else:
        raise AssertionError(args.positive_scope)
    enc_x, enc_lengths, _ = pad_sequences(enc, input_dim)
    clean_x, _, mask = pad_sequences(clean, input_dim)
    return enc_x, enc_lengths, clean_x, mask, torch.tensor(labels, dtype=torch.long)


def supcon_loss(z, labels, temperature):
    z = F.normalize(z, dim=-1)
    logits = z @ z.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
    pos = labels[:, None].eq(labels[None, :]) & ~self_mask
    denom = torch.logsumexp(logits.masked_fill(self_mask, -torch.inf), dim=1)
    numer = torch.logsumexp(logits.masked_fill(~pos, -torch.inf), dim=1)
    return (denom - numer)[pos.any(dim=1)].mean()


def step(model, batch, device, args):
    enc, enc_lengths, clean, mask, labels = batch
    enc = enc.to(device)
    enc_lengths = enc_lengths.to(device)
    clean = clean.to(device)
    mask = mask.to(device)
    pred, _, bottleneck, _ = model(enc, enc_lengths, clean)
    pred_unit = F.normalize(pred, dim=-1)
    clean_unit = F.normalize(clean, dim=-1)
    cosine = (pred_unit * clean_unit).sum(dim=-1)
    recon = (1.0 - cosine)[mask].mean()
    if args.contrastive_weight <= 0.0:
        return recon, recon, recon.new_zeros(()), cosine[mask].mean()
    labels = labels.to(device)
    contrast = supcon_loss(bottleneck, labels, args.temperature)
    loss = recon + args.contrastive_weight * contrast
    return loss, recon, contrast, cosine[mask].mean()


@torch.no_grad()
def evaluate(model, rows, by_bird, birds, bird_to_id, args, rng, input_dim, device):
    model.eval()
    totals = {"loss": 0.0, "recon": 0.0, "contrast": 0.0, "cos": 0.0}
    for _ in range(args.eval_steps):
        batch = sample_batch(rows, by_bird, birds, bird_to_id, args, rng, input_dim)
        loss, recon, contrast, cos = step(model, batch, device, args)
        totals["loss"] += float(loss.item())
        totals["recon"] += float(recon.item())
        totals["contrast"] += float(contrast.item())
        totals["cos"] += float(cos.item())
    return {key: value / args.eval_steps for key, value in totals.items()}


def train(model, train_rows, val_rows, args, input_dim, device, out_dir):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    train_groups = group_by_bird(train_rows)
    val_groups = group_by_bird(val_rows)
    birds = sorted(train_groups)
    val_birds = sorted(val_groups)
    bird_to_id = {bird: i for i, bird in enumerate(sorted(set(birds) | set(val_birds)))}
    train_rng = np.random.default_rng(args.seed)
    val_rng = np.random.default_rng(args.seed + 1)
    best = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        totals = {"loss": 0.0, "recon": 0.0, "contrast": 0.0, "cos": 0.0}
        for _ in range(args.steps_per_epoch):
            batch = sample_batch(train_rows, train_groups, birds, bird_to_id, args, train_rng, input_dim)
            loss, recon, contrast, cos = step(model, batch, device, args)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            totals["loss"] += float(loss.item())
            totals["recon"] += float(recon.item())
            totals["contrast"] += float(contrast.item())
            totals["cos"] += float(cos.item())
        val = evaluate(model, val_rows, val_groups, val_birds, bird_to_id, args, val_rng, input_dim, device)
        row = {
            "epoch": epoch + 1,
            **{f"train_{key}": value / args.steps_per_epoch for key, value in totals.items()},
            **{f"val_{key}": value for key, value in val.items()},
        }
        history.append(row)
        if best is None or row["val_cos"] > best["val_cos"]:
            best = row
            torch.save({"model": model.state_dict(), "args": vars(args), "best_epoch": best}, out_dir / "model_best.pt")
        print(json.dumps(row), flush=True)
    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train a random-window contrastive t2vec GRU on latent trajectories.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--window_size", type=int, default=48)
    parser.add_argument("--keep_prob", type=float, default=0.8)
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--positive_scope", choices=["recording", "bird"], default="recording")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=25)
    parser.add_argument("--batch_pairs", type=int, default=16)
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
    model = T2VecLatentGRU(input_dim, args.hidden_dim, args.layers, args.bidirectional, "gru").to(device)
    history = train(model, train_rows, val_rows, args, input_dim, device, out_dir)
    best_checkpoint = torch.load(out_dir / "model_best.pt", map_location="cpu", weights_only=False)
    metrics = {
        "features_npz": args.features_npz,
        "device": str(device),
        "model_type": "t2vec_latent_gru_random_window_contrastive",
        "input_dim": int(input_dim),
        "hidden_dim": int(args.hidden_dim),
        "layers": int(args.layers),
        "bidirectional": bool(args.bidirectional),
        "window_size": int(args.window_size),
        "keep_prob": float(args.keep_prob),
        "contrastive_weight": float(args.contrastive_weight),
        "positive_scope": args.positive_scope,
        "temperature": float(args.temperature),
        "trainable_parameters": parameter_count(model),
        "recordings": len(rows),
        "train_recordings": len(train_rows),
        "val_recordings": len(val_rows),
        "train_birds": len(group_by_bird(train_rows)),
        "val_birds": len(group_by_bird(val_rows)),
        "history": history,
        "best_epoch": best_checkpoint["best_epoch"],
        **{f"final_{key}": value for key, value in history[-1].items() if key != "epoch"},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "args": vars(args), "metrics": metrics}, out_dir / "model.pt")
    best_checkpoint["metrics"] = metrics
    torch.save(best_checkpoint, out_dir / "model_best.pt")
    print(json.dumps({k: metrics[k] for k in ["trainable_parameters", "final_val_cos", "final_val_contrast"]}, indent=2))


if __name__ == "__main__":
    main()
