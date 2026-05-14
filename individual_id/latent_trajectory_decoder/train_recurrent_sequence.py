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
from individual_id.latent_trajectory_decoder.train_rope_sequence import collate, load_rows, parameter_count, split_rows


class RecurrentSequenceDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, cell):
        super().__init__()
        assert cell in {"lstm", "rnn"}
        recurrent = nn.LSTM if cell == "lstm" else nn.RNN
        self.cell = cell
        self.rnn = recurrent(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, valid):
        h, _ = self.rnn(x)
        h = self.norm(h)
        return self.head(h), h


def loss_and_stats(model, batch, device):
    x, y, valid, _, _ = batch
    x = x.to(device)
    y = y.to(device)
    valid = valid.to(device)
    pred, _ = model(x, valid)
    pred_unit = F.normalize(pred, dim=-1)
    target_unit = F.normalize(y, dim=-1)
    cosine = (pred_unit * target_unit).sum(dim=-1)
    loss = (1.0 - cosine)[valid].mean()
    mse = (pred_unit - target_unit).square().mean(dim=-1)
    return loss, mse[valid].mean(), cosine[valid].mean(), pred.norm(dim=-1)[valid].mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = {"loss": 0.0, "mse": 0.0, "cos": 0.0, "pred_norm": 0.0}
    count = 0
    for batch in loader:
        loss, mse, cos, pred_norm = loss_and_stats(model, batch, device)
        n = int(batch[2].sum().item())
        totals["loss"] += float(loss.item()) * n
        totals["mse"] += float(mse.item()) * n
        totals["cos"] += float(cos.item()) * n
        totals["pred_norm"] += float(pred_norm.item()) * n
        count += n
    return {key: value / count for key, value in totals.items()}


def train(model, train_loader, val_loader, epochs, lr, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            loss, _, _, _ = loss_and_stats(model, batch, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            n = int(batch[2].sum().item())
            total += float(loss.item()) * n
            count += n
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": total / count, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)
    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train an LSTM/RNN next-latent decoder on sequence embeddings.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cell", choices=["lstm", "rnn"], default="lstm")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
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
    model = RecurrentSequenceDecoder(input_dim, args.hidden_dim, args.layers, args.cell).to(device)
    history = train(model, train_loader, val_loader, args.epochs, args.lr, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "features_npz": args.features_npz,
        "device": str(device),
        "model_type": "recurrent",
        "cell": args.cell,
        "input_dim": int(input_dim),
        "hidden_dim": int(args.hidden_dim),
        "layers": int(args.layers),
        "trainable_parameters": parameter_count(model),
        "recordings": len(rows),
        "train_recordings": len(train_rows),
        "val_recordings": len(val_rows),
        "train_tokens": int(sum(row[2].shape[0] - 1 for row in train_rows)),
        "val_tokens": int(sum(row[2].shape[0] - 1 for row in val_rows)),
        "decoder_objective": "next_pca_whiten_l2_cosine_recurrent",
        "output_for_loss": "l2_normalized_decoder_head",
        "history": history,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "final_val_mse": history[-1]["val_mse"],
        "final_val_cos": history[-1]["val_cos"],
        "final_val_pred_norm": history[-1]["val_pred_norm"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"decoder": model.state_dict(), "args": vars(args), "metrics": metrics}, out_dir / "model.pt")
    print(json.dumps({k: metrics[k] for k in ["trainable_parameters", "cell", "final_val_loss", "final_val_cos"]}, indent=2))


if __name__ == "__main__":
    main()
