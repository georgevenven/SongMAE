import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from individual_id.latent_trajectory_decoder.train_rope_sequence import (
    RopeSequenceDecoder,
    SinusoidalSequenceDecoder,
    load_rows,
    parameter_count,
    split_rows,
)


def fit_pca(train_rows, dim, seed):
    x = np.vstack([row[2] for row in train_rows]).astype(np.float32, copy=False)
    pca = PCA(n_components=dim, whiten=True, svd_solver="randomized", random_state=seed)
    pca.fit(x)
    return {
        "mean": pca.mean_.astype(np.float32),
        "components": pca.components_.astype(np.float32),
        "explained_variance": pca.explained_variance_.astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
    }


def apply_pca(x, transform):
    centered = x.astype(np.float32, copy=False) - transform["mean"]
    y = centered @ transform["components"].T
    y = y / np.sqrt(np.maximum(transform["explained_variance"], 1e-12))
    y = y.astype(np.float32, copy=False)
    return y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)


def save_pca_features(raw_npz, transform, out_path):
    data = np.load(raw_npz, allow_pickle=True)
    features = apply_pca(data["features"], transform)
    np.savez_compressed(
        out_path,
        features=features.astype(np.float32, copy=False),
        bird_labels=data["bird_labels"].astype(object, copy=False),
        syllable_labels=data["syllable_labels"].astype(np.int64, copy=False),
        recording_labels=data["recording_labels"].astype(object, copy=False),
    )


def chunk_rows(rows, max_seq_len):
    if max_seq_len <= 0:
        return rows
    chunks = []
    for recording, bird, seq in rows:
        chunk_index = 0
        for start in range(0, seq.shape[0] - 1, max_seq_len):
            end = min(seq.shape[0], start + max_seq_len + 1)
            if end - start <= 2:
                continue
            chunks.append((f"{recording}#chunk{chunk_index:04d}", bird, seq[start:end]))
            chunk_index += 1
    return chunks


def collate_next(rows):
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


def collate_self(rows):
    lengths = torch.tensor([row[2].shape[0] for row in rows], dtype=torch.long)
    max_len = int(lengths.max().item())
    input_dim = rows[0][2].shape[1]
    x = torch.zeros((len(rows), max_len, input_dim), dtype=torch.float32)
    y = torch.zeros((len(rows), max_len, input_dim), dtype=torch.float32)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    birds = []
    recordings = []
    for i, (recording, bird, seq) in enumerate(rows):
        n = seq.shape[0]
        x[i, :n] = torch.from_numpy(seq)
        y[i, :n] = torch.from_numpy(seq)
        mask[i, :n] = True
        birds.append(bird)
        recordings.append(recording)
    return x, y, mask, birds, recordings


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


def train(model, train_loader, val_loader, epochs, lr, device, best_path=None, checkpoint_args=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []
    best = None
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
        if best is None or row["val_cos"] > best["val_cos"]:
            best = row
            if best_path is not None:
                torch.save({"decoder": model.state_dict(), "args": checkpoint_args, "best_epoch": row}, best_path)
        print(json.dumps(row), flush=True)
    return history


def make_model(input_dim, args):
    if args.position_encoding == "rope":
        return RopeSequenceDecoder(input_dim, args.d_model, args.heads, args.layers)
    assert args.position_encoding == "sinusoidal"
    return SinusoidalSequenceDecoder(input_dim, args.d_model, args.heads, args.layers)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a transformer decoder on full or PCA-whitened L2-normalized latent sequences.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--pca_dim", type=int, default=128, help="Use 0 to train directly on full input vectors.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--position_encoding", choices=["rope", "sinusoidal"], default="rope")
    parser.add_argument("--target_mode", choices=["next", "self"], default="next")
    parser.add_argument("--train_on_all", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_len", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows, raw_dim = load_rows(args.features_npz)
    transform = None
    pca_features_npz = None
    if args.pca_dim > 0:
        train_raw, _ = split_rows(raw_rows, args.val_fraction, args.seed)
        transform = fit_pca(train_raw, args.pca_dim, args.seed)
        pca_features_npz = out_dir / f"pca{args.pca_dim}_whiten_l2_features.npz"
        save_pca_features(args.features_npz, transform, pca_features_npz)
        np.savez(out_dir / f"pca{args.pca_dim}_transform.npz", **transform)
        rows, input_dim = load_rows(pca_features_npz)
    else:
        rows = raw_rows
        input_dim = raw_dim
    if args.train_on_all:
        train_rows = list(rows)
        val_rows = list(rows)
    else:
        train_rows, val_rows = split_rows(rows, args.val_fraction, args.seed)
    train_rows = chunk_rows(train_rows, args.max_seq_len)
    val_rows = chunk_rows(val_rows, args.max_seq_len)
    collate = collate_self if args.target_mode == "self" else collate_next
    train_loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_rows, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = make_model(input_dim, args).to(device)
    history = train(
        model,
        train_loader,
        val_loader,
        args.epochs,
        args.lr,
        device,
        best_path=out_dir / "model_best.pt",
        checkpoint_args=vars(args),
    )

    metrics = {
        "raw_features_npz": args.features_npz,
        "pca_features_npz": None if pca_features_npz is None else str(pca_features_npz),
        "device": str(device),
        "raw_input_dim": int(raw_dim),
        "input_dim": int(input_dim),
        "pca_dim": int(args.pca_dim),
        "pca_explained_variance_ratio_sum": None if transform is None else float(transform["explained_variance_ratio"].sum()),
        "d_model": int(args.d_model),
        "heads": int(args.heads),
        "layers": int(args.layers),
        "position_encoding": args.position_encoding,
        "target_mode": args.target_mode,
        "train_on_all": bool(args.train_on_all),
        "max_seq_len": int(args.max_seq_len),
        "trainable_parameters": parameter_count(model),
        "recordings": len(rows),
        "train_recordings": len(train_rows),
        "val_recordings": len(val_rows),
        "train_tokens": int(sum(row[2].shape[0] - 1 for row in train_rows)),
        "val_tokens": int(sum(row[2].shape[0] - 1 for row in val_rows)),
        "decoder_objective": f"{args.target_mode}_{'full' if args.pca_dim <= 0 else 'pca_whiten_l2'}_cosine_full_variable_sequences_{args.position_encoding}",
        "output_for_loss": "l2_normalized_decoder_head",
        "history": history,
        "best_epoch": max(history, key=lambda row: row["val_cos"]),
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "final_val_mse": history[-1]["val_mse"],
        "final_val_cos": history[-1]["val_cos"],
        "final_val_pred_norm": history[-1]["val_pred_norm"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"decoder": model.state_dict(), "args": vars(args), "metrics": metrics}, out_dir / "model.pt")
    best_checkpoint = torch.load(out_dir / "model_best.pt", map_location="cpu", weights_only=False)
    best_checkpoint["metrics"] = metrics
    torch.save(best_checkpoint, out_dir / "model_best.pt")
    print(json.dumps({k: metrics[k] for k in ["trainable_parameters", "input_dim", "final_val_loss", "final_val_cos"]}, indent=2))


if __name__ == "__main__":
    main()
