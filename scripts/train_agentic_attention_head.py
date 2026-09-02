#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.qwen_box_tokens import BoxTokens, read_rows, split_rows
from src.core.data_structures import AudioParams
from src.core.utils import load_model_from_checkpoint


class AttentionHead(nn.Module):
    def __init__(self, input_dim, hidden, height, width, patch_width, position=True, dropout=.1):
        super().__init__()
        self.height, self.width, self.patch_width = height, width, patch_width
        self.project = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden))
        self.freq = nn.Parameter(torch.zeros(1, height, 1, hidden), requires_grad=position)
        self.time = nn.Parameter(torch.zeros(1, 1, width, hidden), requires_grad=position)
        self.block = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout, "gelu", batch_first=True, norm_first=True)
        self.output = nn.Linear(hidden, 1)

    def forward(self, tokens, valid):
        batch = len(tokens)
        x = self.project(tokens).reshape(batch, self.height, self.width, -1) + self.freq + self.time
        columns = torch.arange(self.width, device=x.device).repeat(self.height)
        valid = (valid.to(x.device) + self.patch_width - 1) // self.patch_width
        padding = columns[None] >= valid[:, None]
        return self.output(self.block(x.flatten(1, 2), src_key_padding_mask=padding)).squeeze(-1), ~padding


def counts(labels, scores, threshold):
    prediction, truth = scores >= threshold, labels.astype(bool)
    return np.asarray([(prediction & truth).sum(), (prediction & ~truth).sum(), (~prediction & truth).sum()])


def prf(values):
    tp, fp, fn = values
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return float(precision), float(recall), float(2 * precision * recall / max(precision + recall, 1e-12))


def targets_to_grid(targets, height, width, patch_width):
    targets = targets.reshape(len(targets), height, -1)
    if patch_width > 1:
        targets = F.max_pool1d(targets, patch_width, patch_width)
    return targets[:, :, :width].flatten(1)


def best(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(np.nanargmax(f1[:-1]))
    return float(thresholds[index]), float(precision[index]), float(recall[index]), float(f1[index])


@torch.no_grad()
def evaluate(backbone, head, loader, data, device):
    head.eval()
    output, losses, offset = [], [], 0
    for specs, targets, valid in loader:
        tokens, _ = backbone.forward_encoder_inference(specs.to(device), valid_timebins=valid.to(device))
        logits, keep = head(tokens.float(), valid)
        truth = targets_to_grid(targets.to(device), data.height, head.width, head.patch_width)
        losses.append(float(F.binary_cross_entropy_with_logits(logits[keep], truth[keep])))
        for row in range(len(specs)):
            output.append((data.windows[offset + row][0]["recording"], truth[row][keep[row]].cpu().numpy(), logits[row][keep[row]].cpu().numpy()))
        offset += len(specs)
    return float(np.mean(losses)), output


def summarize(rows, threshold):
    total = np.zeros(3, np.int64)
    recordings = defaultdict(lambda: np.zeros(3, np.int64))
    for recording, labels, scores in rows:
        value = counts(labels, scores, threshold)
        total += value
        recordings[recording] += value
    return {"precision": prf(total)[0], "recall": prf(total)[1], "micro_f1": prf(total)[2],
        "recording_macro_f1": float(np.mean([prf(x)[2] for x in recordings.values()])), "counts": total.tolist()}


def main():
    parser = argparse.ArgumentParser(description="Train one contextual token layer from Qwen boxes.")
    parser.add_argument("--annotations", type=Path, default=Path("data/XCL/qwen38_adaptive_review_5s_annotations.jsonl"))
    parser.add_argument("--run", type=Path, default=Path("runs/xcl_large_500k_p32x4_c0025"))
    parser.add_argument("--checkpoint", default="model_step_499999.pth")
    parser.add_argument("--out", type=Path, default=Path("runs/song_unit_32x4/adaptive_attention_head.pt"))
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--no-position", action="store_true")
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args()
    torch.manual_seed(options.seed)
    device = torch.device("cuda")
    backbone, config = load_model_from_checkpoint(options.run, options.checkpoint)
    backbone.requires_grad_(False).eval().to(device)
    train_rows, val_rows = split_rows(read_rows(options.annotations, None, options.seed), options.val_fraction, options.seed)
    audio = AudioParams.from_dir(options.run)
    train = BoxTokens(train_rows, audio, config["num_timebins"], config["patch_height"])
    val = BoxTokens(val_rows, audio, config["num_timebins"], config["patch_height"])
    width = config["num_timebins"] // config["patch_width"]
    head = AttentionHead(config["enc_hidden_d"], options.hidden, train.height, width, config["patch_width"],
        not options.no_position, options.dropout).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), 1e-3, weight_decay=options.weight_decay)
    loaders = [DataLoader(x, options.batch_size, shuffle=i == 0, num_workers=2, pin_memory=True) for i, x in enumerate((train, val))]
    best_state, best_loss = None, float("inf")
    print(f"{len(train)} train / {len(val)} validation windows; {sum(p.numel() for p in head.parameters() if p.requires_grad):,} trainable parameters", flush=True)
    for epoch in range(1, options.epochs + 1):
        head.train()
        losses = []
        for specs, targets, valid in loaders[0]:
            with torch.no_grad():
                tokens, _ = backbone.forward_encoder_inference(specs.to(device), valid_timebins=valid.to(device))
            logits, keep = head(tokens.float(), valid)
            truth = targets_to_grid(targets.to(device), train.height, width, config["patch_width"])
            loss = F.binary_cross_entropy_with_logits(logits[keep], truth[keep])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        val_loss, rows = evaluate(backbone, head, loaders[1], val, device)
        labels, scores = np.concatenate([x[1] for x in rows]), np.concatenate([x[2] for x in rows])
        metric = best(labels, scores)
        print(f"epoch {epoch}: train={np.mean(losses):.4f} val={val_loss:.4f} f1={metric[3]:.3f} p={metric[1]:.3f} r={metric[2]:.3f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu() for key, value in head.state_dict().items()}
    head.load_state_dict(best_state)
    _, rows = evaluate(backbone, head, loaders[1], val, device)
    recordings = sorted({x[0] for x in rows})
    np.random.default_rng(123).shuffle(recordings)
    calibration = set(recordings[:len(recordings) // 2])
    calibration_rows = [x for x in rows if x[0] in calibration]
    test_rows = [x for x in rows if x[0] not in calibration]
    threshold = best(np.concatenate([x[1] for x in calibration_rows]), np.concatenate([x[2] for x in calibration_rows]))[0]
    metrics = {"best_val_loss": best_loss, "threshold": threshold, "calibration_recordings": len(calibration),
        "test_recordings": len(set(recordings) - calibration), "untouched_test": summarize(test_rows, threshold)}
    options.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": best_state, "hidden": options.hidden, "height": train.height, "width": width,
        "position": not options.no_position, "dropout": options.dropout, "weight_decay": options.weight_decay,
        "val_fraction": options.val_fraction,
        "patch_size": [config["patch_height"], config["patch_width"]], "backbone_run": str(options.run),
        "checkpoint": options.checkpoint, "threshold": threshold, "metrics": metrics}, options.out)
    print(json.dumps(metrics, indent=2))
    print(f"saved {options.out}")


if __name__ == "__main__":
    main()
