#!/usr/bin/env python3
"""Tune a Transformer encoder on train/dev, refit on train, then test once."""

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.extract_embedding import load_recording_segments
from src.core.utils import load_model_from_checkpoint, resolve_run_dir, timebins_to_ms
from src.evals.syllable_classification import (
    MLP,
    SplitPolicy,
    build_syllable_split,
    budget_train_groups,
    class_counts,
    classes_for,
    class_labels,
    group_bounds,
    group_classes,
    group_seconds,
    load_units,
    max_train_seconds,
    raster_metrics,
    select_val_groups,
    train_group_order,
)
from src.external_models.aves import load_audio, load_model, min_input_samples
from src.external_models.data_loader import WavFromSpectrogramDataset, labels_for_features, limited_items
from src.external_models.hubert import load_model as load_hubert_model


@dataclass(frozen=True)
class Example:
    x: torch.Tensor
    labels: np.ndarray
    group: str
    stem: str
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class FineTuneSplit:
    train_groups: list[str]
    tune_groups: list[str]
    dev_groups: list[str]
    test_groups: list[str]
    train_order: list[str]
    train_seconds: float
    tune_seconds: float
    dev_seconds: float
    test_seconds: float
    missing_dev_classes: list[int]


@dataclass(frozen=True)
class TokenBatch:
    logits: torch.Tensor
    targets: torch.Tensor
    spans: list[tuple[str, int, int]]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patch_labels(labels, width):
    pooled = []
    for start in range(0, len(labels), width):
        values = labels[start : start + width]
        values = values[values >= 0]
        pooled.append(-1 if values.size == 0 else int(values.max()))
    return class_labels(pooled)


def token_spans(example, count, patch_width=None):
    if patch_width is None:
        edges = np.linspace(example.start_ms, example.end_ms, count + 1)
    else:
        offsets = np.minimum(np.arange(count + 1) * patch_width, len(example.labels))
        edges = example.start_ms + (example.end_ms - example.start_ms) * offsets / len(example.labels)
    edges = np.rint(edges).astype(np.int64)
    return [(example.stem, int(start), int(end)) for start, end in zip(edges[:-1], edges[1:])]


def make_token_batch(logits, counts, examples, label_fn, patch_width=None):
    rows, targets, spans = [], [], []
    for index, count in enumerate(counts):
        count = int(count)
        if count == 0:
            continue
        rows.append(logits[index, :count])
        targets.append(torch.as_tensor(label_fn(examples[index].labels, count), device=logits.device))
        spans.extend(token_spans(examples[index], count, patch_width))
    assert rows, "batch has no output tokens"
    return TokenBatch(torch.cat(rows), torch.cat(targets), spans)


def songmae_examples(args):
    run_dir = resolve_run_dir(args.songmae_run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    chunk_size = int(config["num_timebins"])
    assert args.chunk_timebins == chunk_size, "SongMAE chunk size must match its training window"
    raw = load_recording_segments(
        {
            "spec_dir": args.spec_dir,
            "json_path": args.annotation_file,
            "bird": args.bird,
            "recording_mode": "events",
            "num_timebins": args.num_timebins,
        }
    )
    examples = []
    for segment in raw["segments"]:
        labels = segment["labels_original"]
        spec = segment["spectrogram"]
        group = f'{segment["recording_stem"]}:{segment["song_id"]}'
        for start in range(0, len(labels), chunk_size):
            end = min(start + chunk_size, len(labels))
            x = torch.from_numpy(spec[:, start:end]).unsqueeze(0)
            x = F.pad(x, (0, chunk_size - x.shape[-1]))
            start_ms = segment["start_ms"] + timebins_to_ms(start, raw["audio_params"])
            end_ms = segment["start_ms"] + timebins_to_ms(end, raw["audio_params"])
            examples.append(Example(x, labels[start:end], group, segment["recording_stem"], start_ms, end_ms))
    return examples


def audio_examples(args):
    dataset = WavFromSpectrogramDataset(
        args.spec_dir,
        args.wav_dir,
        args.annotation_file,
        recording_mode="events",
        selected_bird=args.bird,
        wav_exts=args.wav_exts,
    )
    examples = []
    cache = {}
    for item in limited_items(dataset, args.num_timebins):
        labels = item["labels"].numpy()
        wav = load_audio(item, args.audio_sr, cache)
        group = f'{item["recording_stem"]}:{item["song_id"]}'
        for start in range(0, len(labels), args.chunk_timebins):
            end = min(start + args.chunk_timebins, len(labels))
            lo = round(start / len(labels) * len(wav))
            hi = round(end / len(labels) * len(wav))
            start_ms = item["start_ms"] + (item["end_ms"] - item["start_ms"]) * start / len(labels)
            end_ms = item["start_ms"] + (item["end_ms"] - item["start_ms"]) * end / len(labels)
            examples.append(Example(wav[lo:hi].clone(), labels[start:end], group, item["recording_stem"], start_ms, end_ms))
    return examples


def build_finetune_split(examples, units, args):
    spans = [(row.stem, round(row.start_ms), round(row.end_ms)) for row in examples]
    groups = [row.group for row in examples]
    outer = build_syllable_split(
        spans,
        groups,
        units,
        SplitPolicy(args.test_fraction, args.seed, None, True),
    )
    bounds = group_bounds(spans, groups)
    seconds = group_seconds(bounds)
    group_to_classes = group_classes(bounds, units)
    target_classes = classes_for(groups, group_to_classes)
    outer_train_classes = {group: group_to_classes[group] for group in outer.train_groups}
    strict_dev = all(count > 1 for count in class_counts(outer_train_classes).values())
    dev_groups = select_val_groups(
        np.asarray(outer.train_groups),
        outer_train_classes,
        SplitPolicy(args.dev_fraction, args.seed + 1, None, strict_dev),
    )
    candidates = np.asarray([group for group in outer.train_groups if group not in dev_groups])
    budget = max_train_seconds(args.max_train_seconds)
    tune_order = train_group_order(candidates, seconds, group_to_classes, args.seed)
    tune_groups = budget_train_groups(tune_order, seconds, budget)
    train_groups = budget_train_groups(outer.train_order, seconds, budget)
    assert classes_for(train_groups, group_to_classes) == target_classes, "Training budget does not cover every syllable class."
    assert classes_for(tune_groups, group_to_classes) == target_classes, "Tuning budget does not cover every syllable class."
    duration = lambda selected: round(sum(seconds[group] for group in sorted(selected)), 6)
    return FineTuneSplit(
        train_groups=sorted(train_groups),
        tune_groups=sorted(tune_groups),
        dev_groups=sorted(dev_groups),
        test_groups=outer.val_groups,
        train_order=outer.train_order,
        train_seconds=duration(train_groups),
        tune_seconds=duration(tune_groups),
        dev_seconds=duration(dev_groups),
        test_seconds=duration(outer.val_groups),
        missing_dev_classes=sorted(target_classes - classes_for(dev_groups, group_to_classes)),
    ), sorted({0, *target_classes})


class SongMAEFinetuner(nn.Module):
    def __init__(self, args, classes):
        super().__init__()
        self.backbone, self.config = load_model_from_checkpoint(args.songmae_run_dir, args.checkpoint)
        self.patch_width = int(self.config["patch_width"])
        self.patch_height = int(self.config["patch_height"])
        hidden = int(self.config["enc_hidden_d"]) * (int(self.config["mels"]) // self.patch_height)
        self.head = MLP(hidden, len(classes))
        self.backbone.requires_grad_(False)
        self.backbone.encoder.layers.requires_grad_(True)

    def train_mode(self):
        self.backbone.eval()
        self.head.train()
        self.backbone.encoder.layers.train()

    def forward_examples(self, examples):
        device = next(self.parameters()).device
        x = torch.stack([row.x for row in examples]).to(device)
        valid = torch.tensor([len(row.labels) for row in examples], device=device)
        hidden, _ = self.backbone.forward_encoder_inference(x, valid_timebins=valid)
        batch, _, dim = hidden.shape
        height = int(self.config["mels"]) // self.patch_height
        width = hidden.shape[1] // height
        hidden = hidden.reshape(batch, height, width, dim).permute(0, 2, 1, 3).flatten(2)
        counts = [(len(row.labels) + self.patch_width - 1) // self.patch_width for row in examples]
        return make_token_batch(self.head(hidden), counts, examples, lambda labels, _: patch_labels(labels, self.patch_width), self.patch_width)


class AvesFinetuner(nn.Module):
    def __init__(self, args, classes):
        super().__init__()
        self.backbone = load_model(args.aves_config_path, args.aves_model_path)
        self.minimum_samples = min_input_samples(self.backbone)
        hidden = int(json.loads(Path(args.aves_config_path).read_text())["encoder_embed_dim"])
        self.head = MLP(hidden, len(classes))
        self.backbone.requires_grad_(False)
        self.backbone.encoder.transformer.layers.requires_grad_(True)
        self.backbone.encoder.transformer.layer_norm.requires_grad_(True)

    def train_mode(self):
        self.backbone.eval()
        self.head.train()
        self.backbone.encoder.transformer.layers.train()
        self.backbone.encoder.transformer.layer_norm.train()

    def forward_examples(self, examples):
        device = next(self.parameters()).device
        logits, counts = [], []
        for row in examples:
            length = row.x.numel()
            wav = F.pad(row.x, (0, max(0, self.minimum_samples - length))).unsqueeze(0).to(device)
            hidden, out_length = self.backbone(wav, torch.tensor([length], device=device))
            logits.append(self.head(hidden[0]))
            counts.append(int(out_length[0]))
        return make_token_batch(
            pad_sequence(logits, batch_first=True),
            counts,
            examples,
            lambda labels, count: class_labels(labels_for_features(labels, count)),
        )


class HubertFinetuner(nn.Module):
    def __init__(self, args, classes):
        super().__init__()
        self.feature_extractor, self.backbone = load_hubert_model(args.model_name)
        self.audio_sr = args.audio_sr
        self.minimum_samples = min_input_samples(self.backbone)
        self.head = MLP(self.backbone.config.hidden_size, len(classes))
        self.backbone.requires_grad_(False)
        self.backbone.encoder.layers.requires_grad_(True)

    def train_mode(self):
        self.backbone.eval()
        self.head.train()
        self.backbone.encoder.layers.train()

    def forward_examples(self, examples):
        device = next(self.parameters()).device
        logits, counts = [], []
        for row in examples:
            wav = F.pad(row.x, (0, max(0, self.minimum_samples - row.x.numel())))
            values = wav.detach().cpu().numpy().astype(np.float32, copy=False)
            inputs = self.feature_extractor(values, sampling_rate=self.audio_sr, return_tensors="pt")
            hidden = self.backbone(input_values=inputs.input_values.to(device)).last_hidden_state
            logits.append(self.head(hidden[0]))
            counts.append(hidden.shape[1])
        return make_token_batch(
            pad_sequence(logits, batch_first=True),
            counts,
            examples,
            lambda labels, count: class_labels(labels_for_features(labels, count)),
        )


def make_model(args, classes, device):
    if args.model == "songmae":
        model = SongMAEFinetuner(args, classes)
    elif args.model == "aves":
        model = AvesFinetuner(args, classes)
    elif args.model == "hubert":
        model = HubertFinetuner(args, classes)
    else:
        raise ValueError(f"unknown model: {args.model}")
    return model.to(device)


def group_examples(examples, groups):
    groups = set(groups)
    rows = [row for row in examples if row.group in groups]
    assert rows
    return rows


def class_lookup(classes, device):
    lookup = torch.full((max(classes) + 1,), -1, dtype=torch.long, device=device)
    lookup[torch.tensor(classes, device=device)] = torch.arange(len(classes), device=device)
    return lookup


def train_epoch(model, examples, classes, args, optimizer, scaler, epoch):
    model.train_mode()
    generator = torch.Generator().manual_seed(args.seed + epoch)
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True, collate_fn=list, generator=generator)
    lookup = class_lookup(classes, next(model.parameters()).device)
    total_loss = 0.0
    total_tokens = 0
    for rows in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
            batch = model.forward_examples(rows)
            targets = lookup[batch.targets]
            assert int(targets.min()) >= 0, "batch contains a class absent from training"
            loss = F.cross_entropy(batch.logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * targets.numel()
        total_tokens += targets.numel()
    return total_loss / total_tokens


@torch.no_grad()
def evaluate(model, examples, classes, units, args):
    model.eval()
    predictions, spans = [], []
    class_tensor = torch.tensor(classes, device=next(model.parameters()).device)
    for rows in DataLoader(examples, batch_size=args.batch_size, collate_fn=list):
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=class_tensor.is_cuda):
            batch = model.forward_examples(rows)
        predictions.extend(class_tensor[batch.logits.argmax(dim=1)].cpu().tolist())
        spans.extend(batch.spans)
    return raster_metrics(np.asarray(predictions), spans, units)


def make_optimizer(model, encoder_lr, args):
    head = list(model.head.parameters())
    head_ids = {id(parameter) for parameter in head}
    encoder = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in head_ids]
    return torch.optim.AdamW(
        [{"params": encoder, "lr": encoder_lr}, {"params": head, "lr": args.head_lr}],
        weight_decay=args.weight_decay,
    )


def fit(args, examples, split, classes, units, device):
    history = []
    selected = (None, args.encoder_lrs[0], args.epochs) if args.fixed else None
    selected_dev = None

    if args.fixed:
        assert len(args.encoder_lrs) == 1
    else:
        tune_rows = group_examples(examples, split.tune_groups)
        dev_rows = group_examples(examples, split.dev_groups)
        for encoder_lr in args.encoder_lrs:
            seed_everything(args.seed)
            model = make_model(args, classes, device)
            optimizer = make_optimizer(model, encoder_lr, args)
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
            lr_history = []
            for epoch in range(1, args.epochs + 1):
                train_loss = train_epoch(model, tune_rows, classes, args, optimizer, scaler, epoch)
                dev = evaluate(model, dev_rows, classes, units, args)
                lr_history.append({"epoch": epoch, "train_loss": train_loss, "dev_macro_fer": dev["macro_fer"]})
                key = (dev["macro_fer"], encoder_lr, epoch)
                if selected is None or key < selected:
                    selected = key
                    selected_dev = dev
            history.append({"encoder_lr": encoder_lr, "epochs": lr_history})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    assert selected is not None
    seed_everything(args.seed)
    model = make_model(args, classes, device)
    optimizer = make_optimizer(model, selected[1], args)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_rows = group_examples(examples, split.train_groups)
    refit_loss = [
        train_epoch(model, train_rows, classes, args, optimizer, scaler, epoch)
        for epoch in range(1, selected[2] + 1)
    ]
    test = evaluate(model, group_examples(examples, split.test_groups), classes, units, args)
    test.update(
        {
            "selected_encoder_lr": selected[1],
            "selected_epoch": selected[2],
            "encoder_scope": "all_blocks",
            "classifier": "mlp_1024_256",
            "head_lr": args.head_lr,
            "selection_mode": "fixed" if args.fixed else "dev",
            "dev_macro_fer": None if selected_dev is None else selected_dev["macro_fer"],
            "lr_history": history,
            "refit_train_loss": refit_loss,
            "train_seconds": split.train_seconds,
            "tune_seconds": split.tune_seconds,
            "dev_seconds": split.dev_seconds,
            "test_seconds": split.test_seconds,
            "train_groups": len(split.train_groups),
            "tune_groups": len(split.tune_groups),
            "dev_groups": len(split.dev_groups),
            "test_groups": len(split.test_groups),
            "missing_dev_classes": split.missing_dev_classes,
        }
    )
    return test


def parse_lrs(value):
    values = sorted({float(item) for item in value.split(",")})
    assert values and all(value > 0 for value in values)
    return values


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a pretrained encoder for syllable classification.")
    parser.add_argument("--spec_dir", required=True)
    parser.add_argument("--annotation_file", required=True)
    parser.add_argument("--bird", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_train_seconds", default="MAX")
    parser.add_argument("--encoder_lrs", type=parse_lrs, default=parse_lrs("1e-5,5e-5,1e-4"))
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--dev_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_timebins", type=int, default=720000)
    parser.add_argument("--chunk_timebins", type=int, default=1000)
    parser.add_argument("--fixed", action="store_true")
    subparsers = parser.add_subparsers(dest="model", required=True)

    songmae = subparsers.add_parser("songmae")
    songmae.add_argument("--songmae_run_dir", required=True)
    songmae.add_argument("--checkpoint")

    aves = subparsers.add_parser("aves")
    aves.add_argument("--wav_dir", required=True)
    aves.add_argument("--aves_model_path", required=True)
    aves.add_argument("--aves_config_path", required=True)
    aves.add_argument("--audio_sr", type=int, default=16000)
    aves.add_argument("--wav_exts", default=".wav,.flac,.ogg,.mp3")

    hubert = subparsers.add_parser("hubert")
    hubert.add_argument("--wav_dir", required=True)
    hubert.add_argument("--model_name", default="facebook/hubert-base-ls960")
    hubert.add_argument("--audio_sr", type=int, default=16000)
    hubert.add_argument("--wav_exts", default=".wav,.flac,.ogg,.mp3")
    return parser.parse_args()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    args = parse_args()
    assert args.epochs > 0 and args.batch_size > 0
    examples = songmae_examples(args) if args.model == "songmae" else audio_examples(args)
    units = load_units(args.annotation_file)
    split, classes = build_finetune_split(examples, units, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = fit(args, examples, split, classes, units, device)
    output = Path(args.out_dir)
    write_json(output / "config.json", {**vars(args), "encoder_scope": "all_blocks"})
    write_json(output / "split.json", asdict(split))
    write_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
