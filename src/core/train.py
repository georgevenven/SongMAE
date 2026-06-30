import argparse
import json
import math
import os
import shutil
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .model import SongMAE
from .data_loader import SpectrogramDataset, SpectrogramDatasetSupervised
from .data_structures import AudioParams, ModelConfig, TrainConfig
from src.plotting_utils.pretrain_plotting import save_loss_curve, save_masked_reconstruction_plot


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "runs"

DEFAULT_CONFIG = {
    "task": "unsupervised",
    "steps": 500_000,
    "lr": 1e-4,
    "batch_size": 128,
    "num_workers": 8,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "eval_every": 500,
    "save_every": 5_000,
    "warmup_steps": 5_000,
    "min_lr": 1e-5,
    "patch_height": 32,
    "patch_width": 1,
    "num_timebins": 1000,
    "dropout": 0.1,
    "mask_p": 0.75,
    "mask_c": 0.1,
    "mask_type": "voronoi",
    "enc_hidden_d": 384,
    "enc_n_head": 6,
    "enc_n_layer": 6,
    "enc_dim_ff": 1536,
    "dec_hidden_d": 192,
    "dec_n_head": 6,
    "dec_n_layer": 2,
    "dec_dim_ff": 768,
    "amp": True,
    "amp_dtype": "fp16",
    "wandb": True,
    "recording_mode": "full_recordings",
}

MODEL_PRESETS = {
    "base": {
        "enc_hidden_d": 384,
        "enc_n_head": 6,
        "enc_n_layer": 6,
        "enc_dim_ff": 1536,
        "dec_hidden_d": 192,
        "dec_n_head": 3,
        "dec_n_layer": 1,
        "dec_dim_ff": 768,
    },
    "tiny": {
        "enc_hidden_d": 256,
        "enc_n_head": 4,
        "enc_n_layer": 6,
        "enc_dim_ff": 1024,
        "dec_hidden_d": 128,
        "dec_n_head": 2,
        "dec_n_layer": 1,
        "dec_dim_ff": 512,
    },
    "micro": {
        "enc_hidden_d": 128,
        "enc_n_head": 2,
        "enc_n_layer": 6,
        "enc_dim_ff": 512,
        "dec_hidden_d": 64,
        "dec_n_head": 1,
        "dec_n_layer": 1,
        "dec_dim_ff": 256,
    },
}

# Run folder layout:
# runs/<run_name>/
#   audio_params.json   copied from the training spectrogram folder
#   config.json         legacy combined config, kept for older scripts/checkpoints
#   model.json          model architecture and input contract
#   train.json          training recipe for this run
#   logs.txt            TUI training log and refactor notes
#   losses.csv          every train step and validation loss
#   weights/            model_step_000000.pth, ...
#   imgs/               optional visualizations


class Trainer:
    def __init__(self, config):
        self.config = config
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.distributed = self.world_size > 1
        if self.distributed:
            assert torch.cuda.is_available(), "DDP training expects CUDA/NCCL"
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
        self.is_main = self.rank == 0
        self.run_dir = self.setup_run()
        self.logs_path = self.run_dir / "logs.txt"
        self.losses_path = self.run_dir / "losses.csv"
        self.weights_dir = self.run_dir / "weights"
        self.imgs_dir = self.run_dir / "imgs"
        if self.is_main:
            self.weights_dir.mkdir(exist_ok=True)
            self.imgs_dir.mkdir(exist_ok=True)
            if not self.losses_path.exists():
                self.losses_path.write_text("step,split,loss\n")
        self.barrier()

        self.device = torch.device(f"cuda:{self.local_rank}" if self.distributed else "cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = config.get("amp", False) and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if config.get("amp_dtype", "fp16") == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.amp_dtype is torch.float16)
        self.model = self.build_model().to(self.device)
        self.start_step = self.load_checkpoint() if config.get("continue_from") else 0
        if self.distributed:
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank])
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 0.0),
        )
        self.steps = config["steps"]
        self.end_step = self.start_step + self.steps - 1
        self.eval_every = config.get("eval_every", 500)
        self.save_every = config.get("save_every", 5_000)
        self.grad_clip = config.get("grad_clip", 1.0)
        assert self.grad_clip > 0
        global_batch_size = config["batch_size"]
        assert global_batch_size >= self.world_size
        base_batch, extra_batches = divmod(global_batch_size, self.world_size)
        rank_batch_sizes = [base_batch + int(rank < extra_batches) for rank in range(self.world_size)]
        self.batch_size = rank_batch_sizes[self.rank]
        self.loader_batch_size = rank_batch_sizes[0]
        n_params = sum(p.numel() for p in self.model.parameters())
        self.write_log(f"run={self.run_dir}")
        msg = f"device={self.device} | ddp={self.world_size} | params={n_params:,} | steps={self.steps} | batch={global_batch_size}"
        if self.distributed:
            msg += f" | local_batches={','.join(str(size) for size in rank_batch_sizes)}"
        self.write_log(msg)
        self.wandb_run = self.setup_wandb(n_params, rank_batch_sizes)

    def barrier(self):
        if self.distributed:
            dist.barrier()

    def setup_run(self):
        if self.config.get("continue_from"):
            run_dir = resolve_run_path(self.config["continue_from"])
            assert run_dir.exists(), f"continue run not found: {run_dir}"
            return run_dir

        RUNS_ROOT.mkdir(exist_ok=True)
        run_dir = RUNS_ROOT / self.config["run_name"]
        if self.is_main:
            if run_dir.exists():
                archive_dir = RUNS_ROOT / "archive"
                archive_dir.mkdir(exist_ok=True)
                stamped_name = f"{run_dir.name}_{datetime.now():%Y%m%d_%H%M%S}"
                shutil.move(run_dir, archive_dir / stamped_name)

            run_dir.mkdir()
            self.save_run_files(run_dir)
        self.barrier()
        return run_dir

    def save_run_files(self, run_dir):
        train_audio = Path(self.config["train_dir"]) / "audio_params.json"
        assert train_audio.exists(), f"audio_params.json not found: {train_audio}"
        shutil.copy2(train_audio, run_dir / "audio_params.json")
        if self.config.get("annotation_file"):
            shutil.copy2(self.config["annotation_file"], run_dir / "labels.json")

        self.write_json(run_dir / "config.json", self.config)
        self.write_json(run_dir / "model.json", self.config_slice(ModelConfig))
        self.write_json(run_dir / "train.json", self.config_slice(TrainConfig))

    def config_slice(self, cls):
        keys = [field.name for field in fields(cls) if field.init]
        data = {key: self.config[key] for key in keys if key in self.config}
        return asdict(cls(**data))

    def write_json(self, path, data):
        path.write_text(json.dumps(data, indent=2) + "\n")

    def load_checkpoint(self):
        checkpoints = sorted(self.weights_dir.glob("model_step_*.pth"))
        assert checkpoints, f"no checkpoints found in: {self.weights_dir}"
        checkpoint = checkpoints[-1]
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        step = int(checkpoint.stem.split("_step_")[1])
        self.write_log(f"loaded checkpoint: {checkpoint}")
        return step + 1

    def save_checkpoint(self, step):
        if not self.is_main:
            return
        path = self.weights_dir / f"model_step_{step:06d}.pth"
        model = self.model.module if self.distributed else self.model
        torch.save(model.state_dict(), path)

    def save_reconstruction(self, batch, step):
        pass

    def write_log(self, msg):
        if not self.is_main:
            return
        print(msg)
        with self.logs_path.open("a") as f:
            f.write(msg + "\n")

    def write_loss(self, step, split, loss):
        if not self.is_main:
            return
        with self.losses_path.open("a") as f:
            f.write(f"{step},{split},{loss:.8f}\n")
        self.write_wandb(step, split, loss)

    def setup_wandb(self, n_params, rank_batch_sizes):
        if not self.is_main or not self.config.get("wandb"):
            return None
        import wandb

        run = wandb.init(
            project="TinyBird",
            name=self.run_dir.name,
            dir=str(self.run_dir),
            config=self.config | {"params": n_params, "local_batches": rank_batch_sizes},
        )
        self.write_log(f"wandb={run.url}")
        return run

    def write_wandb(self, step, split, loss):
        if not self.wandb_run:
            return
        metrics = {f"{split}/loss": loss}
        if split == "train":
            metrics["train/lr"] = self.optimizer.param_groups[0]["lr"]
        self.wandb_run.log(metrics, step=step)

    def finish_wandb(self):
        if self.wandb_run:
            self.wandb_run.finish()

    def build_model(self):
        return SongMAE(self.config)

    def raw_model(self):
        return self.model.module if self.distributed else self.model

    def build_datasets(self):
        raise NotImplementedError

    def forward_loss(self, batch):
        raise NotImplementedError

    def batch_count(self, batch):
        assert isinstance(batch, (list, tuple))
        return int(batch[0].shape[0])

    def trim_batch(self, batch):
        if not self.distributed or self.batch_count(batch) <= self.batch_size:
            return batch
        trimmed = [field[:self.batch_size] for field in batch]
        if isinstance(batch, tuple):
            return tuple(trimmed)
        return trimmed

    def summarize_loss(self, loss, local_count):
        count = torch.tensor(float(local_count), device=self.device)
        value = torch.stack((loss.detach().float() * count, count))
        if self.distributed:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return (value[0] / value[1]).item(), value[1].item()

    def learning_rate(self, step):
        warmup = self.config.get("warmup_steps", 0)
        lr = self.config["lr"]
        min_lr = self.config.get("min_lr", 0.0)
        if warmup and step < warmup:
            return lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, self.end_step + 1 - warmup)
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))

    def set_learning_rate(self, step):
        lr = self.learning_rate(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self, batch, train=True, step=0):
        batch = self.trim_batch(batch)
        local_count = self.batch_count(batch)
        self.model.train() if train else self.model.eval()
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                loss = self.forward_loss(batch)
        mean_loss, global_count = self.summarize_loss(loss, local_count)
        if train:
            self.set_learning_rate(step)
            self.optimizer.zero_grad(set_to_none=True)
            loss_scale = self.world_size * local_count / global_count
            self.scaler.scale(loss * loss_scale).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return mean_loss

    def log(self, step, train_loss, val_loss=None):
        msg = f"step {step:>7}/{self.end_step} | train {train_loss:.4f}"
        if val_loss is not None:
            msg += f" | val {val_loss:.4f}"
        self.write_log(msg)

    def train(self):
        train_ds, val_ds = self.build_datasets()
        train_sampler = DistributedSampler(train_ds) if self.distributed else None
        val_sampler = DistributedSampler(val_ds, shuffle=False) if self.distributed else None
        kw = dict(batch_size=self.loader_batch_size, num_workers=self.config.get("num_workers", 4), pin_memory=True)
        train_loader = DataLoader(train_ds, shuffle=train_sampler is None, sampler=train_sampler, **kw)
        val_loader = DataLoader(val_ds, shuffle=False, sampler=val_sampler, **kw)
        train_iter, val_iter = iter(train_loader), iter(val_loader)

        for step in range(self.start_step, self.end_step + 1):
            if train_sampler and step % len(train_loader) == 0:
                train_sampler.set_epoch(step // len(train_loader))
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            train_loss = self.step(batch, train=True, step=step)
            self.write_loss(step, "train", train_loss)

            if step % self.eval_every == 0:
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_batch = next(val_iter)
                val_loss = self.step(val_batch, train=False)
                self.write_loss(step, "val", val_loss)
                self.log(step, train_loss, val_loss)
                if self.is_main:
                    self.save_reconstruction(val_batch, step)
            if step % self.save_every == 0:
                self.save_checkpoint(step)

        self.save_checkpoint(self.end_step)
        if self.is_main:
            self.write_log(f"saved loss curve: {save_loss_curve(self.losses_path, self.imgs_dir / 'loss_curve.png')}")
        self.finish_wandb()
        self.barrier()
        if self.distributed:
            dist.destroy_process_group()


class UnsupervisedTrainer(Trainer):
    def build_datasets(self):
        c = self.config
        return (SpectrogramDataset(c["train_dir"], c["num_timebins"]),
                SpectrogramDataset(c["val_dir"], c["num_timebins"]))

    def forward_loss(self, batch):
        x = batch[0].to(self.device, non_blocking=True)
        valid_timebins = batch[2].to(self.device, non_blocking=True)
        return self.model(x, valid_timebins=valid_timebins)

    def save_reconstruction(self, batch, step):
        sample_idx = (step // self.eval_every) % batch[0].shape[0]
        path = save_masked_reconstruction_plot(
            self.raw_model(),
            batch,
            self.config,
            self.device,
            self.use_amp,
            self.imgs_dir,
            step,
            sample_idx=sample_idx,
            amp_dtype=self.amp_dtype,
        )
        self.write_log(f"saved reconstruction: {path}")


class SupervisedTrainer(Trainer):
    def build_model(self):
        return SongMAE(self.config)  # TODO: supervised SongMAE head

    def build_datasets(self):
        c = self.config
        return (SpectrogramDatasetSupervised(c["train_dir"], c["annotation_file"], n_timebins=c["num_timebins"], recording_mode=c["recording_mode"]),
                SpectrogramDatasetSupervised(c["val_dir"], c["annotation_file"], n_timebins=c["num_timebins"], recording_mode=c["recording_mode"]))

    def forward_loss(self, batch):
        x = batch[0].to(self.device, non_blocking=True)
        labels = batch[1].to(self.device, non_blocking=True)
        logits = self.model(x)
        return self.model.compute_loss(logits, labels)


def resolve_run_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return RUNS_ROOT / path


def add_train_args(parser):
    parser.add_argument("--task", choices=["unsupervised", "supervised"])
    parser.add_argument("--train_dir")
    parser.add_argument("--val_dir")
    parser.add_argument("--run_name")
    parser.add_argument("--continue_from")
    parser.add_argument("--annotation_file")
    parser.add_argument("--recording_mode", choices=["events", "full_recordings"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--grad_clip", type=float)
    parser.add_argument("--eval_every", type=int)
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--min_lr", type=float)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--amp_dtype", choices=["bf16", "fp16"])
    parser.add_argument("--wandb", action="store_true", default=None)


def add_model_args(parser):
    parser.add_argument("--model_preset", choices=sorted(MODEL_PRESETS))
    parser.add_argument("--patch_height", type=int)
    parser.add_argument("--patch_width", type=int)
    parser.add_argument("--num_timebins", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--mask_p", type=float)
    parser.add_argument("--mask_c", type=float)
    parser.add_argument("--mask_type", choices=["voronoi", "random"])
    parser.add_argument("--enc_hidden_d", type=int)
    parser.add_argument("--enc_n_head", type=int)
    parser.add_argument("--enc_n_layer", type=int)
    parser.add_argument("--enc_dim_ff", type=int)
    parser.add_argument("--dec_hidden_d", type=int)
    parser.add_argument("--dec_n_head", type=int)
    parser.add_argument("--dec_n_layer", type=int)
    parser.add_argument("--dec_dim_ff", type=int)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SongMAE models.")
    add_train_args(parser)
    add_model_args(parser)
    return parser.parse_args()


def config_from_args(args):
    overrides = {key: value for key, value in vars(args).items() if value is not None}
    model_preset = overrides.pop("model_preset", None)
    run_dir = None
    if args.continue_from:
        run_dir = resolve_run_path(args.continue_from)
        config = json.loads((run_dir / "config.json").read_text())
    else:
        assert args.train_dir and args.val_dir and args.run_name
        config = dict(DEFAULT_CONFIG)

    if model_preset:
        config.update(MODEL_PRESETS[model_preset])
        config["model_preset"] = model_preset
    config.update(overrides)

    config.setdefault("task", "unsupervised")
    if config["task"] == "supervised":
        assert config.get("annotation_file"), "--annotation_file is required for supervised training"

    audio = AudioParams.from_dir(run_dir or config["train_dir"])
    config["mels"] = audio.mels
    config["patch_size"] = (config["patch_height"], config["patch_width"])
    model_keys = [field.name for field in fields(ModelConfig) if field.init]
    ModelConfig(**{key: config[key] for key in model_keys if key in config})
    return config


def trainer_from_config(config):
    if config["task"] == "unsupervised":
        return UnsupervisedTrainer(config)
    if config["task"] == "supervised":
        return SupervisedTrainer(config)
    raise ValueError(f"unknown training task: {config['task']}")


def main():
    trainer_from_config(config_from_args(parse_args())).train()


if __name__ == "__main__":
    main()
