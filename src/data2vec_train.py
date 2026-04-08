import argparse
import glob
import json
import math
import os
import shutil
import time
from datetime import datetime

# Set matplotlib backend BEFORE importing plotting_utils
import matplotlib

matplotlib.use("Agg")

import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from data_loader import SpectogramDataset
from model import TinyBird
from plotting_utils import plot_loss_curves, save_data2vec_latent_plot
from utils import (
    count_parameters,
    load_audio_params,
    load_model_from_checkpoint,
    load_training_state,
)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")


def resolve_run_path(path_fragment):
    if os.path.isabs(path_fragment):
        return path_fragment

    project_relative = os.path.abspath(os.path.join(PROJECT_ROOT, path_fragment))
    if os.path.exists(project_relative):
        return project_relative

    return os.path.abspath(os.path.join(RUNS_ROOT, path_fragment))


class Data2VecTinyBird(nn.Module):
    def __init__(self, config, pretrained_model=None):
        super().__init__()

        self.config = config
        self.teacher_encoder_layer_idx = config.get("teacher_encoder_layer_idx", None)
        teacher_top_k = config.get("teacher_top_k")
        if teacher_top_k is None:
            teacher_top_k = config["enc_n_layer"]
        self.teacher_top_k = int(teacher_top_k)
        self.teacher_target_feature = config.get("teacher_target_feature", "ffn")
        self.loss_type = config.get("loss_type", "mse")
        self.target_layer_norm = bool(config.get("target_layer_norm", True))

        if pretrained_model is None:
            self.student = TinyBird(config)
        else:
            self.student = pretrained_model

        self.teacher = TinyBird(config)
        self.teacher.load_state_dict(self.student.state_dict())
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.predictor_input_proj = nn.Linear(config["dec_hidden_d"], config["enc_hidden_d"])
        self.predictor = nn.TransformerEncoderLayer(
            d_model=config["enc_hidden_d"],
            nhead=config["enc_n_head"],
            dim_feedforward=config["enc_dim_ff"],
            dropout=config["dropout"],
            batch_first=True,
            norm_first=True,
        )

    def _decode_student_latents(self, h, idx_restore, T):
        B = h.size(0)
        y = self.student.encoder_to_decoder(h)
        D_dec = self.student.decoder_to_pixel.in_features
        keep = y.size(1)

        mask_tokens = self.student.mask_token.expand(B, T - keep, D_dec)
        y_full = torch.cat([y, mask_tokens], dim=1)
        y_full = torch.gather(y_full, 1, idx_restore.unsqueeze(-1).expand(B, T, D_dec))

        pos_enc_seq = self.student.pos_enc.flatten(2, 3).transpose(1, 2)[:, :T, :]
        pos_dec = self.student.encoder_to_decoder(pos_enc_seq)
        y_full = y_full + pos_dec

        return self.student.decoder(y_full)

    def forward(self, x):
        h_student, idx_restore, bool_mask, T = self.student.forward_encoder(x)
        decoded = self._decode_student_latents(h_student, idx_restore, T)
        pred = self.predictor(self.predictor_input_proj(decoded))

        with torch.no_grad():
            self.teacher.eval()
            if self.teacher_encoder_layer_idx is None:
                target, _ = self.teacher.forward_encoder_inference(
                    x,
                    average_top_k=self.teacher_top_k,
                    target_feature_type=self.teacher_target_feature,
                )
            else:
                target, _ = self.teacher.forward_encoder_inference(
                    x,
                    encoder_layer_idx=self.teacher_encoder_layer_idx,
                    target_feature_type=self.teacher_target_feature,
                )

        return pred, target.detach(), bool_mask

    def compute_loss(self, pred, target, bool_mask):
        masked_pred = pred[bool_mask]
        masked_target = target[bool_mask]
        assert masked_pred.numel() > 0

        if self.target_layer_norm:
            masked_pred = F.layer_norm(masked_pred, (masked_pred.shape[-1],))
            masked_target = F.layer_norm(masked_target, (masked_target.shape[-1],))

        if self.loss_type == "mse":
            return F.mse_loss(masked_pred, masked_target)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(masked_pred, masked_target)
        if self.loss_type == "cosine":
            sim = F.cosine_similarity(masked_pred.float(), masked_target.float(), dim=-1)
            return 1.0 - sim.mean()
        raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    @staticmethod
    def activation_stats(pred, target, bool_mask):
        masked_pred = pred[bool_mask].float()
        masked_target = target[bool_mask].float()
        assert masked_pred.numel() > 0
        assert masked_target.numel() > 0
        return {
            "pred_activation_std": float(masked_pred.std(dim=0, unbiased=False).mean().item()),
            "target_activation_std": float(masked_target.std(dim=0, unbiased=False).mean().item()),
            "pred_activation_rms": float(masked_pred.pow(2).mean().sqrt().item()),
            "target_activation_rms": float(masked_target.pow(2).mean().sqrt().item()),
        }

    @torch.no_grad()
    def update_teacher(self, ema_decay):
        ema_decay = float(ema_decay)
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data.mul_(ema_decay).add_(student_param.data, alpha=1.0 - ema_decay)


def load_data2vec_model_from_checkpoint(run_dir="", checkpoint_file=None):
    if not run_dir:
        raise ValueError("run_dir cannot be empty")

    run_path = resolve_run_path(run_dir)
    if not os.path.exists(run_path):
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    config_path = os.path.join(run_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    model = Data2VecTinyBird(config)
    weights_dir = os.path.join(run_path, "weights")
    if not os.path.exists(weights_dir):
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    if checkpoint_file is not None:
        if os.path.isabs(checkpoint_file):
            checkpoint_path = checkpoint_file
        else:
            checkpoint_path = os.path.join(weights_dir, checkpoint_file)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Specified checkpoint file not found: {checkpoint_path}")
    else:
        checkpoint_pattern = os.path.join(weights_dir, "trainer_step_*.pth")
        checkpoint_files = glob.glob(checkpoint_pattern)
        if not checkpoint_files:
            raise FileNotFoundError(
                f"No trainer checkpoint files found in: {weights_dir}. "
                "Data2Vec resume expects trainer_step_*.pth files."
            )
        checkpoint_path = max(
            checkpoint_files,
            key=lambda x: int(x.split("_step_")[1].split(".pth")[0]),
        )

    print(f"Loading data2vec checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    try:
        step_num = int(checkpoint_path.split("_step_")[1].split(".pth")[0])
        print(f"Model loaded from step {step_num}")
    except (IndexError, ValueError):
        print(f"Model loaded from: {os.path.basename(checkpoint_path)}")

    return model, config


class Trainer:
    def __init__(self, config, model=None, pretrained_student=None):
        self.config = config

        if config.get("is_continuing", False):
            continue_from = config["continue_from"]
            self.run_path = resolve_run_path(continue_from)
            if not os.path.exists(self.run_path):
                raise FileNotFoundError(f"Continue directory not found: {self.run_path}")
            print(f"Continuing training from: {self.run_path}")
        else:
            os.makedirs(RUNS_ROOT, exist_ok=True)

            self.run_path = os.path.join(RUNS_ROOT, config["run_name"])
            if os.path.exists(self.run_path):
                archive_dir = os.path.join(RUNS_ROOT, "archive")
                os.makedirs(archive_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archived_path = os.path.join(archive_dir, f"{config['run_name']}_{timestamp}")
                shutil.move(self.run_path, archived_path)
                print(f"Moved existing run directory to: {archived_path}")

            os.makedirs(self.run_path, exist_ok=True)

            audio_params_src = os.path.join(config.get("train_dir", ""), "audio_params.json")
            if os.path.isfile(audio_params_src):
                shutil.copy2(audio_params_src, os.path.join(self.run_path, "audio_params.json"))
            else:
                print(f"Warning: audio_params.json not found in train_dir: {audio_params_src}")

        self.weights_path = os.path.join(self.run_path, "weights")
        self.imgs_path = os.path.join(self.run_path, "imgs")
        os.makedirs(self.weights_path, exist_ok=True)
        os.makedirs(self.imgs_path, exist_ok=True)

        if not config.get("is_continuing", False):
            config_path = os.path.join(self.run_path, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.use_wandb = config.get("wandb", False)
        if self.use_wandb:
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "tinybird"),
                name=config.get("run_name"),
                config=config,
            )

        if model is not None:
            self.model = model.to(self.device)
            print("Using loaded data2vec model from checkpoint")
        else:
            self.model = Data2VecTinyBird(config, pretrained_model=pretrained_student).to(self.device)
            if pretrained_student is None:
                print("Initialized new data2vec model")
            else:
                print("Initialized data2vec model from pretrained TinyBird weights")

        count_parameters(self.model.student)
        predictor_params = sum(p.numel() for p in self.model.predictor.parameters())
        predictor_params += sum(p.numel() for p in self.model.predictor_input_proj.parameters())
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Predictor parameters:  {predictor_params:,}")
        print(f"Total wrapper params:  {total_params:,}")
        print(f"Trainable parameters:  {trainable_params:,}")

        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        warmup_steps = int(config.get("warmup_steps", 1000))
        min_lr = float(config.get("min_lr", 1e-6))
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0. Got {warmup_steps}")
        if min_lr < 0:
            raise ValueError(f"min_lr must be >= 0. Got {min_lr}")

        if warmup_steps > 0 or min_lr > 0.0:
            total_steps = int(config["steps"])
            base_lr = float(config["lr"])
            decay_steps = max(1, total_steps - warmup_steps)

            def lr_lambda(step_idx):
                step_num = step_idx + 1
                if warmup_steps > 0 and step_num <= warmup_steps:
                    return step_num / float(warmup_steps)
                decay_step = step_num - warmup_steps
                decay_step = min(max(decay_step, 0), decay_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * decay_step / float(decay_steps)))
                target_lr = min_lr + (base_lr - min_lr) * cosine
                return target_lr / base_lr if base_lr > 0 else 1.0

            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config["steps"])

        self.use_amp = config.get("amp", False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.train_loss_history = []
        self.val_loss_history = []
        self.train_steps = []
        self.val_steps = []

        self.starting_step = 0
        if config.get("is_continuing", False):
            training_state = load_training_state(self.run_path, config.get("eval_every", 500))
            self.starting_step = training_state["starting_step"]
            self.train_steps = list(training_state.get("steps", []))
            self.train_loss_history = list(training_state.get("train_losses", []))
            self.val_steps = list(training_state.get("val_steps", []))
            self.val_loss_history = list(training_state.get("val_losses", []))

            if training_state["found_state"]:
                for _ in range(self.starting_step):
                    self.scheduler.step()

        self.loss_log_path = os.path.join(self.run_path, "loss_log.txt")
        if not config.get("is_continuing", False):
            with open(self.loss_log_path, "w") as f:
                f.write(
                    "step,train_loss,val_loss,gnorm,"
                    "pred_activation_std,target_activation_std,"
                    "pred_activation_rms,target_activation_rms,"
                    "samples_processed,steps_per_sec,samples_per_sec\n"
                )
        elif not os.path.exists(self.loss_log_path):
            print(f"Warning: Loss log not found at {self.loss_log_path}, starting fresh")
            with open(self.loss_log_path, "w") as f:
                f.write(
                    "step,train_loss,val_loss,gnorm,"
                    "pred_activation_std,target_activation_std,"
                    "pred_activation_rms,target_activation_rms,"
                    "samples_processed,steps_per_sec,samples_per_sec\n"
                )

    def step(self, batch, is_training=True):
        spectrograms, _ = batch
        x = spectrograms.to(self.device, non_blocking=True)

        if is_training:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.model.eval()

        with torch.set_grad_enabled(is_training):
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pred, target, bool_mask = self.model(x)
                    loss = self.model.compute_loss(pred, target, bool_mask)
            else:
                pred, target, bool_mask = self.model(x)
                loss = self.model.compute_loss(pred, target, bool_mask)
            stats = self.model.activation_stats(pred, target, bool_mask)

        gnorm = None
        if is_training:
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf"))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf"))
                self.optimizer.step()

            self.model.update_teacher(self.config["ema_decay"])
            self.scheduler.step()

        return loss.item(), gnorm.item() if gnorm is not None else None, stats

    def save_checkpoint(self, step_num):
        student_path = os.path.join(self.weights_path, f"model_step_{step_num:06d}.pth")
        trainer_path = os.path.join(self.weights_path, f"trainer_step_{step_num:06d}.pth")
        torch.save(self.model.student.state_dict(), student_path)
        torch.save(self.model.state_dict(), trainer_path)

    def save_validation_plot(self, batch, step_num):
        save_data2vec_latent_plot(
            self.model,
            batch,
            config=self.config,
            device=self.device,
            use_amp=self.use_amp,
            output_dir=self.imgs_path,
            step_num=step_num,
        )

    def train(self):
        from torch.utils.data import DataLoader

        train_dataset = SpectogramDataset(
            dir=self.config["train_dir"],
            n_timebins=self.config["num_timebins"],
            normalization=self.config["input_normalization"],
            output_dtype=self.config["input_dtype"],
        )
        val_dataset = SpectogramDataset(
            dir=self.config["val_dir"],
            n_timebins=self.config["num_timebins"],
            normalization=self.config["input_normalization"],
            output_dtype=self.config["input_dtype"],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )

        train_iter = iter(train_loader)
        val_iter = iter(val_loader)

        total_steps = self.config["steps"]
        end_step = self.starting_step + total_steps

        last_eval_time = time.time()
        last_eval_step = self.starting_step

        print(f"Training from step {self.starting_step} to {end_step}")

        for step_num in range(self.starting_step, end_step):
            try:
                train_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                train_batch = next(train_iter)

            train_loss, gnorm, train_stats = self.step(train_batch, is_training=True)
            self.train_loss_history.append(train_loss)
            self.train_steps.append(step_num)

            samples_processed = self.config["batch_size"] * (step_num + 1)

            if step_num % self.config["eval_every"] == 0:
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_batch = next(val_iter)

                val_loss, _, val_stats = self.step(val_batch, is_training=False)
                self.val_loss_history.append(val_loss)
                self.val_steps.append(step_num)

                progress_pct = ((step_num - self.starting_step + 1) / total_steps) * 100
                current_time = time.time()
                elapsed_time = current_time - last_eval_time
                steps_since_last_eval = step_num - last_eval_step
                steps_per_sec = steps_since_last_eval / elapsed_time if elapsed_time > 0 else 0
                samples_per_sec = steps_per_sec * self.config["batch_size"]

                last_eval_time = current_time
                last_eval_step = step_num

                current_lr = self.scheduler.get_last_lr()[0]
                print(
                    f"Step {step_num} ({progress_pct:.1f}%): Train Loss = {train_loss:.6f}, "
                    f"Val Loss = {val_loss:.6f}, "
                    f"Gnorm = {gnorm:.6f}, "
                    f"Pred Std = {train_stats['pred_activation_std']:.4f}, "
                    f"Teacher Std = {train_stats['target_activation_std']:.4f}, "
                    f"Samples = {samples_processed}, "
                    f"LR = {current_lr:.2e}, "
                    f"EMA = {self.config['ema_decay']:.5f}, "
                    f"Steps/sec = {steps_per_sec:.2f}, "
                    f"Samples/sec = {samples_per_sec:.1f}"
                )

                if self.use_wandb:
                    wandb.log(
                        {
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "gnorm": gnorm,
                            "samples_processed": samples_processed,
                            "steps_per_sec": steps_per_sec,
                            "samples_per_sec": samples_per_sec,
                            "lr": current_lr,
                            "ema_decay": self.config["ema_decay"],
                            "train_pred_activation_std": train_stats["pred_activation_std"],
                            "train_target_activation_std": train_stats["target_activation_std"],
                            "train_pred_activation_rms": train_stats["pred_activation_rms"],
                            "train_target_activation_rms": train_stats["target_activation_rms"],
                            "val_pred_activation_std": val_stats["pred_activation_std"],
                            "val_target_activation_std": val_stats["target_activation_std"],
                            "val_pred_activation_rms": val_stats["pred_activation_rms"],
                            "val_target_activation_rms": val_stats["target_activation_rms"],
                        },
                        step=step_num,
                    )

                self.save_validation_plot(val_batch, step_num)
                self.save_checkpoint(step_num)

                val_loss_str = f"{val_loss:.6f}"
                steps_per_sec_str = f"{steps_per_sec:.2f}"
                samples_per_sec_str = f"{samples_per_sec:.1f}"
            else:
                val_loss_str = ""
                steps_per_sec_str = ""
                samples_per_sec_str = ""

            with open(self.loss_log_path, "a") as f:
                f.write(
                    f"{step_num},{train_loss:.6f},{val_loss_str},"
                    f"{gnorm:.6f},"
                    f"{train_stats['pred_activation_std']:.6f},{train_stats['target_activation_std']:.6f},"
                    f"{train_stats['pred_activation_rms']:.6f},{train_stats['target_activation_rms']:.6f},"
                    f"{samples_processed},{steps_per_sec_str},{samples_per_sec_str}\n"
                )

        final_step = self.starting_step + self.config["steps"] - 1
        self.save_checkpoint(final_step)
        self.end_of_train_viz()
        if self.use_wandb:
            wandb.finish()

    def end_of_train_viz(self):
        plot_path = os.path.join(self.imgs_path, "loss_plot.png")
        plot_loss_curves(
            train_steps=self.train_steps,
            train_losses=self.train_loss_history,
            val_steps=self.val_steps,
            val_losses=self.val_loss_history,
            loss_log_path=self.loss_log_path,
            output_path=plot_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data2vec-style pretrain args")

    parser.add_argument("--train_dir", type=str, help="training directory")
    parser.add_argument("--val_dir", type=str, help="validation directory")
    parser.add_argument("--run_name", type=str, help="directory name inside /runs to store train run details")

    parser.add_argument("--steps", type=int, default=500_000, help="number of training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=48, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="number of DataLoader worker processes")

    parser.add_argument("--patch_height", type=int, default=32, help="patch height")
    parser.add_argument("--patch_width", type=int, default=1, help="patch width")
    parser.add_argument("--num_timebins", type=int, default=1024, help="number of time bins")

    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--mask_p", type=float, default=0.75, help="mask probability")
    parser.add_argument("--mask_c", type=float, default=0.1, help="seed probability for Voronoi mask")
    parser.add_argument(
        "--mask_type",
        type=str,
        default="voronoi",
        choices=["voronoi", "random"],
        help="masking strategy",
    )
    parser.add_argument("--eval_every", type=int, default=500, help="evaluate every N steps")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="linear warmup steps before decay")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="minimum learning rate for cosine decay")
    parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision training")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay")
    parser.add_argument(
        "--input_normalization",
        choices=["none", "audio_params", "per_file_zscore"],
        default=None,
        help="normalize each spectrogram using dataset audio_params stats or per-file z-score",
    )
    parser.add_argument("--input_dtype", choices=["float32", "float16", "bfloat16"], default="float32", help="dtype to emit from the dataloader")
    parser.add_argument("--continue_from", type=str, help="continue training from existing data2vec run directory")
    parser.add_argument("--init_from_pretrained_run", type=str, help="initialize student/teacher from a TinyBird pretrain run")
    parser.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")

    parser.add_argument("--ema_decay", type=float, default=0.99, help="EMA decay used to update the teacher")
    parser.add_argument(
        "--teacher_encoder_layer_idx",
        type=int,
        default=None,
        help="teacher encoder layer index to use as target; default averages top teacher layers instead",
    )
    parser.add_argument(
        "--teacher_top_k",
        type=int,
        default=None,
        help="average the top K teacher layers; default uses all encoder layers",
    )
    parser.add_argument(
        "--teacher_target_feature",
        type=str,
        default="ffn",
        choices=["ffn", "end_of_block"],
        help="teacher feature type to regress; data2vec paper favors FFN outputs",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["mse", "smooth_l1", "cosine"],
        help="latent prediction loss",
    )
    parser.add_argument(
        "--no_target_layer_norm",
        action="store_false",
        dest="target_layer_norm",
        help="disable per-token layer norm before latent loss computation",
    )

    parser.add_argument("--enc_hidden_d", type=int, default=384, help="encoder hidden dimension")
    parser.add_argument("--enc_n_head", type=int, default=6, help="encoder number of attention heads")
    parser.add_argument("--enc_n_layer", type=int, default=6, help="encoder number of transformer layers")
    parser.add_argument("--enc_dim_ff", type=int, default=1536, help="encoder feed-forward dimension")

    parser.add_argument("--dec_hidden_d", type=int, default=192, help="decoder hidden dimension")
    parser.add_argument("--dec_n_head", type=int, default=6, help="decoder number of attention heads")
    parser.add_argument("--dec_n_layer", type=int, default=2, help="decoder number of transformer layers")
    parser.add_argument("--dec_dim_ff", type=int, default=768, help="decoder feed-forward dimension")

    args = parser.parse_args()

    pretrained_student = None
    loaded_model = None

    if args.continue_from:
        resolved_continue = resolve_run_path(args.continue_from)
        loaded_model, config = load_data2vec_model_from_checkpoint(resolved_continue)
        config["continue_from"] = resolved_continue
        config["is_continuing"] = True
        config["wandb"] = args.wandb
        config.setdefault("mask_c", args.mask_c)
        config.setdefault("mask_type", args.mask_type)
        config.setdefault("input_normalization", "audio_params")
        config.setdefault("input_dtype", "float32")
        config.setdefault("ema_decay", args.ema_decay)
        config.setdefault("loss_type", args.loss_type)
        config.setdefault("teacher_top_k", args.teacher_top_k)
        config.setdefault("teacher_target_feature", args.teacher_target_feature)
        config.setdefault("target_layer_norm", True)
        if args.input_normalization is not None:
            config["input_normalization"] = args.input_normalization
        config["input_dtype"] = args.input_dtype
    else:
        if not args.train_dir or not args.val_dir or not args.run_name:
            parser.error("--train_dir, --val_dir, and --run_name are required when not using --continue_from")

        config = vars(args)
        config["is_continuing"] = False
        if config["input_normalization"] is None:
            config["input_normalization"] = "audio_params"

        audio_params = load_audio_params(config["train_dir"])
        config["mels"] = audio_params["mels"]

        if args.init_from_pretrained_run:
            pretrained_path = resolve_run_path(args.init_from_pretrained_run)
            pretrained_student, pretrained_config = load_model_from_checkpoint(
                pretrained_path,
                fallback_to_random=False,
            )
            print(f"Loaded pretrained TinyBird model from: {pretrained_path}")
            if int(config["mels"]) != int(pretrained_config["mels"]):
                raise ValueError(
                    f"Dataset mels ({config['mels']}) must match pretrained run mels ({pretrained_config['mels']})"
                )

            arch_keys = [
                "patch_height",
                "patch_width",
                "num_timebins",
                "enc_hidden_d",
                "enc_n_head",
                "enc_n_layer",
                "enc_dim_ff",
                "dec_hidden_d",
                "dec_n_head",
                "dec_n_layer",
                "dec_dim_ff",
                "dropout",
            ]
            for key in arch_keys:
                config[key] = pretrained_config[key]
            config["init_from_pretrained_run"] = pretrained_path

    config["patch_size"] = (config["patch_height"], config["patch_width"])
    config["max_seq"] = (
        (config["num_timebins"] // config["patch_width"])
        * (config["mels"] // config["patch_height"])
    )

    assert config["num_timebins"] % config["patch_width"] == 0, (
        f"num_timebins ({config['num_timebins']}) must be divisible by patch_width "
        f"({config['patch_width']})"
    )
    assert config["mels"] % config["patch_height"] == 0, (
        f"mels ({config['mels']}) must be divisible by patch_height "
        f"({config['patch_height']})"
    )

    trainer = Trainer(
        config,
        model=loaded_model,
        pretrained_student=pretrained_student,
    )
    trainer.train()
