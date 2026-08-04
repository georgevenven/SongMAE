#!/usr/bin/env python3
import argparse
import os
import json
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local deps
from src.core.data_loader import SpectrogramDataset, SpectrogramDatasetSupervised
from src.core.utils import load_model_from_checkpoint
from src.plotting_utils.plotting_utils import MASK_CMAP, masked_cmap


def depatchify(pred_patches, H, W, patch_size):
    # pred_patches: (B, T, P) → (B, 1, H, W)
    fold = nn.Fold(output_size=(H, W), kernel_size=patch_size, stride=patch_size)
    return fold(pred_patches.transpose(1, 2))


def masked_original(x_patches, bool_mask):
    # x_patches: (B, T, P), bool_mask: (B, T)
    masked = x_patches.clone()
    masked[bool_mask] = float("nan")
    return masked


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)


def filename_at(filenames, index):
    if isinstance(filenames, (list, tuple)):
        return filenames[index]
    return str(filenames)


def infer_valid_timebins(spectrograms, pad_value):
    # spectrograms: (B, 1, H, W)
    diff = (spectrograms[:, 0] - pad_value).abs().amax(dim=1)
    valid = []
    for row in diff > 1e-5:
        indices = torch.nonzero(row, as_tuple=False).flatten()
        valid.append(int(indices[-1].item()) + 1 if indices.numel() else row.numel())
    return torch.tensor(valid, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="Reconstruct spectrograms and compute MSE.")
    parser.add_argument("--run_dir", required=True, type=str, help="Run directory or name under ../runs")
    parser.add_argument("--spec_dir", required=True, type=str, help="Directory of spectrogram tensors (val-style)")
    parser.add_argument("--out_dir", required=True, type=str, help="Folder to store results")
    parser.add_argument("--num_samples", type=int, default=10000, help="Max samples to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint filename to load")
    parser.add_argument("--annotation_file", type=str, default=None, help="Optional annotation JSON for event crops")
    parser.add_argument("--recording_mode", type=str, default="events", choices=["events", "full_recordings"])
    parser.add_argument("--bird", type=str, default=None, help="Optional bird_id filter for annotations")
    parser.add_argument("--per_patch_norm", action="store_true", help="Enable per-patch normalization for visualization")
    parser.add_argument("--inference_mode", action="store_true", help="Disable masking (autoencoder-style reconstruction)")
    parser.add_argument("--numbers_only", action="store_true", help="Only compute CSV/summary metrics; skip image generation")
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible crops and masks")
    parser.add_argument(
        "--image_format",
        type=str,
        default="png",
        choices=["png", "pdf"],
        help="Output format for saved visualizations",
    )
    args = parser.parse_args()
    if args.batch_size != 1 and not args.numbers_only:
        raise SystemExit("--batch_size > 1 is only supported with --numbers_only")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model + config
    model, config = load_model_from_checkpoint(
        run_dir=args.run_dir,
        checkpoint_file=args.checkpoint,
        fallback_to_random=False
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Dataset and loader (val-style)
    if args.annotation_file:
        dataset = SpectrogramDatasetSupervised(
            dir=args.spec_dir,
            annotation_file=args.annotation_file,
            n_timebins=config["num_timebins"],
            recording_mode=args.recording_mode,
            selected_bird=args.bird,
        )
    else:
        dataset = SpectrogramDataset(
            dir=args.spec_dir,
            n_timebins=config["num_timebins"]
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Output dirs
    mse_root = os.path.join(args.out_dir, "MSE analysis")
    os.makedirs(mse_root, exist_ok=True)
    imgs_dir = os.path.join(mse_root, "imgs")
    if not args.numbers_only:
        os.makedirs(imgs_dir, exist_ok=True)

    # Save a copy of run config for traceability
    with open(os.path.join(mse_root, "eval_config.json"), "w") as f:
        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_dir": args.run_dir,
            "checkpoint": args.checkpoint,
            "spec_dir": args.spec_dir,
            "annotation_file": args.annotation_file,
            "recording_mode": args.recording_mode,
            "bird": args.bird,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": str(device),
            "numbers_only": args.numbers_only,
            "model_config": config
        }
        json.dump(meta, f, indent=2)

    patch_size = (int(config["patch_height"]), int(config["patch_width"]))
    H = int(dataset.params.mels)
    W = int(config["num_timebins"])
    pad_value = float((0.0 - dataset.mean) / dataset.std)

    unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    # Aggregators for true dataset-wide MSE (sum of squared errors / total elements)
    SSE_all = 0.0
    N_all = 0
    SSE_masked = 0.0
    N_masked = 0

    # Per-sample CSV
    csv_path = os.path.join(mse_root, "per_sample_mse.csv")
    with open(csv_path, "w") as fcsv:
        fcsv.write("index,filename,mse_all,mse_masked\n")

    pbar = tqdm(total=min(args.num_samples, len(dataset)), desc="Evaluating", unit="sample")
    evaluated = 0

    with torch.no_grad():
        for batch in loader:
            if evaluated >= args.num_samples:
                break

            remaining = args.num_samples - evaluated
            spectrograms = batch[0][:remaining]
            if torch.is_tensor(batch[1]):
                filenames = batch[2][:remaining]
                valid_timebins = infer_valid_timebins(spectrograms, pad_value)
            else:
                filenames = batch[1][:remaining]
                valid_timebins = batch[2][:remaining]
            x = spectrograms.to(device, non_blocking=True)
            valid_timebins = valid_timebins.to(device, non_blocking=True)
            batch_size = x.size(0)

            pred, bool_mask = model.reconstruct(x, valid_timebins=valid_timebins)

            # Prepare patches of target
            x_patches = unfold(x).transpose(1, 2)  # (1, T, P)

            # Disable per-patch denormalization so we visualise raw decoder outputs.
            # target_mean = x_patches.mean(dim=-1, keepdim=True)
            # target_std = x_patches.std(dim=-1, keepdim=True)
            # pred_denorm = pred * (target_std + 1e-6) + target_mean  # (1, T, P)
            pred_denorm = pred.to(dtype=x_patches.dtype)

            diff2 = (pred_denorm - x_patches) ** 2
            mse_all_values = diff2.flatten(1).mean(dim=1)
            masked_elems = bool_mask.sum(dim=1) * diff2.size(-1)
            mse_masked_values = [
                diff2[j][bool_mask[j]].mean().item() if masked_elems[j].item() else float("nan")
                for j in range(batch_size)
            ]

            # Global aggregates
            SSE_all += diff2.sum().item()
            N_all += diff2.numel()
            SSE_masked += diff2[bool_mask].sum().item()
            N_masked += masked_elems.sum().item()

            fname = sanitize(filename_at(filenames, 0))

            if not args.numbers_only:
                import matplotlib.pyplot as plt

                # Overlay image: original for unmasked, prediction for masked.
                # In inference_mode there is no mask, so show full reconstruction.
                if args.inference_mode:
                    overlay_patches = pred_denorm
                else:
                    overlay_patches = x_patches.clone()
                    overlay_patches[bool_mask] = pred_denorm[bool_mask]

                if args.per_patch_norm:
                    overlay_mean = overlay_patches.mean(dim=-1, keepdim=True)
                    overlay_std = overlay_patches.std(dim=-1, keepdim=True)
                    overlay_patches = (overlay_patches - overlay_mean) / (overlay_std + 1e-6)

                overlay_img = depatchify(overlay_patches, H=H, W=W, patch_size=patch_size)
                masked_patches = masked_original(x_patches, bool_mask)
                masked_img = depatchify(masked_patches, H=H, W=W, patch_size=patch_size)

                display_w = int(valid_timebins[0].item())
                x_img = x[0, 0, :, :display_w].detach().cpu().numpy()
                masked_img_np = masked_img[0, 0, :, :display_w].detach().cpu().numpy()
                overlay_np = overlay_img[0, 0, :, :display_w].detach().cpu().numpy()

                seconds = display_w * dataset.params.hop_size / dataset.params.sr
                extent = (0, seconds, 0, H)
                fig2, axes = plt.subplots(3, 1, figsize=(7.9, 5.8933), sharex=True)
                titles = (
                    "Input Spectrogram",
                    "Input Spectrogram With Mask",
                    "Decoder Output" if args.inference_mode else "Decoder Predictions and Original Spectrogram",
                )
                for ax, image, title, cmap in zip(
                    axes,
                    (x_img, masked_img_np, overlay_np),
                    titles,
                    (MASK_CMAP, masked_cmap(), MASK_CMAP),
                ):
                    ax.imshow(image, origin="lower", aspect="auto", cmap=cmap, extent=extent)
                    ax.set_title(title, fontsize=16, fontweight="bold")
                    ax.set_yticks([0, H // 2, H])
                    ax.set_ylabel("Mels", fontsize=12)
                    ax.tick_params(labelsize=10)
                axes[-1].set_xticks(np.arange(0, seconds + 1e-9, 1))
                axes[-1].set_xlabel("Time (s)", fontsize=12)

                fig2.tight_layout()
                out_image = os.path.join(imgs_dir, f"{evaluated:06d}_{fname}_overlay.{args.image_format}")
                save_kwargs = {"facecolor": "white", "edgecolor": "none"}
                if args.image_format == "png":
                    save_kwargs["dpi"] = 300
                fig2.savefig(out_image, **save_kwargs)
                plt.close(fig2)

            with open(csv_path, "a") as fcsv:
                for j in range(batch_size):
                    fname = sanitize(filename_at(filenames, j))
                    fcsv.write(
                        f"{evaluated + j},{fname},{mse_all_values[j].item():.8f},{mse_masked_values[j]:.8f}\n"
                    )

            evaluated += batch_size
            pbar.set_postfix(
                mse_all=f"{mse_all_values.mean().item():.5g}",
                mse_masked=f"{float(np.nanmean(mse_masked_values)):.5g}",
            )
            pbar.update(batch_size)

    pbar.close()

    # Final summary
    summary = {
        "evaluated_samples": evaluated,
        "pixels_per_patch": int(patch_size[0] * patch_size[1]),
        "SSE_all": SSE_all,
        "N_all": N_all,
        "MSE_all_dataset_mean": (SSE_all / N_all) if N_all > 0 else float("nan"),
        "SSE_masked": SSE_masked,
        "N_masked": N_masked,
        "MSE_masked_dataset_mean": (SSE_masked / N_masked) if N_masked > 0 else float("nan"),
    }
    with open(os.path.join(mse_root, "summary.json"), "w") as fsum:
        json.dump(summary, fsum, indent=2)

    print("Done.")
    print(f"Summary: {os.path.join(mse_root, 'summary.json')}")
    print(f"Per-sample CSV: {csv_path}")
    if not args.numbers_only:
        print(f"Images dir: {imgs_dir}")


if __name__ == "__main__":
    main()
