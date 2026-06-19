"""Masked reconstruction plots for SongMAE pretraining."""

import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn

from src.plotting_utils.plotting_utils import (
    masked_cmap,
    plot_spec,
    save_fig,
)

RECON_FIGSIZE = (10.0, 8.0)
LOSS_FIGSIZE = (8.0, 5.0)


def save_loss_curve(losses_path, output_path):
    with Path(losses_path).open() as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=LOSS_FIGSIZE)
    for split, color in (("train", "tab:blue"), ("val", "tab:orange")):
        points = [row for row in rows if row["split"] == split]
        ax.plot(
            [int(row["step"]) for row in points],
            [float(row["loss"]) for row in points],
            label=split,
            linewidth=1.5,
            color=color,
        )
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.25)
    return save_fig(fig, output_path)


def depatchify(patches, mels, timebins, patch_size):
    fold = nn.Fold(output_size=(mels, timebins), kernel_size=patch_size, stride=patch_size)
    return fold(patches.transpose(1, 2))


def patch_mask_to_image(mask, patch_size, spec_shape):
    patch_h, patch_w = patch_size
    spec_h, spec_w = spec_shape
    mask = mask.reshape(spec_h // patch_h, spec_w // patch_w)
    mask = np.repeat(mask, patch_h, axis=0)
    return np.repeat(mask, patch_w, axis=1).astype(bool)


def save_masked_reconstruction_plot(
    model,
    batch,
    config,
    device,
    use_amp,
    output_dir,
    step,
    sample_idx=0,
    amp_dtype=torch.bfloat16,
):
    x = batch[0].to(device, non_blocking=True)
    valid_timebins = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        pred, bool_mask = model.reconstruct(x, valid_timebins=valid_timebins)

    patch_size = config["patch_size"]
    spec_shape = (config["mels"], config["num_timebins"])
    patches = nn.Unfold(kernel_size=patch_size, stride=patch_size)(x).transpose(1, 2)
    overlay = patches.clone()
    bool_mask = bool_mask.reshape(bool_mask.size(0), -1)
    overlay[bool_mask] = pred.to(dtype=overlay.dtype)[bool_mask]
    overlay = depatchify(overlay, config["mels"], config["num_timebins"], patch_size)

    x_img = x[sample_idx, 0].detach().cpu().numpy()
    overlay_img = overlay[sample_idx, 0].detach().cpu().numpy()
    mask_img = patch_mask_to_image(
        bool_mask[sample_idx].detach().cpu().numpy(),
        patch_size,
        spec_shape,
    )
    masked_img = x_img.copy()
    masked_img[mask_img] = np.nan

    fig = plt.figure(figsize=RECON_FIGSIZE)
    grid = fig.add_gridspec(3, 1, hspace=0.6)
    axes = [fig.add_subplot(grid[i, 0]) for i in range(3)]
    images = [
        (x_img, None, "Input Spectrogram"),
        (masked_img, masked_cmap(), "Masked Input"),
        (overlay_img, None, "Reconstruction Overlay"),
    ]
    for ax, (image, cmap, title) in zip(axes, images):
        plot_spec(ax, image, spec_shape, title, cmap=cmap)

    return save_fig(fig, Path(output_dir) / f"recon_step_{step:06d}.png")
