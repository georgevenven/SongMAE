#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch, Rectangle
from scipy.ndimage import label
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_agentic_attention_head import AttentionHead
from scripts.qwen_box_tokens import BoxTokens, read_rows, split_rows
from src.core.data_structures import AudioParams
from src.core.utils import load_model_from_checkpoint, load_spec_slice


def main():
    parser = argparse.ArgumentParser(description="Plot contextual-head masks and boxes.")
    parser.add_argument("--probe", type=Path, default=Path("runs/song_unit_32x4/adaptive_attention_head.pt"))
    parser.add_argument("--annotations", type=Path, default=Path("data/XCL/qwen38_adaptive_review_5s_annotations.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("/home/george-vengrovski/Downloads/songmae_32x4_adaptive_attention_holdout"))
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args()
    device = torch.device("cuda")
    probe = torch.load(options.probe, map_location=device, weights_only=True)
    backbone, config = load_model_from_checkpoint(probe["backbone_run"], probe["checkpoint"])
    backbone.requires_grad_(False).eval().to(device)
    head = AttentionHead(config["enc_hidden_d"], probe["hidden"], probe["height"], probe["width"], probe["patch_size"][1],
        probe.get("position", True), probe.get("dropout", .1)).to(device)
    head.load_state_dict(probe["head"])
    head.eval()
    audio = AudioParams.from_dir(probe["backbone_run"])
    _, rows = split_rows(read_rows(options.annotations, None, options.seed), .1, options.seed)
    random.Random(options.seed).shuffle(rows)
    options.out.mkdir(parents=True, exist_ok=True)
    patch_height, patch_width = probe["patch_size"]
    token_seconds = patch_width * audio.hop_size / audio.sr

    for index, row in enumerate(rows[:options.n], 1):
        data = BoxTokens([row], audio, config["num_timebins"], patch_height)
        parts = []
        with torch.inference_mode():
            for specs, _, valid in DataLoader(data, 8):
                tokens, _ = backbone.forward_encoder_inference(specs.to(device), valid_timebins=valid.to(device))
                logits, _ = head(tokens.float(), valid)
                logits = logits.reshape(len(specs), probe["height"], probe["width"])
                for item, length in zip(logits, (valid + patch_width - 1) // patch_width):
                    parts.append(item[:, :length].cpu().numpy())
        logits = np.concatenate(parts, axis=1)
        probability = 1 / (1 + np.exp(-logits))
        components, count = label(logits >= probe["threshold"], structure=np.ones((3, 3)))
        predicted = []
        for component in range(1, count + 1):
            frequency, time = np.where(components == component)
            if len(time) < 2:
                continue
            predicted.append((time.min(), time.max() + 1, frequency.min(), frequency.max() + 1))

        source, tile = row["source"], row["tile"]
        raw = load_spec_slice(source["shard"], source["start"] + tile["start_timebin"], source["start"] + tile["end_timebin"])
        start, end = tile["onset_ms"] / 1000, tile["offset_ms"] / 1000
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, sharey=True)
        axes[0].imshow(raw, origin="lower", aspect="auto", extent=(start, end, 0, 128), cmap="magma")
        image = axes[1].imshow(probability, origin="lower", aspect="auto", extent=(start, end, 0, 128),
            cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        for axis in axes:
            for box in row["boxes"]:
                axis.add_patch(Rectangle((box["onset_ms"] / 1000, box["low_mel_bin"]),
                    (box["offset_ms"] - box["onset_ms"]) / 1000, box["high_mel_bin"] - box["low_mel_bin"],
                    fill=False, edgecolor="white", linestyle="--", linewidth=1.5))
            for left, right, low, high in predicted:
                axis.add_patch(Rectangle((start + left * token_seconds, low * patch_height),
                    (right - left) * token_seconds, (high - low) * patch_height,
                    fill=False, edgecolor="lime", linewidth=1.5))
        axes[0].legend(handles=[Patch(facecolor="none", edgecolor="white", linestyle="--", label="Qwen box"),
            Patch(facecolor="none", edgecolor="lime", label="Contextual-head box")], loc="upper right")
        axes[0].set(title=f"Held-out: {row['recording']} — {len(predicted)} predicted boxes", ylabel="Mel bin")
        axes[1].set(title="Contextual song probability", xlabel="Time (s)", ylabel="Mel bin")
        fig.colorbar(image, ax=axes[1], label="P(song)", pad=.01)
        fig.tight_layout()
        fig.savefig(options.out / f"holdout_{index:02d}_{row['recording']}.png", dpi=150)
        plt.close(fig)
    print(f"wrote {min(options.n, len(rows))} held-out contextual predictions to {options.out}")


if __name__ == "__main__":
    main()
