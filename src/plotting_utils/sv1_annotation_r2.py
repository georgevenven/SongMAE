from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_metric_bars(summary, out_path):
    rows = [summary["global"], *summary["by_dataset"]]
    names = [row["dataset"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    has_pixel = all("pixel_intensity" in row for row in rows)

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    if has_pixel:
        ax.bar(x - width / 2, [row["r2"] for row in rows], width, label="SV1", color="#4c78a8")
        ax.bar(x + width / 2, [row["pixel_intensity"]["r2"] for row in rows], width, label="Pixel intensity", color="#f58518")
        ax.legend(frameon=False)
    else:
        ax.bar(x, [row["r2"] for row in rows], color="#4c78a8")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel("R^2")
    ax.set_title("Song-state variance explained")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_sv1_overlay(items, out_path):
    items = items[:12]
    values = np.concatenate([item["score"] for item in items])
    vmin, vmax = np.percentile(values, [1, 99])
    if vmax <= vmin:
        vmax = vmin + 1.0

    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    grid = fig.add_gridspec(4, 3)
    fig.suptitle("SV1 centered dot-product projection overlay; yellow strip = song", fontsize=12)

    for i, item in enumerate(items):
        spec = item["spectrogram"]
        score = item["score"]
        song = item["song"]
        ax = fig.add_subplot(grid[i // 3, i % 3])
        spec_vmin, spec_vmax = np.percentile(spec, [1, 99.5])
        ax.imshow(spec, aspect="auto", origin="lower", cmap="magma", vmin=spec_vmin, vmax=spec_vmax)

        token_x = np.arange(score.size)
        scaled = np.clip((score - vmin) / (vmax - vmin), 0.0, 1.0)
        ax.plot(token_x, 8 + scaled * (spec.shape[0] - 16), color="#2dd4bf", linewidth=1.4)
        ax.fill_between(token_x, 0, 5, where=song > 0, color="#fef08a", alpha=0.85, step="mid")

        ax.set_title(item["name"], fontsize=8)
        ax.set_xticks([])
        if i % 3 == 0:
            ax.set_ylabel(f"{item['dataset']}\nmel")
        else:
            ax.set_yticks([])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
