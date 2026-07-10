"""Plot the labeling error imposed by coarse temporal resolution."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[3]
ANNOTATIONS = ROOT / "files/annotation jsons/bf_annotations.json"
SPEC_DIR = Path("/media/george-vengrovski/disk2/specs/legacy_specs_2ms/bf_64hop_32khz")
RECORDING = "1_bird2"
START_MS, STOP_MS = 3600, 6100
RESOLUTIONS_MS = (200, 20, 5)
OUT = ROOT / "imgs/token_coarseness_classification.png"
BLUE_OUT = ROOT / "imgs/token_coarseness_blue_units.png"

plt.rcParams.update(
    {"font.size": 16, "axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16}
)


def coarsen(labels, width):
    coarse = np.empty_like(labels)
    for start in range(0, len(labels), width):
        stop = min(start + width, len(labels))
        coarse[start:stop] = np.bincount(labels[start:stop]).argmax()
    return coarse


def plot_bar(ax, labels, label, cmap, vmax):
    ax.imshow(
        labels[None],
        aspect="auto",
        cmap=cmap,
        extent=(0, len(labels), 0, 1),
        interpolation="nearest",
        vmin=0,
        vmax=vmax,
    )
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center")


def render(spec, params, labels, cmap, error_index, out):
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(12, 8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [6, 0.8, 0.8, 0.8, 0.8]},
    )
    axes[0].imshow(
        spec,
        origin="lower",
        aspect="auto",
        extent=(0, STOP_MS - START_MS, 0, params["mels"]),
        interpolation="nearest",
        cmap="viridis",
    )
    axes[0].set_yticks(np.arange(0, params["mels"] + 1, 32))
    axes[0].set_ylabel("Mels")
    axes[0].set_xlabel("")

    plot_bar(axes[1], labels, "Human\nannotations", cmap, error_index)
    for ax, resolution in zip(axes[2:], RESOLUTIONS_MS):
        coarse = coarsen(labels, resolution)
        error = coarse != labels
        display = np.where(error, error_index, coarse)
        plot_bar(ax, display, f"{resolution} ms", cmap, error_index)

    axes[-1].set_xticks(np.arange(0, STOP_MS - START_MS + 1, 500))
    axes[-1].set_xlabel("Time (ms)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    records = json.loads(ANNOTATIONS.read_text())["recordings"]
    record = next(r for r in records if Path(r["recording"]["filename"]).stem == RECORDING)
    units = [unit for event in record["detected_events"] for unit in event["units"]]
    units = [u for u in units if u["offset_ms"] > START_MS and u["onset_ms"] < STOP_MS]
    classes = {class_id: i + 1 for i, class_id in enumerate(sorted({u["id"] for u in units}))}

    labels = np.zeros(STOP_MS - START_MS, dtype=int)
    for unit in units:
        start = max(START_MS, round(unit["onset_ms"])) - START_MS
        stop = min(STOP_MS, round(unit["offset_ms"])) - START_MS
        labels[start:stop] = classes[unit["id"]]

    params = json.loads((SPEC_DIR / "audio_params.json").read_text())
    spec = np.load(SPEC_DIR / f"{RECORDING}.npy")
    ms_per_bin = params["hop_size"] / params["sr"] * 1000
    spec = spec[:, round(START_MS / ms_per_bin) : round(STOP_MS / ms_per_bin)]

    type_colors = [plt.get_cmap("tab20")(i) for i in (0, 2, 4, 8, 10, 14, 16, 18, 1, 3)]
    error_index = len(classes) + 1
    render(
        spec,
        params,
        labels,
        ListedColormap(["black", *type_colors[: len(classes)], "tab:red"]),
        error_index,
        OUT,
    )
    render(
        spec,
        params,
        labels,
        ListedColormap(["black", *["tab:blue"] * len(classes), "tab:red"]),
        error_index,
        BLUE_OUT,
    )


if __name__ == "__main__":
    main()
