import json
from pathlib import Path

import numpy as np

from src.core.embedding_store import EmbeddingStore


def load_plot_specs(path):
    data = EmbeddingStore(path)
    keys = ("segment_recording_stem", "segment_song_id", "segment_spec_path")
    if any(key not in data for key in keys):
        return {}
    return {
        f"{stem}:{song}": Path(str(spec_path))
        for stem, song, spec_path in zip(data["segment_recording_stem"], data["segment_song_id"], data["segment_spec_path"])
    }


def ground_truth(units, stem, start, end):
    y = np.zeros(max(0, end - start), dtype=np.int64)
    for onset, offset, label in units.get(stem, []):
        lo = max(start, onset)
        hi = min(end, offset)
        if lo < hi:
            y[lo - start : hi - start] = label
    return y


def spec_ms_per_bin(spec_path):
    params = json.loads((Path(spec_path).parent / "audio_params.json").read_text())
    return 1000.0 * float(params["hop_size"]) / float(params["sr"])


def load_spec_crop(spec_path, start, end):
    spec = np.load(spec_path)
    if spec.shape[0] == 128:
        spec = spec.T
    ms_per_bin = spec_ms_per_bin(spec_path)
    lo = max(0, int(np.floor(start / ms_per_bin)))
    hi = min(spec.shape[0], int(np.ceil(end / ms_per_bin)))
    return spec[lo:hi].T, lo * ms_per_bin, hi * ms_per_bin


def save_spectrogram_prediction_vs_groundtruth(predictions, spans, groups, units, out_dir, plot_specs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_group = {}
    for pred, span, group in zip(predictions, spans, groups):
        by_group.setdefault(group, []).append((pred, span))

    for index, (group, rows) in enumerate(sorted(by_group.items())[:50]):
        start = min(span[1] for _, span in rows)
        end = max(span[2] for _, span in rows)
        true = ground_truth(units, rows[0][1][0], start, end)
        pred = np.zeros(end - start, dtype=np.int64)
        for label, (_, lo, hi) in rows:
            pred[lo - start : hi - start] = label

        vmax = max(1, int(max(true.max(initial=0), pred.max(initial=0))))
        if group in plot_specs:
            spec, spec_start, spec_end = load_spec_crop(plot_specs[group], start, end)
            fig, axes = plt.subplots(
                3,
                1,
                figsize=(10, 3.2),
                dpi=160,
                sharex=True,
                gridspec_kw={"height_ratios": [4, 0.55, 0.55]},
            )
            axes[0].imshow(
                spec,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=[spec_start, spec_end, 0, spec.shape[0]],
                cmap="viridis",
                vmin=np.percentile(spec, 2),
                vmax=np.percentile(spec, 99),
            )
            axes[0].set_yticks([])
            axes[0].set_ylabel("spec")
            axes[0].set_title(group, fontsize=8)
            label_axes = axes[1:]
        else:
            fig, label_axes = plt.subplots(2, 1, figsize=(8, 1.6), dpi=160, sharex=True)

        for ax, row, title in zip(label_axes, [true, pred], ["truth", "prediction"]):
            ax.imshow(
                row[None, :],
                aspect="auto",
                interpolation="nearest",
                vmin=0,
                vmax=vmax,
                extent=[start, end, 0, 1],
            )
            ax.set_yticks([])
            ax.set_ylabel(title, rotation=45, ha="right", va="center", labelpad=18)
        label_axes[-1].set_xlabel("ms")
        label_axes[-1].set_xlim(start, end)
        fig.tight_layout()
        fig.savefig(out_dir / f"{index:04d}_{group.replace(':', '_')}.png")
        plt.close(fig)
