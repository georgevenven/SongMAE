#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


DEFAULT_SERIES = (
    ("32 x 4", Path("results/mse_v_steps_xcl_micro_every20"), "#1f77b4"),
    ("32 x 1", Path("results/mse_v_steps_xcl_micro_p32x1_every20"), "#d62728"),
    ("16 x 1", Path("results/mse_v_steps_xcl_micro_p16x1_every20"), "#2ca02c"),
)


def load_rows(root: Path, metric: str) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/step_*/MSE analysis/summary.json")):
        with path.open() as f:
            summary = json.load(f)
        species = path.parts[-4]
        step = int(path.parts[-3].replace("step_", ""))
        rows.append({"species": species, "step": step, "mse": float(summary[metric])})
    assert rows, f"No summary.json files found under {root}"
    return rows


def style_ax(ax, metric: str) -> None:
    ax.set_title("Mean Reconstruction MSE vs Step", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Training Step", fontsize=10, fontweight="bold")
    ax.set_ylabel("MSE", fontsize=10, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    ax.grid(True, alpha=0.22)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax.tick_params(axis="both", labelsize=10.5, width=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#404040")


def save_plot(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")


def aggregate_rows(rows: list[dict]) -> tuple[list[int], list[float]]:
    by_step = {}
    for row in rows:
        by_step.setdefault(row["step"], []).append(row["mse"])
    for values in by_step.values():
        assert len(values) == 3
    steps = [step for step in sorted(by_step) if step > 0]
    return steps, [sum(by_step[step]) / len(by_step[step]) for step in steps]


def step_ticks(steps: list[int]) -> list[int]:
    return sorted({steps[0], 100_000, steps[-1]})


def plot_series(ax, label: str, root: Path, metric: str, color: str) -> list[int]:
    steps, mses = aggregate_rows(load_rows(root, metric))
    ax.plot(
        steps,
        mses,
        label=label,
        marker="o",
        markersize=4.8,
        linewidth=2.2,
        color=color,
        alpha=0.95,
    )
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reconstruction MSE against checkpoint step.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional single MSE sweep output root.",
    )
    parser.add_argument(
        "--metric",
        default="MSE_masked_dataset_mean",
        choices=("MSE_masked_dataset_mean", "MSE_all_dataset_mean"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)
    series = (("MSE", args.root, "#1f77b4"),) if args.root else DEFAULT_SERIES
    all_steps = []
    for label, root, color in series:
        all_steps.extend(plot_series(ax, label, root, args.metric, color))

    style_ax(ax, args.metric)
    ax.set_xticks(step_ticks(all_steps))
    ax.legend(frameon=False, fontsize=10.5)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.13, top=0.92)

    output = args.output or Path("results/mse_v_steps_patch_size_compare") / f"{args.metric}_vs_steps.png"
    if args.root and not args.output:
        output = args.root / f"{args.metric}_vs_steps.png"
    save_plot(fig, output)
    plt.close(fig)
    print(f"Saved: {output}")
    print(f"Saved: {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
