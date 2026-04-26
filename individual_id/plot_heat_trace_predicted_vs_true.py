#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "results" / "individual_id_knn_graph_metrics" / "all_species_safe_heat"
SPECIES_KEYS = ["zf", "bf", "canary", "chiffchaff", "european_starling", "tree_pipit", "little_owl", "orangutan"]


def _read_rows(base_dir, prediction_column):
    rows = []
    for species_key in SPECIES_KEYS:
        csv_path = base_dir / species_key / "heat_trace_calibration.csv"
        summary_path = base_dir / species_key / "heat_trace_summary.json"
        if not csv_path.exists():
            continue

        summary = json.loads(summary_path.read_text())
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "species_key": species_key,
                        "species": summary["species"],
                        "best_heat_feature": summary["best_heat_feature"],
                        "true_individuals": float(row["individuals"]),
                        "predicted_individuals": float(row[prediction_column]),
                    }
                )
    assert rows
    return rows


def _scores(rows):
    y = np.asarray([row["true_individuals"] for row in rows], dtype=np.float32)
    pred = np.asarray([row["predicted_individuals"] for row in rows], dtype=np.float32)
    r2 = 1.0 - float(np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    mae = float(np.mean(np.abs(y - pred)))
    return r2, mae


def _write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path, rows, title):
    r2, mae = _scores(rows)
    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=300)
    colors = plt.get_cmap("tab10")
    max_value = max(max(row["true_individuals"], row["predicted_individuals"]) for row in rows)
    lims = (0.0, max_value + 1.5)

    for index, species_key in enumerate(SPECIES_KEYS):
        subset = [row for row in rows if row["species_key"] == species_key]
        if not subset:
            continue
        x = [row["true_individuals"] for row in subset]
        y = [row["predicted_individuals"] for row in subset]
        ax.scatter(x, y, s=38, alpha=0.78, color=colors(index % 10), label=species_key)

    ax.plot(lims, lims, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True individuals in subset")
    ax.set_ylabel("Predicted individuals")
    ax.set_title(title)
    ax.text(0.04, 0.96, f"R^2={r2:.2f}\nMAE={mae:.2f}", transform=ax.transAxes, va="top")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot heat-trace predicted vs true individual count across species.")
    parser.add_argument("--base_dir", default=str(DEFAULT_BASE))
    parser.add_argument("--prediction", default="loco", choices=["loco", "fit"])
    parser.add_argument("--out_prefix", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    prediction_column = "loco_predicted_individuals" if args.prediction == "loco" else "predicted_individuals"
    out_prefix = Path(args.out_prefix).resolve() if args.out_prefix else base_dir / f"all_species_heat_trace_predicted_vs_true_{args.prediction}"

    rows = _read_rows(base_dir, prediction_column)
    _write_rows(out_prefix.with_suffix(".csv"), rows)
    title = "Heat-trace individual-count prediction"
    if args.prediction == "loco":
        title += " (leave-one-count-out)"
    _plot(out_prefix, rows, title)

    r2, mae = _scores(rows)
    summary = {"prediction": args.prediction, "r2": r2, "mae": mae, "points": len(rows)}
    out_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[heat-prediction-plot] {out_prefix} r2={r2:.4f} mae={mae:.4f}")


if __name__ == "__main__":
    main()
