#!/usr/bin/env python3
"""Select one finetuning learning rate and epoch across tuning birds."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--birds", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    models = sorted(path.name for path in (args.root / "zf" / args.birds[0]).iterdir())
    selections = {}
    for model in models:
        histories = []
        for bird in args.birds:
            path = args.root / "zf" / bird / model / "all" / "train_32s" / "metrics.json"
            histories.append(json.loads(path.read_text())["lr_history"])

        candidates = []
        for lr_index, lr_run in enumerate(histories[0]):
            lr = lr_run["encoder_lr"]
            assert all(history[lr_index]["encoder_lr"] == lr for history in histories)
            epochs = []
            for epoch_index in range(len(lr_run["epochs"])):
                values = [history[lr_index]["epochs"][epoch_index]["dev_macro_fer"] for history in histories]
                epochs.append({"epoch": epoch_index + 1, "mean_dev_macro_fer": sum(values) / len(values)})
            candidates.append({"encoder_lr": lr, **min(epochs, key=lambda row: (row["mean_dev_macro_fer"], row["epoch"]))})

        best = min(candidates, key=lambda row: (row["mean_dev_macro_fer"], row["encoder_lr"], row["epoch"]))
        selections[model] = {**best, "candidates": candidates}

    payload = {"birds": args.birds, "metric": "mean_dev_macro_fer", "models": selections}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
