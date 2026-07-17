import argparse
import copy
from pathlib import Path

import yaml


TASKS = (
    "dog_classification",
    "bat_classification",
    "mosquito_classification",
    "bird_classification",
    "marine_mammal_classification",
    "esc50_classification",
    "enabirds_detection",
    "rfcx_detection",
    "hiceas_detection",
    "hainan_gibbons_detection",
    "dcase_detection",
)


def stock_config():
    import avex

    return Path(avex.__file__).parents[1] / "configs/data_configs/benchmark_beans.yml"


def local_evaluation(evaluation, root):
    evaluation = copy.deepcopy(evaluation)
    detection = evaluation["train"]["type"] == "detection"
    evaluation["metrics"] = ["mAP" if detection else "accuracy"]
    for name in ("train", "validation", "test"):
        split = evaluation[name]
        split["data_root"] = str(root)
        split["sample_rate"] = 32000
        if detection:
            split["audio_max_length_seconds"] = 5
            for transform in split["transformations"]:
                if "label_map" in transform:
                    labels = [label for label in transform["label_map"] if label != "None"]
                    transform["label_map"] = {label: index for index, label in enumerate(labels)}
    return evaluation


def write_config(path, evaluations):
    config = {"benchmark_name": "beans", "evaluation_sets": evaluations}
    with path.open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, width=120)


def prepare(root, config_dir):
    with stock_config().open() as stream:
        stock = yaml.safe_load(stream)
    by_name = {evaluation["name"]: evaluation for evaluation in stock["evaluation_sets"]}
    evaluations = [local_evaluation(by_name[name], root) for name in TASKS]
    config_dir.mkdir(parents=True, exist_ok=True)
    write_config(config_dir / "beans_all.yml", evaluations)
    for kind in ("classification", "detection"):
        selected = [evaluation for evaluation in evaluations if evaluation["name"].endswith(kind)]
        write_config(config_dir / f"beans_{kind}.yml", selected)
    for evaluation in evaluations:
        write_config(config_dir / f"{evaluation['name']}.yml", [evaluation])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.config_dir)


if __name__ == "__main__":
    main()
