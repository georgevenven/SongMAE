import argparse
import copy
import json
from functools import cache
from pathlib import Path

import yaml
from datasets import Audio, load_dataset, load_dataset_builder
from huggingface_hub import HfApi, hf_hub_download

from .birdset import TASKS, normalize_manifest


REPO = "mteb/BirdSet"
REVISION = "bdaa5020a8dc594a9a1d3b344e6ca9dbfaa33c74"
ALIASES = {
    "brnjay": "Passeriformes Corvidae Cyanocorax morio",
    "crelar1": "Passeriformes Alaudidae Galerida cristata",
    "pasfly": "Passeriformes Tyrannidae Empidonax difficilis [difficilis Group]",
    "scbwoo5": "Piciformes Picidae Celeus undatus [grammicus Group]",
    "yehcar1": "Falconiformes Falconidae Daptrius chimachima",
}


def stock_files():
    import avex

    root = Path(avex.__file__).parents[1]
    return root / "configs/data_configs/benchmark_birdset.yml", root / "avex/data/ebird_taxonomy_v2021.json"


def label_lookup(label_map, taxonomy, codes):
    lookup = {}
    for code in codes:
        matches = [label for label in label_map if label.endswith(" " + taxonomy[code]["sci_name"])]
        label = ALIASES.get(code, matches[0] if len(matches) == 1 else None)
        assert label in label_map or code == "runwre1", (code, matches)
        lookup[code] = label
    return lookup


def local_config(evaluation_set, root):
    evaluation_set = copy.deepcopy(evaluation_set)
    for name in ("train", "validation", "test"):
        split = evaluation_set[name]
        split["sample_rate"] = 32000
        split["data_root"] = str(root)
        for transform in split["transformations"]:
            if transform["type"] == "train_val_split":
                transform["stratify_column"] = "stratify_label"
            if "label_map" in transform:
                labels = [label for label in transform["label_map"] if label != "None"]
                transform["label_map"] = {label: index for index, label in enumerate(labels)}
    return {"benchmark_name": "birdset_avex_paper", "evaluation_sets": [evaluation_set]}


@cache
def repo_files():
    return tuple(HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION))


def remote_files(task, split):
    prefix = f"{task}/{split}-"
    return sorted(path for path in repo_files() if path.startswith(prefix) and path.endswith(".parquet"))


def download_files(task, split, cache_dir):
    return [
        hf_hub_download(
            REPO,
            path,
            repo_type="dataset",
            revision=REVISION,
            cache_dir=cache_dir,
        )
        for path in remote_files(task, split)
    ]


def write_split(task, split, root, cache_dir, lookup, codes):
    output_split = "test" if split == "test_5s" else "train"
    manifest = root / "manifests" / f"{task}-{output_split}.jsonl"
    complete = manifest.with_suffix(".complete")
    if complete.exists():
        normalize_manifest(manifest)
        return

    files = download_files(task, split, cache_dir)
    dataset = load_dataset("parquet", data_files={split: files}, split=split, streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    audio_dir = root / "audio" / task / output_split
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    temporary = manifest.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for index, row in enumerate(dataset):
            ids = row["ebird_code_multilabel"] if split == "test_5s" else [row["ebird_code"]]
            labels = [lookup[codes[label]] for label in ids if lookup[codes[label]] is not None]
            if split == "train" and not labels:
                continue

            audio = row["audio"]
            suffix = Path(audio["path"] or "audio.ogg").suffix or ".ogg"
            relative = Path("audio") / task / output_split / f"{index:06d}{suffix}"
            path = root / relative
            encoded = audio["bytes"]
            assert encoded
            if not path.exists() or path.stat().st_size != len(encoded):
                path.write_bytes(encoded)
            item = {"path": str(relative), "labels_as_list": labels}
            if split == "train":
                item["stratify_label"] = labels[0]
            stream.write(json.dumps(item) + "\n")
            kept += 1
            if kept % 1000 == 0:
                print(f"{task} {split}: {kept}", flush=True)

    temporary.replace(manifest)
    marker = complete.with_name(complete.name + ".tmp")
    marker.write_text(json.dumps({"rows": kept, "revision": REVISION}) + "\n")
    marker.replace(complete)


def prepare(task, root, cache_dir, config_dir, stock, taxonomy):
    name = f"birdset_{task.lower()}_detection"
    evaluation_set = next(item for item in stock["evaluation_sets"] if item["name"] == name)
    label_map = evaluation_set["train"]["transformations"][-1]["label_map"]
    builder = load_dataset_builder(REPO, task, revision=REVISION, cache_dir=cache_dir)
    codes = builder.info.features["ebird_code"].names
    lookup = label_lookup(label_map, taxonomy, codes)
    write_split(task, "train", root, cache_dir, lookup, codes)
    write_split(task, "test_5s", root, cache_dir, lookup, codes)
    config_dir.mkdir(parents=True, exist_ok=True)
    config = local_config(evaluation_set, root)
    with (config_dir / f"birdset_{task.lower()}_detection.yml").open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return config["evaluation_sets"][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    args = parser.parse_args()

    config_path, taxonomy_path = stock_files()
    with config_path.open() as stream:
        stock = yaml.safe_load(stream)
    with taxonomy_path.open() as stream:
        taxonomy = json.load(stream)
    evaluations = [
        prepare(task, args.root, args.cache_dir, args.config_dir, stock, taxonomy)
        for task in args.tasks
    ]
    config = {"benchmark_name": "birdset_avex_paper", "evaluation_sets": evaluations}
    with (args.config_dir / "birdset_all.yml").open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)


if __name__ == "__main__":
    main()
