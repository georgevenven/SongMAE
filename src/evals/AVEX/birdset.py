import json
from pathlib import Path


TASKS = ("HSN", "UHH", "PER", "NES", "POW", "SNE", "NBP")


def manifest_paths(root):
    root = Path(root) / "manifests"
    return {
        f"{task}-{split}": str(root / f"{task}-{'test' if split == 'test' else 'train'}.jsonl")
        for task in TASKS
        for split in ("train", "validation", "test")
    }


def normalize_manifest(path):
    path = Path(path)
    complete = path.with_suffix(".complete")
    assert path.exists() and complete.exists()
    metadata = json.loads(complete.read_text())

    train = path.name.endswith("-train.jsonl")
    temporary = path.with_suffix(".normalizing")
    changed = kept = seen = 0
    with path.open() as source, temporary.open("w") as target:
        for line in source:
            seen += 1
            row = json.loads(line)
            labels = [label for label in row["labels_as_list"] if label != "None"]
            if labels != row["labels_as_list"]:
                changed += 1
                if train and not labels:
                    continue
                row["labels_as_list"] = labels
                if train:
                    row["stratify_label"] = labels[0]
                line = json.dumps(row) + "\n"
            target.write(line)
            kept += 1

    assert seen == metadata["rows"]
    if not changed:
        temporary.unlink()
        return 0
    temporary.replace(path)
    metadata["rows"] = kept
    temporary = complete.with_suffix(".normalizing")
    temporary.write_text(json.dumps(metadata) + "\n")
    temporary.replace(complete)
    return changed


def normalize_empty_labels(root):
    paths = set(manifest_paths(root).values())
    complete = (path for path in paths if Path(path).with_suffix(".complete").exists())
    return sum(normalize_manifest(path) for path in complete)


def use_local_manifests(root):
    from avex.data.birdset_train_splits import BirdSetTrainSplits

    normalize_empty_labels(root)
    BirdSetTrainSplits.info.split_paths.update(manifest_paths(root))


def keep_test_clean():
    import avex.run_evaluate as runner

    build_dataloaders = runner.build_dataloaders

    def build(*args, **kwargs):
        loaders = build_dataloaders(*args, **kwargs)
        if kwargs["enable_eval_augmentations"]:
            assert loaders[2] is not None
            loaders[2].dataset.postprocessors = []
        return loaders

    runner.build_dataloaders = build
