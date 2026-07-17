#!/usr/bin/env python3

import ast
import csv
import json
from collections import defaultdict
from pathlib import Path


DISK = Path("/media/george-vengrovski/disk1")
DATA = DISK / "data"
ANNOTATIONS = DATA / "XCL" / "XCL_train_annotations.json"
CLASSES = next((DISK / "huggingface/modules/datasets_modules/datasets/DBD-research-group--BirdSet").glob("*/classes.py"))
TAXONOMY = DISK / "avex/avex/data/ebird_taxonomy_v2021.json"


def read_index(split):
    with (DATA / split / "shards/index.tsv").open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def xcl_labels():
    tree = ast.parse(CLASSES.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "BIRD_NAMES_XENOCANTO"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(CLASSES)


def main():
    data = json.loads(ANNOTATIONS.read_text())
    codes = {
        Path(item["recording"]["filename"]).stem: int(item["recording"]["ebird_code"])
        for item in data["recordings"]
    }
    train = read_index("XCL")
    test = read_index("XCL_val")
    assert all(row["name"] in codes for row in train + test)

    train_codes = {codes[row["name"]] for row in train}
    totals = defaultdict(lambda: [0, 0, 0, ""])
    for row in test:
        species = codes[row["name"]]
        if species not in train_codes:
            timebins = int(row["end"]) - int(row["start"])
            totals[species][0] += 1
            totals[species][1] += timebins
            if timebins > totals[species][2]:
                totals[species][2:] = [timebins, row["name"]]

    labels = xcl_labels()
    taxonomy = json.loads(TAXONOMY.read_text())
    print("label_id\tebird_code\tcommon_name\tscientific_name\trecordings\thours\tlongest_recording\tlongest_minutes")
    for species, (recordings, timebins, longest, recording) in sorted(
        totals.items(), key=lambda item: item[1][2], reverse=True
    ):
        code = labels[species]
        bird = taxonomy[code]
        hours = timebins * 0.005 / 3600
        print(
            f"{species}\t{code}\t{bird['common_name']}\t{bird['sci_name']}\t"
            f"{recordings}\t{hours:.6f}\t{recording}\t{longest * 0.005 / 60:.3f}"
        )


if __name__ == "__main__":
    main()
