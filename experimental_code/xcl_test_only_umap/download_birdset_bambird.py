#!/usr/bin/env python3

import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


ROOT = Path("/media/george-vengrovski/disk1/data/BirdSet_bambird")
BASE = "https://huggingface.co/datasets/DBD-research-group/BirdSet/resolve/data"
SHARDS = {
    "PER": (11, 3, 1),
    "NES": (13, 8, 1),
    "UHH": (5, 7, 1),
    "HSN": (7, 3, 1),
    "NBP": (32, 1, 1),
    "POW": (9, 3, 1),
    "SSW": (29, 36, 4),
    "SNE": (21, 5, 1),
}
SPLITS = ("train", "test", "test5s")


def jobs():
    for dataset, counts in SHARDS.items():
        for split, count in zip(SPLITS, counts):
            for shard in range(1, count + 1):
                yield dataset, split, shard


def download(job):
    dataset, split, shard = job
    name = f"{dataset}_{split}_shard_{shard:04d}.tar.gz"
    archive = ROOT / ".archives" / dataset / name
    marker = ROOT / ".complete" / name
    out = ROOT / dataset / "audio" / split.replace("test5s", "test_5s")
    if marker.exists():
        return

    archive.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "20", "--continue-at", "-", "--output", str(archive), f"{BASE}/{dataset}/{name}"],
        check=True,
    )
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            target = out / Path(member.name).name
            if target.exists() and target.stat().st_size == member.size:
                continue
            source = tar.extractfile(member)
            assert source is not None
            with source, (target.with_suffix(target.suffix + ".part")).open("wb") as handle:
                shutil.copyfileobj(source, handle)
            target.with_suffix(target.suffix + ".part").replace(target)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(name + "\n")
    archive.unlink()
    print(f"complete {name}", flush=True)


def main():
    for dataset in SHARDS:
        metadata = ROOT / dataset / "metadata" / f"{dataset}_metadata_train.parquet"
        assert {"detected_events", "event_cluster"} <= set(pd.read_parquet(metadata).columns)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, jobs()))


if __name__ == "__main__":
    main()
