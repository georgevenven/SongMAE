#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


EXCLUDED_EBIRD_CODES = {
    "comcan",
    "islcan1",
    "x01005",
    "zebfin1",
    "zebfin2",
    "zebfin3",
    "chefin1",
    "x00906",
    "casvir",
    "casvir1",
    "casvir2",
    "y00485",
    "y00484",
    "solvir1",
    "amerob",
    "amerob1",
    "amerob2",
    "amerob3",
    "y00822",
    "whrmun",
    "whrmun8",
    "x01069",
}

XCL_EBIRD_ID_TO_CODE = {
    5821: "casvir",
    6108: "comcan",
    7550: "whrmun",
    7807: "amerob",
}


@dataclass(frozen=True)
class Row:
    name: str
    shard: str
    start: str
    end: str
    root: Path


def parse_args():
    parser = argparse.ArgumentParser(description="Make clean XCL train/val indexes without rewriting shards.")
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out_train_dir", required=True)
    parser.add_argument("--out_val_dir", required=True)
    parser.add_argument("--label_map", help="Optional id-to-eBird-code JSON for numeric annotation labels.")
    return parser.parse_args()


def read_index(root):
    rows = []
    with (root / "shards" / "index.tsv").open() as f:
        assert f.readline().strip() == "name\tshard\tstart\tend"
        for line in f:
            name, shard, start, end = line.rstrip("\n").split("\t")
            rows.append(Row(name, shard, start, end, root))
    return rows


def read_label_map(path):
    if path is None:
        return XCL_EBIRD_ID_TO_CODE

    with path.open() as f:
        raw = json.load(f)
    if "id2label" in raw:
        raw = raw["id2label"]
    if "label2id" in raw:
        return XCL_EBIRD_ID_TO_CODE | {int(v): k for k, v in raw["label2id"].items()}
    if isinstance(raw, list):
        return XCL_EBIRD_ID_TO_CODE | {i: code for i, code in enumerate(raw)}
    return XCL_EBIRD_ID_TO_CODE | {int(k): v for k, v in raw.items()}


def read_codes(path, id_to_code):
    with path.open() as f:
        data = json.load(f)
    codes = {}
    for item in data["recordings"]:
        rec = item["recording"]
        raw = rec["ebird_code"]
        codes[Path(rec["filename"]).stem] = raw if isinstance(raw, str) else id_to_code.get(int(raw), raw)
    return codes


def link_shards(rows, out_dir):
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True)
    for row in rows:
        for suffix in (".npy", ".txt"):
            src = row.root / "shards" / Path(row.shard).with_suffix(suffix)
            if not src.exists():
                continue
            dst = shard_dir / src.name
            if dst.exists():
                assert dst.resolve() == src.resolve(), f"shard name collision: {dst}"
                continue
            dst.symlink_to(src)


def write_index(rows, out_dir):
    lines = ["name\tshard\tstart\tend"]
    lines += [f"{row.name}\t{row.shard}\t{row.start}\t{row.end}" for row in rows]
    (out_dir / "shards" / "index.tsv").write_text("\n".join(lines) + "\n")


def copy_audio_params(src_dir, out_dir):
    shutil.copy2(src_dir / "audio_params.json", out_dir / "audio_params.json")


def main():
    args = parse_args()
    train_dir = Path(args.train_dir).expanduser().resolve()
    val_dir = Path(args.val_dir).expanduser().resolve()
    out_train_dir = Path(args.out_train_dir).expanduser().resolve()
    out_val_dir = Path(args.out_val_dir).expanduser().resolve()

    assert not out_train_dir.exists(), out_train_dir
    assert not out_val_dir.exists(), out_val_dir

    train_rows = read_index(train_dir)
    val_rows = read_index(val_dir)
    label_map = Path(args.label_map).expanduser().resolve() if args.label_map else None
    codes = read_codes(Path(args.annotations).expanduser().resolve(), read_label_map(label_map))

    moved = [row for row in train_rows if codes[row.name] in EXCLUDED_EBIRD_CODES]
    clean_train = [row for row in train_rows if codes[row.name] not in EXCLUDED_EBIRD_CODES]
    clean_val = val_rows + moved

    out_train_dir.mkdir(parents=True)
    out_val_dir.mkdir(parents=True)
    copy_audio_params(train_dir, out_train_dir)
    copy_audio_params(val_dir, out_val_dir)
    link_shards(clean_train, out_train_dir)
    link_shards(clean_val, out_val_dir)
    write_index(clean_train, out_train_dir)
    write_index(clean_val, out_val_dir)

    counts = {code: 0 for code in sorted(EXCLUDED_EBIRD_CODES)}
    for row in moved:
        counts[codes[row.name]] += 1

    print(f"train rows: {len(train_rows)} -> {len(clean_train)}")
    print(f"val rows:   {len(val_rows)} -> {len(clean_val)}")
    print(f"moved rows: {len(moved)} {counts}")
    print(f"wrote {out_train_dir}")
    print(f"wrote {out_val_dir}")


if __name__ == "__main__":
    main()
