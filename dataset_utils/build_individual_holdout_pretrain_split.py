#!/usr/bin/env python3

# This builds a pooled pretraining split from the individual-ID species by
# keeping a small number of individuals per species for train and holding out
# all other individuals for eval. It is intentionally "dirty": flat train/eval
# dirs, deterministic selection, and no per-species subfolders.
#
# The pooled dirs only get structural audio params, not pooled mean/std stats.
# Future agents should use these splits with --input_normalization
# per_file_zscore unless they explicitly recompute mixed-corpus normalization.

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


SPECIES_CONFIGS = {
    "zf": {
        "annotation_json": "zf_annotations.json",
        "spec_dir": "zf_64hop_32khz",
    },
    "bf": {
        "annotation_json": "bf_annotations.json",
        "spec_dir": "bf_64hop_32khz",
    },
    "canary": {
        "annotation_json": "canary_annotations_for_individual_id.json",
        "spec_dir": "canary_individual_identification_64hop_32khz",
    },
    "chiffchaff": {
        "annotation_json": "chiffchaff_annotations.json",
        "spec_dir": "chiffchaff_64hop_32khz",
    },
    "european_starling": {
        "annotation_json": "european_starling_annotations_unprefixed.json",
        "spec_dir": "european_starling_64hop_32khz",
    },
    "tree_pipit": {
        "annotation_json": "tree_pipit_annotations.json",
        "spec_dir": "tree_pipit_64hop_32khz",
    },
    "little_owl": {
        "annotation_json": "little_owl_annotations.json",
        "spec_dir": "little_owl_64hop_32khz",
    },
    "ovenbird": {
        "annotation_json": "lapp_ovenbird.json",
        "spec_dir": "ovenbird_lapp_sample_64hop_32khz",
    },
}

STRUCTURAL_AUDIO_KEYS = ("fft", "hop_size", "mels", "sr")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a pooled train/eval holdout split for continued pretraining on individual-ID data.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output directory that will receive flat train/ and eval/ dirs.",
    )
    parser.add_argument(
        "--source_root",
        default="/media/george-vengrovski/disk2/data2vec_train_data",
        help="Root directory that contains the per-species feature dirs.",
    )
    parser.add_argument(
        "--annotation_root",
        default="/home/george-vengrovski/Documents/projects/TinyBird/files",
        help="Root directory that contains the annotation JSON files.",
    )
    parser.add_argument(
        "--species",
        default=",".join(SPECIES_CONFIGS),
        help="Comma-separated species keys to include.",
    )
    parser.add_argument(
        "--train_individuals_per_species",
        type=int,
        default=2,
        help="How many individuals to keep for train in each species.",
    )
    parser.add_argument(
        "--max_train_stems_per_species",
        type=int,
        default=500,
        help="Optional cap on pooled train stems per species. Use 0 to disable.",
    )
    parser.add_argument(
        "--selection",
        choices=("largest", "random"),
        default="largest",
        help="How to choose the train individuals for each species.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic individual and stem sampling.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the planned split without creating output files.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def total_ms(recording):
    return float(
        sum(event["offset_ms"] - event["onset_ms"] for event in recording["detected_events"])
    )


def get_species_keys(raw_species):
    keys = [item.strip() for item in raw_species.split(",") if item.strip()]
    unknown = sorted(set(keys) - set(SPECIES_CONFIGS))
    assert not unknown, f"Unknown species keys: {unknown}"
    return keys


def get_audio_shape(spec_dir):
    audio_params = load_json(spec_dir / "audio_params.json")
    return {key: audio_params[key] for key in STRUCTURAL_AUDIO_KEYS}


def assert_shared_audio_shape(spec_dirs):
    shared = None
    for spec_dir in spec_dirs:
        current = get_audio_shape(spec_dir)
        if shared is None:
            shared = current
            continue
        assert current == shared, f"Mismatched audio params in {spec_dir}: {current} != {shared}"
    assert shared is not None
    return shared


def get_species_rng(seed, species_key):
    return random.Random(f"{seed}:{species_key}")


def load_species_rows(species_key, config, annotation_root, source_root):
    annotation_path = annotation_root / config["annotation_json"]
    spec_dir = source_root / config["spec_dir"]
    assert annotation_path.is_file(), f"Missing annotation file: {annotation_path}"
    assert spec_dir.is_dir(), f"Missing spec dir: {spec_dir}"

    source_files = {path.stem for path in spec_dir.glob("*.npy")}
    rows_by_stem = {}
    conflicting_stems = set()
    for recording in load_json(annotation_path)["recordings"]:
        stem = Path(recording["recording"]["filename"]).stem
        if stem not in source_files:
            continue
        if stem in conflicting_stems:
            continue
        row = {
            "species": species_key,
            "bird_id": recording["recording"]["bird_id"],
            "stem": stem,
            "ms": total_ms(recording),
            "source_path": spec_dir / f"{stem}.npy",
        }
        existing = rows_by_stem.get(stem)
        if existing is None:
            rows_by_stem[stem] = row
            continue
        if existing["bird_id"] != row["bird_id"]:
            conflicting_stems.add(stem)
            del rows_by_stem[stem]
    rows = sorted(rows_by_stem.values(), key=lambda row: row["stem"])
    assert rows, f"No matching feature files found for {species_key}"
    return rows, spec_dir, len(conflicting_stems)


def choose_train_ids(by_bird, train_individuals_per_species, selection, rng):
    bird_ids = sorted(by_bird)
    assert len(bird_ids) > train_individuals_per_species, (
        f"Need held-out individuals, but only found {len(bird_ids)} bird_ids"
    )
    if selection == "random":
        return sorted(rng.sample(bird_ids, train_individuals_per_species))
    ranked = sorted(
        bird_ids,
        key=lambda bird_id: (-len(by_bird[bird_id]), bird_id),
    )
    return sorted(ranked[:train_individuals_per_species])


def choose_train_rows(rows, train_ids, max_train_stems_per_species, rng):
    train_rows = [row for row in rows if row["bird_id"] in train_ids]
    if max_train_stems_per_species <= 0 or len(train_rows) <= max_train_stems_per_species:
        return sorted(train_rows, key=lambda row: row["stem"])
    picked = rng.sample(train_rows, max_train_stems_per_species)
    return sorted(picked, key=lambda row: row["stem"])


def build_species_plan(
    species_key,
    rows,
    skipped_conflicting_stems,
    train_individuals_per_species,
    max_train_stems_per_species,
    selection,
    seed,
):
    by_bird = defaultdict(list)
    for row in rows:
        by_bird[row["bird_id"]].append(row)

    rng = get_species_rng(seed, species_key)
    train_ids = choose_train_ids(by_bird, train_individuals_per_species, selection, rng)
    train_rows = choose_train_rows(rows, set(train_ids), max_train_stems_per_species, rng)
    train_stems = {row["stem"] for row in train_rows}
    eval_rows = sorted(
        [row for row in rows if row["stem"] not in train_stems],
        key=lambda row: row["stem"],
    )
    eval_ids = sorted(set(by_bird) - set(train_ids))
    assert train_rows, f"No train rows selected for {species_key}"
    assert eval_rows, f"No eval rows selected for {species_key}"
    assert not ({row["stem"] for row in eval_rows} & train_stems)

    return {
        "species": species_key,
        "train_ids": train_ids,
        "eval_ids": eval_ids,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "train_total_ms": sum(row["ms"] for row in train_rows),
        "eval_total_ms": sum(row["ms"] for row in eval_rows),
        "available_individuals": len(by_bird),
        "available_stems": len(rows),
        "skipped_conflicting_stems": skipped_conflicting_stems,
    }


def copy_rows(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for row in rows:
        target = output_dir / row["source_path"].name
        assert not target.exists(), f"Filename collision in {output_dir}: {target.name}"
        shutil.copy2(row["source_path"], target)
        copied += 1
    return copied


def write_audio_params(output_dir, shared_audio_params, species_keys):
    payload = dict(shared_audio_params)
    payload["pooled_species"] = species_keys
    payload["normalization_hint"] = "Use per_file_zscore for pooled mixed-species runs."
    with open(output_dir / "audio_params.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def summarize_plan(plan):
    total_train_ms = 0.0
    total_eval_ms = 0.0
    total_train_rows = 0
    total_eval_rows = 0
    lines = []
    for item in plan:
        train_hours = item["train_total_ms"] / 1000.0 / 3600.0
        eval_hours = item["eval_total_ms"] / 1000.0 / 3600.0
        total_train_ms += item["train_total_ms"]
        total_eval_ms += item["eval_total_ms"]
        total_train_rows += len(item["train_rows"])
        total_eval_rows += len(item["eval_rows"])
        lines.append(
            {
                "species": item["species"],
                "available_individuals": item["available_individuals"],
                "available_stems": item["available_stems"],
                "skipped_conflicting_stems": item["skipped_conflicting_stems"],
                "train_ids": item["train_ids"],
                "eval_ids": item["eval_ids"],
                "train_stems": len(item["train_rows"]),
                "eval_stems": len(item["eval_rows"]),
                "train_hours": round(train_hours, 3),
                "eval_hours": round(eval_hours, 3),
            }
        )
    return {
        "species": lines,
        "totals": {
            "train_stems": total_train_rows,
            "eval_stems": total_eval_rows,
            "train_hours": round(total_train_ms / 1000.0 / 3600.0, 3),
            "eval_hours": round(total_eval_ms / 1000.0 / 3600.0, 3),
        },
    }


def write_manifest(output_root, args, summary):
    payload = {
        "config": {
            "source_root": str(Path(args.source_root).resolve()),
            "annotation_root": str(Path(args.annotation_root).resolve()),
            "species": get_species_keys(args.species),
            "train_individuals_per_species": args.train_individuals_per_species,
            "max_train_stems_per_species": args.max_train_stems_per_species,
            "selection": args.selection,
            "seed": args.seed,
        },
        "summary": summary,
    }
    with open(output_root / "split_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main():
    args = parse_args()
    assert args.train_individuals_per_species > 0

    species_keys = get_species_keys(args.species)
    annotation_root = Path(args.annotation_root).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    train_dir = output_root / "train"
    eval_dir = output_root / "eval"

    species_rows = {}
    skipped_conflicting_stems = {}
    spec_dirs = []
    for species_key in species_keys:
        rows, spec_dir, skipped_count = load_species_rows(
            species_key,
            SPECIES_CONFIGS[species_key],
            annotation_root,
            source_root,
        )
        species_rows[species_key] = rows
        skipped_conflicting_stems[species_key] = skipped_count
        spec_dirs.append(spec_dir)

    shared_audio_params = assert_shared_audio_shape(spec_dirs)
    plan = [
        build_species_plan(
            species_key,
            species_rows[species_key],
            skipped_conflicting_stems[species_key],
            args.train_individuals_per_species,
            args.max_train_stems_per_species,
            args.selection,
            args.seed,
        )
        for species_key in species_keys
    ]
    summary = summarize_plan(plan)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        return

    assert not output_root.exists(), f"Output root already exists: {output_root}"
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)

    all_train_rows = []
    all_eval_rows = []
    for item in plan:
        all_train_rows.extend(item["train_rows"])
        all_eval_rows.extend(item["eval_rows"])

    copy_rows(all_train_rows, train_dir)
    copy_rows(all_eval_rows, eval_dir)
    write_audio_params(train_dir, shared_audio_params, species_keys)
    write_audio_params(eval_dir, shared_audio_params, species_keys)
    write_manifest(output_root, args, summary)


if __name__ == "__main__":
    main()
