#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from data_loader import decode_storage_to_raw, load_quantization_manifest  # noqa: E402
from plotting_utils import SPEC_DPI, SPEC_IMSHOW_KW, SPEC_TITLE_KW, SPEC_TITLE_Y  # noqa: E402


SPECIES = {
    "zf": (
        "Zebra Finch",
        "/media/george-vengrovski/disk2/specs/zf_64hop_32khz",
        ROOT / "files/zf_annotations.json",
    ),
    "bf": (
        "Bengalese Finch",
        "/media/george-vengrovski/disk2/specs/bf_64hop_32khz",
        ROOT / "files/bf_annotations.json",
    ),
    "canary": (
        "Canary",
        "/media/george-vengrovski/disk2/specs/canary_individual_identification_64hop_32khz",
        ROOT / "files/canary_annotations_for_individual_id.json",
    ),
    "chiffchaff": (
        "Chiffchaff",
        "/media/george-vengrovski/disk2/specs/chiffchaff_64hop_32khz",
        ROOT / "files/chiffchaff_annotations.json",
    ),
    "european_starling": (
        "European Starling",
        "/media/george-vengrovski/disk2/specs/european_starling_64hop_32khz_prefixed",
        ROOT / "files/european_starling_annotations_fixed.json",
    ),
    "tree_pipit": (
        "Tree Pipit",
        "/media/george-vengrovski/disk2/specs/tree_pipit_64hop_32khz",
        ROOT / "files/tree_pipit_annotations.json",
    ),
    "little_owl": (
        "Little Owl",
        "/media/george-vengrovski/disk2/specs/little_owl_64hop_32khz",
        ROOT / "files/little_owl_annotations.json",
    ),
    "orangutan": (
        "Orangutan",
        "/media/george-vengrovski/disk2/specs/orangutan_64hop_32khz",
        ROOT / "files/orangutan_annotations.json",
    ),
    "ovenbird": (
        "Ovenbird",
        "/media/george-vengrovski/disk2/specs/ovenbird_lapp_sample_64hop_32khz",
        ROOT / "files/lapp_ovenbird.json",
    ),
}


def clean_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def stable_seed(seed, *parts):
    key = "|".join([str(seed), *[str(part) for part in parts]])
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)


def load_audio_params(spec_dir):
    path = Path(spec_dir) / "audio_params.json"
    assert path.is_file(), f"audio_params.json not found: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def ms_to_timebin(ms, audio):
    return int(round(float(ms) * audio["sr"] / (1000.0 * audio["hop_size"])))


def resolve_spec_path(spec_dir, stem):
    direct = spec_dir / f"{stem}.npy"
    if direct.is_file():
        return direct
    matches = sorted(spec_dir.rglob(f"{stem}.npy"))
    assert len(matches) == 1, f"Expected one spectrogram for {stem}, found {len(matches)}"
    return matches[0]


def events_by_bird(annotation_json):
    data = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
    grouped = {}
    for row in data["recordings"]:
        recording = row["recording"]
        bird_id = str(recording["bird_id"])
        stem = Path(recording["filename"]).stem
        events = row.get("detected_events", []) or [{"onset_ms": 0.0}]
        for event_index, event in enumerate(events):
            grouped.setdefault(bird_id, []).append((stem, event_index, event))
    return grouped


def read_clip(path, start, width, audio, manifest):
    arr = np.load(path, mmap_mode="r")
    stop = min(start + width, arr.shape[1])
    clip = arr[:, start:stop]
    clip = decode_storage_to_raw(
        clip,
        audio.get("storage_dtype", "float32"),
        audio.get("storage_normalization", "none"),
        manifest.get(path.name),
    )
    if clip.shape[1] < width:
        pad_value = float(clip.min()) if clip.size else 0.0
        clip = np.pad(
            clip,
            ((0, 0), (0, width - clip.shape[1])),
            mode="constant",
            constant_values=pad_value,
        )
    return clip


def plot_examples(examples, out_path, seconds):
    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 3.0 * len(examples)), dpi=SPEC_DPI)
    fig.subplots_adjust(hspace=0.75)
    axes = np.atleast_1d(axes)
    for ax, example in zip(axes, examples):
        ax.imshow(example["clip"], extent=(0, seconds, 0, example["clip"].shape[0]), **SPEC_IMSHOW_KW)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(
            0.5,
            SPEC_TITLE_Y,
            example["title"],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            **SPEC_TITLE_KW,
        )
    fig.savefig(out_path, dpi=SPEC_DPI, bbox_inches="tight")
    plt.close(fig)


def render_species(key, label, spec_dir, annotation_json, args):
    spec_dir = Path(spec_dir)
    annotation_json = Path(annotation_json)
    if not spec_dir.is_dir() or not annotation_json.is_file():
        print(f"skip {key}: missing spec dir or annotation JSON")
        return 0

    audio = load_audio_params(spec_dir)
    manifest = load_quantization_manifest(spec_dir, audio)
    clip_width = int(round(args.seconds * audio["sr"] / audio["hop_size"]))
    species_dir = args.out_dir / clean_name(key)
    species_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    bird_events = events_by_bird(annotation_json)
    bird_ids = sorted(bird_events)[: args.max_individuals or None]
    for bird_id in bird_ids:
        events = bird_events[bird_id]
        rng = np.random.default_rng(stable_seed(args.seed, key, bird_id))
        indices = rng.choice(len(events), size=min(args.songs_per_individual, len(events)), replace=False)
        examples = []
        for index in sorted(indices):
            stem, event_index, event = events[index]
            path = resolve_spec_path(spec_dir, stem)
            start = ms_to_timebin(event["onset_ms"], audio)
            clip = read_clip(path, start, clip_width, audio, manifest)
            title = f"{label} | {bird_id} | song {len(examples) + 1} event {event_index + 1}"
            examples.append({"clip": clip, "title": title})
        if not examples:
            continue
        out_path = species_dir / f"{clean_name(key)}__{clean_name(bird_id)}.png"
        plot_examples(examples, out_path, args.seconds)
        written += 1
    return written


def species_jobs(args):
    if args.spec_dir is not None or args.annotation_json is not None:
        assert args.spec_dir is not None and args.annotation_json is not None
        key = args.species[0] if args.species != ["all"] else "custom"
        return [(key, key.replace("_", " ").title(), args.spec_dir, args.annotation_json)]

    keys = list(SPECIES) if args.species == ["all"] else args.species
    return [(key, *SPECIES[key]) for key in keys]


def main():
    parser = argparse.ArgumentParser(description="Render 5-second individual-id spectrogram examples.")
    parser.add_argument("--species", nargs="+", default=["all"])
    parser.add_argument("--spec_dir", type=Path, default=None)
    parser.add_argument("--annotation_json", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "results/individual_id_spectrogram_examples")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--songs_per_individual", type=int, default=3)
    parser.add_argument("--max_individuals", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert args.seconds > 0.0
    assert args.songs_per_individual > 0
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for key, label, spec_dir, annotation_json in species_jobs(args):
        total += render_species(key, label, spec_dir, annotation_json, args)
    print(f"wrote {total} individual spectrogram sheets to {args.out_dir}")


if __name__ == "__main__":
    main()
