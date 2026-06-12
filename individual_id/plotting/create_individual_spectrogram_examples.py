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
sys.path.append(str(ROOT))

from src.core.utils import load_spec  # noqa: E402
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

FULL_SONG_SPECIES = {
    "zf": SPECIES["zf"],
    "bf": SPECIES["bf"],
    "canary": SPECIES["canary"],
    "chiffchaff": SPECIES["chiffchaff"],
    "european_starling": (
        "European Starling",
        "/media/george-vengrovski/disk2/specs/european_starling_64hop_32khz",
        ROOT / "files/european_starling_annotations_unprefixed.json",
    ),
    "little_owl": SPECIES["little_owl"],
    "ovenbird": SPECIES["ovenbird"],
    "tree_pipit": SPECIES["tree_pipit"],
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


def read_clip(path, start, width):
    arr = load_spec(path)
    stop = min(start + width, arr.shape[1])
    clip = arr[:, start:stop]
    if clip.shape[1] < width:
        pad_value = float(clip.min()) if clip.size else 0.0
        clip = np.pad(
            clip,
            ((0, 0), (0, width - clip.shape[1])),
            mode="constant",
            constant_values=pad_value,
        )
    return clip


def read_full_song(path):
    return load_spec(path)


def read_span(path, start_ms, stop_ms, audio):
    arr = load_spec(path)
    start = ms_to_timebin(start_ms, audio)
    stop = min(ms_to_timebin(stop_ms, audio), arr.shape[1])
    assert stop > start, f"empty spectrogram span: {path} {start_ms}-{stop_ms} ms"
    return arr[:, start:stop]


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


def plot_full_song_sheet(examples, out_path, max_seconds, color_limits, width):
    scale_seconds = 10.0 if max_seconds >= 25.0 else 2.0

    fig, axes = plt.subplots(len(examples), 1, figsize=(width, 1.05 * len(examples) + 0.8), dpi=SPEC_DPI)
    fig.subplots_adjust(left=0.18, right=0.99, top=0.98, bottom=0.12, hspace=0.16)
    axes = np.atleast_1d(axes)

    for ax, example in zip(axes, examples):
        ax.imshow(
            example["clip"],
            extent=(0, example["seconds"], 0, example["clip"].shape[0]),
            vmin=color_limits[0],
            vmax=color_limits[1],
            **SPEC_IMSHOW_KW,
        )
        ax.set_xlim(0, max_seconds)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(example["label"], rotation=0, ha="right", va="center", fontsize=11, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)

    x0 = 0.25
    bottom_ax = axes[-1]
    bottom_ax.plot(
        [x0, x0 + scale_seconds],
        [-0.22, -0.22],
        transform=bottom_ax.get_xaxis_transform(),
        color="black",
        lw=2.0,
        clip_on=False,
    )
    bottom_ax.text(
        x0 + scale_seconds / 2.0,
        -0.35,
        f"{scale_seconds:g} s",
        transform=bottom_ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=10,
        clip_on=False,
    )
    fig.savefig(out_path, dpi=SPEC_DPI, bbox_inches="tight")
    plt.close(fig)


def load_full_song_candidates(key, label, spec_dir, annotation_json, args):
    data = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
    rng = np.random.default_rng(stable_seed(args.seed, key, "full_song_sheet"))
    rows = event_candidates(data) if args.detected_events_only else [(row, None, None) for row in data["recordings"]]
    order = rng.permutation(len(rows))
    selected = [rows[index] for index in order[: args.num_collages]]
    assert len(selected) == args.num_collages, f"{key} has fewer than {args.num_collages} recordings"
    return {
        "key": key,
        "label": label,
        "spec_dir": Path(spec_dir),
        "annotation_json": Path(annotation_json),
        "rows": selected,
        "audio": load_audio_params(Path(spec_dir)),
    }


def event_candidates(data):
    rows = []
    for row in data["recordings"]:
        for event_index, event in enumerate(row.get("detected_events", [])):
            if "offset_ms" in event and event["offset_ms"] > event["onset_ms"]:
                rows.append((row, event_index, event))
    return rows


def read_full_song_example(job, collage_index):
    row, event_index, event = job["rows"][collage_index]
    stem = Path(row["recording"]["filename"]).stem
    path = resolve_spec_path(job["spec_dir"], stem)
    if event is None:
        clip = read_full_song(path)
    else:
        clip = read_span(path, event["onset_ms"], event["offset_ms"], job["audio"])
    seconds = clip.shape[1] * job["audio"]["hop_size"] / job["audio"]["sr"]
    return {
        "label": job["label"],
        "clip": clip,
        "seconds": seconds,
        "metadata": {
            "species": job["key"],
            "label": job["label"],
            "spectrogram": str(path),
            "annotation_json": str(job["annotation_json"]),
            "filename": row["recording"]["filename"],
            "bird_id": row["recording"].get("bird_id"),
            "duration_seconds": seconds,
            "detected_events": len(row.get("detected_events", [])),
            "event_index": event_index,
            "event_onset_ms": None if event is None else event["onset_ms"],
            "event_offset_ms": None if event is None else event["offset_ms"],
        },
    }


def color_limits(sheets):
    values = []
    for examples in sheets:
        for example in examples:
            clip = example["clip"]
            values.append(np.percentile(clip, [1.0, 99.5]))
    limits = np.asarray(values)
    return float(limits[:, 0].min()), float(limits[:, 1].max())


def render_full_song_sheets(args):
    keys = list(FULL_SONG_SPECIES) if args.species == ["all"] else args.species
    jobs = []

    for key in keys:
        assert key in FULL_SONG_SPECIES, f"Unsupported full-song species: {key}"
        label, spec_dir, annotation_json = FULL_SONG_SPECIES[key]
        jobs.append(load_full_song_candidates(key, label, spec_dir, annotation_json, args))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sheets = [[read_full_song_example(job, index) for job in jobs] for index in range(args.num_collages)]
    max_seconds = max(example["seconds"] for examples in sheets for example in examples)
    limits = color_limits(sheets)

    metadata = []
    prefix = "species_detected_event_sheet" if args.detected_events_only else "species_full_song_sheet"
    for collage_index, examples in enumerate(sheets):
        out_path = args.out_dir / f"{prefix}_{collage_index + 1:03d}.png"
        plot_full_song_sheet(examples, out_path, max_seconds, limits, args.sheet_width)
        metadata.append(
            {
                "collage": out_path.name,
                "xlim_seconds": max_seconds,
                "color_vmin": limits[0],
                "color_vmax": limits[1],
                "examples": [example["metadata"] for example in examples],
            }
        )

    (args.out_dir / f"{prefix}s.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def render_species(key, label, spec_dir, annotation_json, args):
    spec_dir = Path(spec_dir)
    annotation_json = Path(annotation_json)
    if not spec_dir.is_dir() or not annotation_json.is_file():
        print(f"skip {key}: missing spec dir or annotation JSON")
        return 0

    audio = load_audio_params(spec_dir)
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
            clip = read_clip(path, start, clip_width)
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
    parser = argparse.ArgumentParser(description="Render individual-id spectrogram examples.")
    parser.add_argument("--species", nargs="+", default=["all"])
    parser.add_argument("--spec_dir", type=Path, default=None)
    parser.add_argument("--annotation_json", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "results/individual_id_spectrogram_examples")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--songs_per_individual", type=int, default=3)
    parser.add_argument("--max_individuals", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--species_song_sheet", action="store_true")
    parser.add_argument("--num_collages", type=int, default=1)
    parser.add_argument("--detected_events_only", action="store_true")
    parser.add_argument("--sheet_width", type=float, default=12.0)
    args = parser.parse_args()

    assert args.seconds > 0.0
    assert args.songs_per_individual > 0
    assert args.num_collages > 0
    assert args.sheet_width > 0.0
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.species_song_sheet:
        render_full_song_sheets(args)
        print(f"wrote {args.num_collages} species sheets to {args.out_dir}")
        return

    total = 0
    for key, label, spec_dir, annotation_json in species_jobs(args):
        total += render_species(key, label, spec_dir, annotation_json, args)
    print(f"wrote {total} individual spectrogram sheets to {args.out_dir}")


if __name__ == "__main__":
    main()
