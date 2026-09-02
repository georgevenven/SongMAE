#!/usr/bin/env python3
import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from qwen_annotation import (LABELS, REVIEW, SELF_REVIEW, SYSTEM, call, canonical_events, context, image, map_final,
    ownership_x, preview_image, read_done, read_tiles, split_tiles)


SHIFT_REVIEW = SYSTEM + """ You are an independent reviewer. Picture 1 is a clean temporally shifted spectrogram. Inspect it completely before consulting Picture 2, which projects the primary agent's reviewed boxes into this view. Return the complete corrected event set. Preserve good work, but recover misses, reject unsupported boxes, and correct labels, confidence, truncation, overlap, and boundaries."""


def project(events, source_start, source_end, target_start, target_end):
    output = []
    for event in events:
        left, top, right, bottom = event["bbox_2d"]
        left = source_start + left / 1000 * (source_end - source_start)
        right = source_start + right / 1000 * (source_end - source_start)
        box = [round((left - target_start) / (target_end - target_start) * 1000), top,
            round((right - target_start) / (target_end - target_start) * 1000), bottom]
        output.append({**event, "bbox_2d": [max(0, box[0]), top, min(1000, box[2]), bottom]})
    return [x for x in output if x["bbox_2d"][0] < x["bbox_2d"][2]]


def iou(a, b):
    left, top, right, bottom = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / max(1, union)


def reconcile(primary, reviewer):
    if len(primary) != len(reviewer):
        return None
    used, output = set(), []
    for event in primary:
        matches = [(iou(event["bbox_2d"], other["bbox_2d"]), index, other) for index, other in enumerate(reviewer)
            if index not in used and event["label"] == other["label"]]
        if not matches:
            return None
        overlap, index, other = max(matches)
        if overlap < .4 or abs(event["confidence"] - other["confidence"]) > .3:
            return None
        used.add(index)
        output.append({**event,
            "bbox_2d": [round((a + b) / 2) for a, b in zip(event["bbox_2d"], other["bbox_2d"])],
            "confidence": round((event["confidence"] + other["confidence"]) / 2, 3),
            "truncated": event["truncated"] or other["truncated"], "overlap": event["overlap"] or other["overlap"],
            "quality_flags": sorted(set(event["quality_flags"] + other["quality_flags"]))})
    return output


def main():
    parser = argparse.ArgumentParser(description="Annotate with self-review, shifted review, and conditional adjudication.")
    parser.add_argument("--spec-dir", type=Path, default=Path("data/XCL"))
    parser.add_argument("--tiles-from", type=Path, default=Path("data/XCL/qwen38_agentic_annotations.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/XCL/qwen38_adaptive_review_5s_annotations.jsonl"))
    parser.add_argument("--progress", type=Path, default=Path("data/XCL/qwen38_adaptive_review_5s_progress.json"))
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reasoning-budget", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-tiles", type=int)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.spec_dir, args.out, args.progress = args.spec_dir.resolve(), args.out.resolve(), args.progress.resolve()
    params = json.loads((args.spec_dir / "audio_params.json").read_text())
    bins_per_second, ms_per_bin = params["sr"] / params["hop_size"], 1000 / (params["sr"] / params["hop_size"])
    second = round(bins_per_second)
    done = read_done(args.out)
    if args.out.exists():
        for line in args.out.open():
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("adjudicated") and not row.get("events") and any(x.get("events") for x in row.get("passes", [])[:-1]):
                done.discard((row["recording"], row["tile"]["ownership_start_timebin"], row["tile"]["ownership_end_timebin"]))
    tiles = split_tiles(read_tiles(args.tiles_from.resolve()), 5 * second)
    tiles = [x for x in tiles if (x[0], x[4], x[5]) not in done]
    if args.max_tiles:
        tiles = random.Random(args.seed).sample(tiles, min(args.max_tiles, len(tiles)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def progress(tile, stage):
        recording, shard, source_start, source_end, owner_start, owner_end = tile
        view_start = owner_start - (5 * second - (owner_end - owner_start)) // 2
        with lock:
            args.progress.write_text(json.dumps({"recording": recording, "shard": str(args.spec_dir / "shards" / shard),
                "source_start": source_start, "source_end": source_end, "view_start": view_start,
                "view_end": view_start + 5 * second, "ownership_start": owner_start, "ownership_end": owner_end,
                "ownership_start_s": owner_start / bins_per_second, "ownership_end_s": owner_end / bins_per_second,
                "stage": stage, "completed": len(done), "remaining": len(tiles) - len(done)}))

    def annotate(index, tile):
        recording, shard, source_start, source_end, owner_start, owner_end = tile
        view_start = owner_start - (5 * second - (owner_end - owner_start)) // 2
        spec, start, end = context(args.spec_dir, tile, view_start, view_start + 5 * second)
        clean, owner = image(spec), ownership_x(owner_start, owner_end, start, end)
        instruction = f"Ownership is x={owner[0]}..{owner[1]} on the 0-1000 axis. Return only events whose midpoint lies inside it."
        progress(tile, "primary proposal")
        initial = call(args, SYSTEM, instruction, [clean], args.seed + index * 31)
        progress(tile, "primary self-review")
        reviewed = call(args, SELF_REVIEW, instruction + " Picture 2 shows your current boxes in red. Current event JSON: "
            + json.dumps(initial["events"], separators=(",", ":")),
            [clean, preview_image(spec, [x["bbox_2d"] for x in initial["events"]])], args.seed + index * 31 + 1)
        primary = canonical_events(reviewed["events"], start, end, start, end, owner_start, owner_end)

        shift = -1 if index % 2 == 0 else 1
        shifted_spec, shifted_start, shifted_end = context(args.spec_dir, tile, start + shift * second, end + shift * second)
        shifted_primary = project(primary, start, end, shifted_start, shifted_end)
        shifted_owner = [max(0, x) if i == 0 else min(1000, x)
            for i, x in enumerate(ownership_x(owner_start, owner_end, shifted_start, shifted_end))]
        progress(tile, f"independent reviewer {shift:+d}s")
        independent = call(args, SHIFT_REVIEW,
            f"Ownership is x={shifted_owner[0]}..{shifted_owner[1]}. Picture 2 shows the reviewed primary boxes in red. "
            "Their exact coordinates in this shifted view are: " + json.dumps(shifted_primary, separators=(",", ":")),
            [image(shifted_spec), preview_image(shifted_spec, [x["bbox_2d"] for x in shifted_primary])], args.seed + index * 31 + 2)
        reviewer = canonical_events(independent["events"], shifted_start, shifted_end, start, end, owner_start, owner_end)
        agreed = reconcile(primary, reviewer)
        passes = [{"stage": "Primary proposal", "shift_seconds": 0, **initial,
                "events": canonical_events(initial["events"], start, end, start, end, owner_start, owner_end)},
            {"stage": "Primary self-review", "shift_seconds": 0, **reviewed, "events": primary},
            {"stage": "Independent shifted reviewer", "shift_seconds": shift, **independent, "events": reviewer}]

        if agreed is None:
            progress(tile, "conditional adjudicator")
            proposals = json.dumps({"primary": primary, "shifted_reviewer": reviewer}, separators=(",", ":"))
            final = call(args, REVIEW, instruction + " Pictures 2 and 3 show the primary and shifted reviewer proposals. "
                "Use these exact proposal coordinates; do not expect coordinate text inside the images: " + proposals,
                [clean, preview_image(spec, [x["bbox_2d"] for x in primary]), preview_image(spec, [x["bbox_2d"] for x in reviewer])],
                args.seed + index * 31 + 3)
            canonical = canonical_events(final["events"], start, end, start, end, owner_start, owner_end)
            fallback = not canonical and primary and reviewer
            if fallback:
                canonical = primary
                final["public_summary"] = "High-recall fallback retained the self-reviewed primary boxes because both proposals were nonempty but adjudication returned empty. " + final["public_summary"]
            passes.append({"stage": "Conditional adjudicator", "shift_seconds": 0, **final, "events": canonical})
            summary, quality, adjudicated = final["public_summary"], final["window_quality"], True
        else:
            canonical, quality, adjudicated = agreed, independent["window_quality"], False
            summary = "Primary self-review and independent shifted reviewer agreed; matching boxes were averaged without another model call."
        events = map_final(canonical, start, end, owner_start, owner_end, params["mels"], ms_per_bin)
        return {"type": "annotation", "workflow": "adaptive_shifted_review", "status": "ok", "recording": recording,
            "source": {"shard": str(args.spec_dir / "shards" / shard), "start": source_start, "end": source_end},
            "tile": {"start_timebin": start, "end_timebin": end, "ownership_start_timebin": owner_start,
                "ownership_end_timebin": owner_end, "onset_ms": round(start * ms_per_bin, 3), "offset_ms": round(end * ms_per_bin, 3),
                "ownership_onset_ms": round(owner_start * ms_per_bin, 3), "ownership_offset_ms": round(owner_end * ms_per_bin, 3)},
            "events": events, "window_quality": quality, "summary": summary, "adjudicated": adjudicated,
            "high_recall_fallback": bool(adjudicated and fallback), "passes": passes}

    metadata = {"type": "metadata", "schema_version": 2, "workflow": "adaptive_shifted_review", "model": "Qwen3.8-27B-Q8_0",
        "labels": LABELS, "reasoning_effort": "high", "reasoning_budget": args.reasoning_budget, "max_tokens": args.max_tokens,
        "image": {"width": 2048, "height": 512, "seconds": 5}, "ownership_seconds": 5,
        "calls": {"usual": 3, "disagreement": 4}, "private_reasoning_stored": False, "system_prompt": SYSTEM}
    if not args.out.exists():
        args.out.write_text(json.dumps(metadata) + "\n")
    elif not any(row.get("type") == "metadata" and row.get("schema_version") == 2
            for row in (json.loads(line) for line in args.out.open())):
        with args.out.open("a") as file:
            file.write(json.dumps(metadata) + "\n")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(annotate, index, tile): tile for index, tile in enumerate(tiles)}
        for index, future in enumerate(as_completed(futures), 1):
            tile = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = {"type": "annotation", "workflow": "adaptive_shifted_review", "status": "error", "recording": tile[0], "error": str(error)}
            with lock:
                with args.out.open("a") as file:
                    file.write(json.dumps(row, separators=(",", ":")) + "\n")
                if row["status"] == "ok":
                    done.add((tile[0], tile[4], tile[5]))
            print(f"{index}/{len(tiles)} {tile[0]}: {row['status']}", flush=True)


if __name__ == "__main__":
    main()
