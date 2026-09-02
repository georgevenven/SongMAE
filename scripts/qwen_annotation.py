import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
from matplotlib import colormaps
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.utils import load_spec_slice


LABELS = ["target_vocalization", "uncertain_vocalization", "chorus", "non_target_biological", "anthropogenic_noise", "environmental_noise"]
FLAGS = ["faint", "ambiguous_boundary", "overlapping_sound", "possible_noise", "partially_occluded", "context_unavailable"]
SYSTEM = """You are an expert bioacoustic spectrogram annotator. Detect and tightly localize every avian vocalization in the supplied spectrogram. Target means any avian vocalization regardless of species; compact confusable non-avian or noise events may receive an explicit non-target label. The image origin is top-left: x increases rightward, y increases downward, and high frequencies are near y=0. Coordinates are integer [x_min,y_min,x_max,y_max] values normalized to 0-1000 over the complete image. One event is one temporally continuous vocal element or tightly spaced phrase. Separate genuinely silent or structurally discontinuous events; never merge repetitions merely because their shapes match. Do not cut off harmonics. Inspect left-to-right twice, including faint and overlapping events. Emit an event only when its temporal midpoint lies inside the supplied ownership interval. Context outside ownership is only for classification and complete boundaries. Use uncertain_vocalization when vocal evidence is plausible but insufficient. Use chorus for dense simultaneous vocal energy from likely multiple singers only when individual events cannot be separated reliably. If simultaneous singers remain separable, emit separate target_vocalization boxes with overlap=true. A dense trill from one apparent singer and sequential non-overlapping calls are not chorus. Never double-label the same energy as chorus and individual vocalizations. Split chorus boxes at genuine temporal silence rather than boxing the whole window. Chorus is positive foreground for song detection and mixed-identity foreground that should be excluded or treated specially for individual identification. Confidence is the probability that the stated label is correct. Never invent identity or species. Spectrogram evidence only; no audio is available. Return the required JSON and a concise public summary of visible evidence and uncertainty; do not expose private chain-of-thought."""
REVIEW = SYSTEM + """ You are the final adjudicator. The first image is clean; following images contain red proposal outlines. Exact proposal coordinates are supplied as JSON in the user message, so boxes do not need printed coordinate text. Reinspect the clean spectrogram yourself: agreement is evidence, not truth. Recover misses, reject unsupported boxes, tighten boundaries, remove duplicates, and obey ownership. Never return empty merely because an overlay lacks readable text; derive boxes directly from visible spectrogram evidence."""
SELF_REVIEW = SYSTEM + """ You are re-inspecting your own annotation. Picture 1 is clean and Picture 2 renders your current events. Perform another complete left-to-right audit. Correct misses, false positives, labels, confidence, truncation, overlap, and boundaries. Return the complete corrected event set."""


def schema():
    event = {"type": "object", "properties": {
        "label": {"type": "string", "enum": LABELS},
        "bbox_2d": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000}, "minItems": 4, "maxItems": 4},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "truncated": {"type": "boolean"}, "overlap": {"type": "boolean"},
        "quality_flags": {"type": "array", "items": {"type": "string", "enum": FLAGS}, "uniqueItems": True}},
        "required": ["label", "bbox_2d", "confidence", "truncated", "overlap", "quality_flags"], "additionalProperties": False}
    quality = {"type": "object", "properties": {"clipping": {"type": "boolean"}, "heavy_noise": {"type": "boolean"}, "low_contrast": {"type": "boolean"}},
        "required": ["clipping", "heavy_noise", "low_contrast"], "additionalProperties": False}
    return {"type": "object", "properties": {"events": {"type": "array", "items": event, "maxItems": 64}, "window_quality": quality,
        "public_summary": {"type": "string", "minLength": 40, "maxLength": 600}},
        "required": ["events", "window_quality", "public_summary"], "additionalProperties": False}


def call(args, system, instruction, pictures, seed):
    content = [{"type": "text", "text": instruction}] + [{"type": "image_url", "image_url": {"url": data_url(x)}} for x in pictures]
    payload = {"model": "qwen3.8-27b-q8", "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        "temperature": .5, "top_p": .95, "top_k": 20, "max_tokens": args.max_tokens, "reasoning_effort": "high",
        "reasoning_budget_tokens": args.reasoning_budget, "chat_template_kwargs": {"add_vision_id": True}, "seed": seed,
        "response_format": {"type": "json_schema", "json_schema": {"name": "spectrogram_events", "strict": True, "schema": schema()}}}
    for attempt in range(3):
        try:
            response = requests.post(args.url, json=payload, timeout=args.timeout)
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
            return json.loads(text[text.find("{"):text.rfind("}") + 1])
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def read_tiles(path):
    rows = (json.loads(line) for line in path.open())
    return [(row["recording"], Path(row["source"]["shard"]).name, row["source"]["start"], row["source"]["end"],
        row["tile"]["start_timebin"], row["tile"]["end_timebin"]) for row in rows if row.get("status") == "ok"]


def image(spec):
    values = np.clip((spec + 100) / 100, 0, 1)
    pixels = colormaps["viridis"](values[::-1], bytes=True)[:, :, :3]
    return Image.fromarray(pixels).resize((2048, 512), Image.Resampling.BILINEAR)


def preview_image(spec, boxes):
    picture = image(spec)
    draw = ImageDraw.Draw(picture)
    width, height = picture.size
    for left, top, right, bottom in boxes:
        draw.rectangle((left * width / 1000, top * height / 1000, right * width / 1000, bottom * height / 1000),
            outline="#ff3b30", width=max(4, min(width, height) // 64))
    return picture


def data_url(picture):
    output = io.BytesIO()
    picture.save(output, "PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def context(spec_dir, tile, start, end):
    _, shard, source_start, source_end, _, _ = tile
    length = source_end - source_start
    valid_start, valid_end = max(0, start), min(length, end)
    spec = load_spec_slice(spec_dir / "shards" / shard, source_start + valid_start, source_start + valid_end)
    return np.pad(spec, ((0, 0), (valid_start - start, end - valid_end)), constant_values=-100), start, end


def ownership_x(owner_start, owner_end, view_start, view_end):
    scale = 1000 / max(1, view_end - view_start)
    return [round((owner_start - view_start) * scale), round((owner_end - view_start) * scale)]


def canonical_events(events, source_start, source_end, canonical_start, canonical_end, owner_start, owner_end):
    output = []
    for event in events:
        left, top, right, bottom = event["bbox_2d"]
        absolute_left = source_start + left / 1000 * (source_end - source_start)
        absolute_right = source_start + right / 1000 * (source_end - source_start)
        if left >= right or top >= bottom or not owner_start <= (absolute_left + absolute_right) / 2 < owner_end:
            continue
        mapped = dict(event)
        mapped["bbox_2d"] = [max(0, round((absolute_left - canonical_start) / (canonical_end - canonical_start) * 1000)), top,
            min(1000, round((absolute_right - canonical_start) / (canonical_end - canonical_start) * 1000)), bottom]
        output.append(mapped)
    return output


def map_final(events, context_start, context_end, owner_start, owner_end, mels, ms_per_bin):
    output = []
    for event in events:
        left, top, right, bottom = event["bbox_2d"]
        start = context_start + round(left / 1000 * (context_end - context_start))
        end = context_start + round(right / 1000 * (context_end - context_start))
        if left >= right or top >= bottom or not owner_start <= (start + end) / 2 < owner_end:
            continue
        output.append({**event, "start_timebin": start, "end_timebin": end, "onset_ms": round(start * ms_per_bin, 3),
            "offset_ms": round(end * ms_per_bin, 3), "low_mel_bin": round((1000 - bottom) / 1000 * mels),
            "high_mel_bin": round((1000 - top) / 1000 * mels)})
    return output


def read_done(path):
    if not path.exists():
        return set()
    rows = (json.loads(line) for line in path.open())
    return {(x["recording"], x["tile"]["ownership_start_timebin"], x["tile"]["ownership_end_timebin"])
        for x in rows if x.get("status") == "ok"}


def split_tiles(tiles, width):
    return [(recording, shard, source_start, source_end, start, min(start + width, owner_end))
        for recording, shard, source_start, source_end, owner_start, owner_end in tiles
        for start in range(owner_start, owner_end, width)]
