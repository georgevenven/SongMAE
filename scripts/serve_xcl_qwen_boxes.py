#!/usr/bin/env python3
import argparse
import io
import json
import sys
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.utils import load_spec_slice


COLORS = {"target_vocalization": "#ff3b30", "uncertain_vocalization": "#ffd60a", "chorus": "#ff9f0a",
    "non_target_biological": "#64d2ff", "anthropogenic_noise": "#bf5af2", "environmental_noise": "#8e8e93"}


def read_rows(path):
    rows = {}
    if not path.exists():
        return []
    for line in path.open():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") != "ok":
            continue
        tile = row["tile"]
        key = (row["recording"], row["source"]["shard"], tile["ownership_start_timebin"], tile["ownership_end_timebin"])
        rows[key] = row
    return list(rows.values())


def spectrogram(row):
    source, tile = row["source"], row["tile"]
    return load_spec_slice(source["shard"], source["start"] + tile["start_timebin"], source["start"] + tile["end_timebin"])


def render(row, events):
    values = np.clip((spectrogram(row)[::-1] + 100) / 100, 0, 1)
    picture = Image.fromarray(colormaps["viridis"](values, bytes=True)[:, :, :3]).resize((1200, 400))
    draw, tile = ImageDraw.Draw(picture), row["tile"]
    duration = tile["end_timebin"] - tile["start_timebin"]
    for value in (tile["ownership_start_timebin"], tile["ownership_end_timebin"]):
        x = round((value - tile["start_timebin"]) / duration * 1200)
        draw.line((x, 0, x, 400), fill="#ffffff", width=3)
    for event in events:
        left, top, right, bottom = event["bbox_2d"]
        color = COLORS[event["label"]]
        draw.rectangle((left * 1.2, top * .4, right * 1.2, bottom * .4), outline=color, width=4)
        draw.text((left * 1.2 + 4, top * .4 + 3), f"{event['label']} {event['confidence']:.2f}", fill=color)
    output = io.BytesIO()
    picture.save(output, "JPEG", quality=92)
    return output.getvalue()


def render_live(progress):
    source_start, source_end = progress["source_start"], progress["source_end"]
    start, end = progress["view_start"], progress["view_end"]
    valid_start, valid_end = max(0, start), min(source_end - source_start, end)
    spec = load_spec_slice(progress["shard"], source_start + valid_start, source_start + valid_end)
    spec = np.pad(spec, ((0, 0), (valid_start - start, end - valid_end)), constant_values=-100)
    values = np.clip((spec[::-1] + 100) / 100, 0, 1)
    picture = Image.fromarray(colormaps["viridis"](values, bytes=True)[:, :, :3]).resize((1200, 300))
    draw = ImageDraw.Draw(picture)
    for value in (progress["ownership_start"], progress["ownership_end"]):
        x = round((value - start) / (end - start) * 1200)
        draw.line((x, 0, x, 300), fill="#ffffff", width=3)
    output = io.BytesIO()
    picture.save(output, "JPEG", quality=92)
    return output.getvalue()


def page(rows, progress):
    cards = []
    offset = max(0, len(rows) - 8)
    for index, row in reversed(list(enumerate(rows[-8:], offset))):
        tile = row["tile"]
        passes = "".join(f"<section><h3>{escape(value['stage'])} · {len(value['events'])} events</h3>"
            f"<p>{escape(value['public_summary'])}</p><img loading=lazy src='/image?id={index}&pass={i}'>"
            f"<details><summary>Events</summary><pre>{escape(json.dumps(value['events'], indent=2))}</pre></details></section>"
            for i, value in enumerate(row["passes"]))
        cards.append(f"<article><h2>{escape(row['recording'])} · ownership {tile['ownership_onset_ms']/1000:g}–{tile['ownership_offset_ms']/1000:g}s</h2>"
            f"<h3>Final · {len(row['events'])} events</h3><p>{escape(row['summary'])}</p><img loading=lazy src='/image?id={index}'>"
            f"<details><summary>Final events</summary><pre>{escape(json.dumps(row['events'], indent=2))}</pre></details>{passes}</article>")
    status = "Waiting for work."
    live = ""
    if progress:
        status = f"Live: {escape(progress.get('recording', ''))} · {escape(progress.get('stage', ''))} · ownership {progress.get('ownership_start_s', 0):g}–{progress.get('ownership_end_s', 0):g}s"
        live = "<h2>Current five-second view</h2><img src='/live-image'>"
    return """<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><meta http-equiv=refresh content=15><title>Qwen adaptive review</title>
<style>:root{color-scheme:dark}body{max-width:1150px;margin:30px auto;padding:0 18px;background:#0c120f;color:#e9f1ec;font:15px/1.5 system-ui}.live{position:sticky;top:0;padding:12px;background:#202d27;border-left:4px solid #60e6a8;z-index:2}article{margin:24px 0;padding:20px;background:#15201b;border:1px solid #2b3a32;border-radius:12px}img{width:100%;margin:10px 0;border-radius:6px}section{border-top:1px solid #2b3a32;padding-top:14px;margin-top:18px}details{margin:8px 0}summary{cursor:pointer;color:#ff9a89}pre{white-space:pre-wrap;color:#c8d5cd}</style>
<h1>Adaptive shifted Qwen review</h1><p>Primary annotation, mandatory self-review, independent shifted review, and conditional adjudication. Chorus denotes unresolved simultaneous singers. No model-side tools.</p><p class=live>""" + status + f"</p>{live}<p>{len(rows)} completed annotations · refreshes every 15 seconds</p>" + "".join(cards)


class Handler(BaseHTTPRequestHandler):
    def send(self, body, content_type="text/html"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode())

    def do_GET(self):
        url, rows = urlparse(self.path), read_rows(self.server.annotations)
        query = parse_qs(url.query)
        if url.path in ("/", "/ensemble"):
            try:
                progress = json.loads(self.server.progress.read_text())
            except (OSError, json.JSONDecodeError):
                progress = {}
            return self.send(page(rows, progress))
        if url.path == "/image":
            index = int(query.get("id", ["-1"])[0])
            if 0 <= index < len(rows):
                pass_index = query.get("pass")
                events = rows[index]["events"] if pass_index is None else rows[index]["passes"][int(pass_index[0])]["events"]
                return self.send(render(rows[index], events), "image/jpeg")
        if url.path == "/live-image":
            try:
                return self.send(render_live(json.loads(self.server.progress.read_text())), "image/jpeg")
            except (KeyError, OSError, json.JSONDecodeError):
                pass
        self.send_error(404)


def main():
    parser = argparse.ArgumentParser(description="Browse adaptive Qwen annotations.")
    parser.add_argument("--annotations", type=Path, default=Path("data/XCL/qwen38_adaptive_review_5s_annotations.jsonl"))
    parser.add_argument("--progress", type=Path, default=Path("data/XCL/qwen38_adaptive_review_5s_progress.json"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.annotations, server.progress = args.annotations.resolve(), args.progress.resolve()
    print(f"Serving {server.annotations} at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
