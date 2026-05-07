#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "individual_id_umap"
OUT_DIR = RESULTS_DIR / "collages"
SPECIES_ORDER = [
    ("zf", "Zebra Finch"),
    ("bf", "Bengalese Finch"),
    ("canary", "Canary"),
    ("ovenbird", "Ovenbird"),
    ("chiffchaff", "Chiffchaff"),
    ("european_starling", "European Starling"),
    ("tree_pipit", "Tree Pipit"),
    ("little_owl", "Little Owl"),
]


def load_font(size, bold=False):
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def trim_image(im):
    bg = Image.new(im.mode, im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    diff = diff.convert("L")
    diff = diff.point(lambda x: 255 if x > 8 else 0)
    bbox = diff.getbbox()
    if bbox is None:
        return im
    left, top, right, bottom = bbox
    pad = 12
    return im.crop(
        (
            max(0, left - pad),
            max(0, top - pad),
            min(im.size[0], right + pad),
            min(im.size[1], bottom + pad),
        )
    )


def fit_panel_image(im, panel_w, panel_h):
    inner = trim_image(im).convert("RGB")
    inner.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
    x = (panel_w - inner.size[0]) // 2
    y = (panel_h - inner.size[1]) // 2
    canvas.paste(inner, (x, y))
    return canvas


def species_assets(run_tag):
    assets = []
    for key, label in SPECIES_ORDER:
        run_dir = RESULTS_DIR / f"{key}_{run_tag}"
        if not run_dir.is_dir():
            matches = sorted(RESULTS_DIR.glob(f"{key}_{run_tag}*"))
            if not matches:
                continue
            assert len(matches) == 1, f"Expected one result dir for {key}_{run_tag}*, found {len(matches)}"
            run_dir = matches[0]
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            continue
        images = sorted(path for path in run_dir.glob("*.png") if not path.stem.endswith("_syllable"))
        assert len(images) == 1, f"Expected one non-syllable UMAP PNG in {run_dir}, found {len(images)}"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        score = summary["silhouette_scores"]["bird"]["score"]
        assets.append((label, images[0], score))
    return assets


def latest_species_assets(prefix):
    assets = []
    for key, label in SPECIES_ORDER:
        run_dirs = []
        for run_dir in RESULTS_DIR.glob(f"{key}_{prefix}*"):
            summary_path = run_dir / "summary.json"
            if summary_path.is_file():
                run_dirs.append(run_dir)
        if not run_dirs:
            continue
        run_dir = max(run_dirs, key=lambda path: (path / "summary.json").stat().st_mtime)
        images = sorted(path for path in run_dir.glob("*.png") if not path.stem.endswith("_syllable"))
        assert len(images) == 1, f"Expected one non-syllable UMAP PNG in {run_dir}, found {len(images)}"
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        score = summary["silhouette_scores"]["bird"]["score"]
        assets.append((label, images[0], score))
    return assets


def score_text(score):
    if score is None:
        return "sil: n/a"
    return f"sil: {float(score):.3f}"


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-tag")
    group.add_argument("--latest-prefix")
    parser.add_argument("--out-name", default=None)
    args = parser.parse_args()

    if args.run_tag is not None:
        assets = species_assets(args.run_tag)
    else:
        assets = latest_species_assets(args.latest_prefix)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cols = 4
    rows = 2
    gutter = 24
    page_margin = 36
    panel_w = 900
    panel_h = 720

    width = page_margin * 2 + cols * panel_w + (cols - 1) * gutter
    height = page_margin * 2 + rows * panel_h + (rows - 1) * gutter

    bg = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(bg)
    score_font = load_font(32, bold=True)

    for idx, (_, img_path, score) in enumerate(assets):
        row = idx // cols
        col = idx % cols
        x = page_margin + col * (panel_w + gutter)
        y = page_margin + row * (panel_h + gutter)
        with Image.open(img_path) as im:
            panel = fit_panel_image(im, panel_w, panel_h)
        bg.paste(panel, (x, y))

        text = score_text(score)
        text_box = draw.textbbox((0, 0), text, font=score_font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        text_x = x + panel_w - text_w - 44
        text_y = y + 58
        draw.rectangle(
            (text_x - 8, text_y - 4, text_x + text_w + 8, text_y + text_h + 6),
            fill=(255, 255, 255),
        )
        draw.text((text_x, text_y), text, font=score_font, fill=(0, 0, 0))

    out_name = args.out_name
    if out_name is None:
        tag = args.run_tag if args.run_tag is not None else f"latest_{args.latest_prefix}"
        out_name = f"individual_id_umaps_{tag}_collage.png"
    out_path = OUT_DIR / out_name
    bg.save(out_path, quality=95)
    print(out_path)


if __name__ == "__main__":
    main()
