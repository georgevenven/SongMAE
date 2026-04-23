#!/usr/bin/env python3

import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "individual_id_umap"
OUT_DIR = RESULTS_DIR / "collages"
SPECIES_ORDER = [
    ("zf", "Zebra Finch"),
    ("bf", "Bengalese Finch"),
    ("canary", "Canary"),
    ("chiffchaff", "Chiffchaff"),
    ("european_starling", "European Starling"),
    ("tree_pipit", "Tree Pipit"),
    ("little_owl", "Little Owl"),
    ("orangutan", "Orangutan"),
    ("ovenbird", "Ovenbird"),
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


def species_assets(run_tag, rep_name):
    available = []
    missing = []
    for key, label in SPECIES_ORDER:
        img_path = RESULTS_DIR / f"{key}_{run_tag}" / rep_name
        if img_path.is_file():
            available.append((key, label, img_path))
        else:
            missing.append(label)
    return available, missing


def draw_round_rect(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--rep-name", required=True)
    parser.add_argument("--title", default="Individual-ID UMAP Collage")
    parser.add_argument("--subtitle", required=True)
    parser.add_argument("--out-name", default=None)
    args = parser.parse_args()

    available, missing = species_assets(args.run_tag, args.rep_name)
    assert available, "No completed UMAP PNGs found."
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cols = 3
    rows = (len(available) + cols - 1) // cols
    page_margin = 72
    gutter = 36
    card_w = 1080
    image_h = 760
    label_h = 92
    card_h = image_h + label_h
    title_h = 180
    footer_h = 90 if missing else 40

    width = page_margin * 2 + cols * card_w + (cols - 1) * gutter
    height = page_margin + title_h + rows * card_h + (rows - 1) * gutter + footer_h + page_margin

    bg = Image.new("RGB", (width, height), (244, 240, 232))
    accent = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    accent_draw = ImageDraw.Draw(accent)
    accent_draw.ellipse((-180, -120, 900, 720), fill=(223, 126, 73, 40))
    accent_draw.ellipse((width - 920, height - 700, width + 120, height + 120), fill=(74, 124, 129, 38))
    bg = Image.alpha_composite(bg.convert("RGBA"), accent).convert("RGB")

    draw = ImageDraw.Draw(bg)
    title_font = load_font(60, bold=True)
    subtitle_font = load_font(28)
    label_font = load_font(30, bold=True)
    note_font = load_font(24)

    draw.text((page_margin, page_margin), args.title, font=title_font, fill=(29, 35, 42))
    draw.text((page_margin, page_margin + 76), args.subtitle, font=subtitle_font, fill=(90, 98, 109))
    draw.line(
        (page_margin, page_margin + 128, width - page_margin, page_margin + 128),
        fill=(204, 190, 173),
        width=3,
    )

    for idx, (_, label, img_path) in enumerate(available):
        row = idx // cols
        col = idx % cols
        x = page_margin + col * (card_w + gutter)
        y = page_margin + title_h + row * (card_h + gutter)
        card_box = (x, y, x + card_w, y + card_h)

        shadow = Image.new("RGBA", (card_w + 24, card_h + 24), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.rounded_rectangle((12, 12, card_w + 12, card_h + 12), radius=26, fill=(35, 30, 24, 44))
        shadow = shadow.filter(ImageFilter.GaussianBlur(10))
        bg.paste(shadow, (x - 12, y - 4), shadow)

        draw_round_rect(draw, card_box, radius=26, fill=(252, 251, 248), outline=(217, 209, 198), width=2)

        with Image.open(img_path) as im:
            panel = fit_panel_image(im, card_w - 40, image_h - 34)
        bg.paste(panel, (x + 20, y + 18))

        label_y = y + image_h + 12
        draw.text((x + 28, label_y), label, font=label_font, fill=(36, 44, 51))
        draw.text((x + 28, label_y + 38), img_path.parent.name.split("_data2vec_")[0], font=note_font, fill=(108, 117, 128))

    if missing:
        note = "Missing in this collage: " + ", ".join(missing)
        draw.text((page_margin, height - page_margin - 34), note, font=note_font, fill=(120, 86, 66))

    out_name = args.out_name
    if out_name is None:
        out_name = f"individual_id_umaps_{args.run_tag}_collage.png"
    out_path = OUT_DIR / out_name
    bg.save(out_path, quality=95)
    print(out_path)


if __name__ == "__main__":
    main()
