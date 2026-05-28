#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SPECIES = [
    ("zf", "Zebra Finch"),
    ("bf", "Bengalese Finch"),
    ("canary", "Canary"),
    ("chiffchaff", "Chiffchaff"),
    ("european_starling", "European Starling"),
    ("little_owl", "Little Owl"),
    ("ovenbird", "Ovenbird"),
    ("tree_pipit", "Tree Pipit"),
]

WINDOWS = [
    ("w10_h2", "w10 h2"),
    ("w30_h5", "w30 h5"),
    ("w70_h10", "w70 h10"),
]

FEATURES = [
    ("recsvd15", "SVD 15"),
    ("recaffrow", "Affinity Row"),
]


def _font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = _font(30)
FONT_LABEL = _font(22)
FONT_SMALL = _font(17)


def _out_dir(root, species, window, feature):
    return root / f"{species}_{window}_stats_pca512_{feature}_nn50_cosine"


def _main_png(out_dir):
    paths = [p for p in sorted(out_dir.glob("*.png")) if not p.name.endswith("_syllable.png")]
    return paths[0] if paths else None


def _score(out_dir):
    path = out_dir / "summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    score = data["silhouette_scores"]["bird"]["score"]
    if score is None:
        return None
    return float(score)


def _tile(path, label, size):
    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(210, 210, 210), width=2)
    draw.text((10, 8), label, fill="black", font=FONT_SMALL)

    if path is None:
        draw.text((width // 2 - 35, height // 2 - 10), "missing", fill=(120, 120, 120), font=FONT_LABEL)
        return image

    plot = Image.open(path).convert("RGB")
    plot.thumbnail((width - 18, height - 42), Image.Resampling.LANCZOS)
    x = (width - plot.width) // 2
    y = 38 + (height - 42 - plot.height) // 2
    image.paste(plot, (x, y))
    return image


def _save_grid(path, title, rows, cols, cell_size):
    cell_w, cell_h = cell_size
    title_h = 56
    image = Image.new("RGB", (len(cols) * cell_w, title_h + len(rows) * cell_h), "white")
    draw = ImageDraw.Draw(image)
    draw.text((14, 10), title, fill="black", font=FONT_TITLE)

    for row_i, row in enumerate(rows):
        for col_i, col in enumerate(cols):
            tile = col(row)
            image.paste(tile, (col_i * cell_w, title_h + row_i * cell_h))

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _config_label(window_label, feature_label, out_dir):
    score = _score(out_dir)
    suffix = "score n/a" if score is None else f"score {score:.3f}"
    return f"{window_label} | {feature_label} | {suffix}"


def make_master(root):
    cols = [(w, wl, f, fl) for w, wl in WINDOWS for f, fl in FEATURES]

    def make_col(window, window_label, feature, feature_label):
        def render(row):
            species, species_label = row
            out_dir = _out_dir(root, species, window, feature)
            label = f"{species_label} | {_config_label(window_label, feature_label, out_dir)}"
            return _tile(_main_png(out_dir), label, (320, 320))

        return render

    _save_grid(
        root / "collages" / "master_grid.png",
        "Stats Pooling UMAP Grid",
        SPECIES,
        [make_col(*col) for col in cols],
        (320, 320),
    )


def make_by_config(root):
    for window, window_label in WINDOWS:
        for feature, feature_label in FEATURES:
            def make_col(index):
                def render(row):
                    species, species_label = row
                    out_dir = _out_dir(root, species, window, feature)
                    label = f"{species_label} | {_config_label(window_label, feature_label, out_dir)}"
                    return _tile(_main_png(out_dir), label, (360, 360))

                return render

            rows = [SPECIES[:4], SPECIES[4:]]
            image = Image.new("RGB", (4 * 360, 56 + 2 * 360), "white")
            draw = ImageDraw.Draw(image)
            draw.text((14, 10), f"{window_label} | {feature_label}", fill="black", font=FONT_TITLE)
            for row_i, row_species in enumerate(rows):
                for col_i, species_row in enumerate(row_species):
                    tile = make_col(col_i)(species_row)
                    image.paste(tile, (col_i * 360, 56 + row_i * 360))
            out = root / "collages" / "by_config" / f"{window}_{feature}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            image.save(out)


def make_by_species(root):
    cols = [(f, fl) for f, fl in FEATURES]

    for species, species_label in SPECIES:
        def make_col(feature, feature_label):
            def render(row):
                window, window_label = row
                out_dir = _out_dir(root, species, window, feature)
                label = _config_label(window_label, feature_label, out_dir)
                return _tile(_main_png(out_dir), label, (420, 420))

            return render

        _save_grid(
            root / "collages" / "by_species" / f"{species}.png",
            species_label,
            WINDOWS,
            [make_col(*col) for col in cols],
            (420, 420),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", required=True)
    args = parser.parse_args()

    root = Path(args.out_root)
    make_master(root)
    make_by_config(root)
    make_by_species(root)


if __name__ == "__main__":
    main()
