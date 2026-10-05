"""Server-side rendering of the day's collage.

The web collage (avian/frontend/apt.js) packs bird silhouettes against
each other on plain paper. This version does the same for a fixed-size
PNG that Home Assistant can show as an image entity: every bird's real
alpha silhouette plus its handwritten label becomes a footprint on a
coarse occupancy grid, and birds are placed largest-first from the
centre outward wherever their footprint fits. Frequent visitors are drawn
larger. When everything doesn't fit, the scale shrinks and it re-packs.
"""
from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CELL = 4  # px per occupancy cell
GAP = 3   # cells of clearance around each footprint

LIGHT = {"paper": (252, 252, 251), "ink": (26, 22, 18), "soft": (144, 133, 118)}
DARK = {"paper": (23, 24, 28), "ink": (236, 232, 225), "soft": (131, 124, 112)}


@dataclass(frozen=True)
class Visitor:
    com: str
    sci: str
    image: Path
    count: int


def _font(path: Path | None, size: int) -> ImageFont.ImageFont:
    if path and path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha1(text.encode()).digest()[:4], "big")


class _Sprite:
    """A bird with its label, rendered at one scale, plus its grid footprint."""

    def __init__(self, visitor: Visitor, bird: Image.Image, edge: int, font_path: Path | None, ink):
        scale = edge / max(bird.size)
        w, h = max(1, round(bird.width * scale)), max(1, round(bird.height * scale))
        bird = bird.resize((w, h), Image.LANCZOS)

        font = _font(font_path, int(min(34, max(18, edge * 0.16))))
        label_box = font.getbbox(visitor.com)
        lw, lh = label_box[2] - label_box[0], label_box[3] - label_box[1]
        pad = max(4, lh // 3)

        self.width = max(w, lw)
        self.height = h + pad + lh + 2
        sprite = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        sprite.alpha_composite(bird, ((self.width - w) // 2, 0))
        ImageDraw.Draw(sprite).text(
            ((self.width - lw) // 2 - label_box[0], h + pad - label_box[1]),
            visitor.com, font=font, fill=ink + (255,))
        self.image = sprite

        # Footprint: the silhouette (alpha) plus the label's full rectangle,
        # sampled down to the grid and dilated so neighbours keep a margin.
        alpha = np.asarray(sprite.getchannel("A"), dtype=np.uint8) > 24
        alpha[h:, (self.width - lw) // 2:(self.width + lw) // 2] = True
        gh, gw = math.ceil(self.height / CELL), math.ceil(self.width / CELL)
        padded = np.zeros((gh * CELL, gw * CELL), dtype=bool)
        padded[:self.height, :self.width] = alpha
        grid = padded.reshape(gh, CELL, gw, CELL).any(axis=(1, 3))
        fp = np.zeros((gh + 2 * GAP, gw + 2 * GAP), dtype=bool)
        for dy in range(2 * GAP + 1):
            for dx in range(2 * GAP + 1):
                if (dy - GAP) ** 2 + (dx - GAP) ** 2 <= GAP * GAP:
                    fp[dy:dy + gh, dx:dx + gw] |= grid
        self.footprint = fp


def _pack(sprites: list[_Sprite], gw: int, gh: int, seed: int) -> list[tuple[int, int]] | None:
    occ = np.zeros((gh, gw), dtype=bool)
    rng = np.random.default_rng(seed)
    placed: list[tuple[int, int]] = []
    cy, cx = gh / 2, gw / 2
    for sp in sprites:
        fh, fw = sp.footprint.shape
        if fh > gh or fw > gw:
            return None
        ys, xs = np.mgrid[0:gh - fh + 1:2, 0:gw - fw + 1:2]
        # Distance from the sprite's centre to the canvas centre, normalised
        # to the canvas aspect so birds spread wide rather than in a disc.
        dist = (((ys + fh / 2 - cy) / gh) ** 2 + ((xs + fw / 2 - cx) / gw) ** 2)
        dist = dist + rng.random(dist.shape) * 0.004  # organic, but stable per day
        order = np.argsort(dist, axis=None)
        sat = np.pad(occ.cumsum(0).cumsum(1), ((1, 0), (1, 0)))  # summed-area table
        spot = None
        for idx in order:
            y, x = int(ys.flat[idx]), int(xs.flat[idx])
            box = sat[y + fh, x + fw] - sat[y, x + fw] - sat[y + fh, x] + sat[y, x]
            if box == 0 or not (occ[y:y + fh, x:x + fw] & sp.footprint).any():
                spot = (y, x)
                break
        if spot is None:
            return None
        y, x = spot
        occ[y:y + fh, x:x + fw] |= sp.footprint
        placed.append((x * CELL + GAP * CELL, y * CELL + GAP * CELL))
    return placed


def render(visitors: list[Visitor], *, day: date, width: int = 1200, height: int = 800,
           title: str = "Avian Visitors", font_path: Path | None = None,
           dark: bool = False) -> bytes:
    pal = DARK if dark else LIGHT
    canvas = Image.new("RGBA", (width, height), pal["paper"] + (255,))
    draw = ImageDraw.Draw(canvas)

    margin = max(16, width // 40)
    head = _font(font_path, max(28, height // 14))
    sub = _font(font_path, max(18, height // 30))
    draw.text((margin, margin), title, font=head, fill=pal["ink"] + (255,))
    total = sum(v.count for v in visitors)
    stamp = f"{day:%A} {day.day} {day:%B}"
    if visitors:
        stamp += f"  ·  {len(visitors)} species, {total} detection{'s' if total != 1 else ''}"
    sw = draw.textlength(stamp, font=sub)
    head_h = head.getbbox(title)[3]
    draw.text((width - margin - sw, margin + head_h - sub.getbbox(stamp)[3]), stamp,
              font=sub, fill=pal["soft"] + (255,))

    top = margin + head_h + margin
    field_w, field_h = width - 2 * margin, height - top - margin
    if not visitors:
        msg = "No visitors yet today"
        big = _font(font_path, max(32, height // 12))
        mw = draw.textlength(msg, font=big)
        draw.text(((width - mw) / 2, top + field_h / 2 - big.size / 2), msg,
                  font=big, fill=pal["soft"] + (255,))
        return _png(canvas)

    birds: list[tuple[Visitor, Image.Image]] = []
    for v in visitors:
        with Image.open(v.image) as im:
            im = im.convert("RGBA")
            bbox = im.getchannel("A").point(lambda a: 255 if a > 24 else 0).getbbox()
            birds.append((v, im.crop(bbox) if bbox else im))

    weights = [min(1.7, 1 + 0.3 * math.log2(max(1, v.count))) for v, _ in birds]
    # Start generous (birds would cover ~3/4 of the field) and shrink until it packs.
    edge = math.sqrt(0.75 * field_w * field_h / sum(w * w * 0.75 for w in weights))
    edge = min(edge, field_h * 0.8)
    gw, gh = field_w // CELL, field_h // CELL
    seed = _seed(f"{day}")
    for _ in range(30):
        sprites = [_Sprite(v, im, max(24, int(edge * w)), font_path, pal["ink"])
                   for (v, im), w in zip(birds, weights)]
        order = sorted(range(len(sprites)), key=lambda i: -sprites[i].footprint.sum())
        spots = _pack([sprites[i] for i in order], gw, gh, seed)
        if spots is not None:
            break
        edge *= 0.9
    else:
        raise RuntimeError("collage did not pack")

    # Centre the packed group in the field.
    boxes = [(x, y, x + sprites[i].width, y + sprites[i].height) for i, (x, y) in zip(order, spots)]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    ox = margin + (field_w - (x1 - x0)) // 2 - x0
    oy = top + (field_h - (y1 - y0)) // 2 - y0
    for i, (x, y) in zip(order, spots):
        canvas.alpha_composite(sprites[i].image, (x + ox, y + oy))
    return _png(canvas)


def _png(canvas: Image.Image) -> bytes:
    buf = io.BytesIO()
    canvas.convert("RGB").save(buf, format="PNG", optimize=True)
    return buf.getvalue()
