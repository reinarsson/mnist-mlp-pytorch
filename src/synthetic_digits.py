"""Synthetic printed-digit generator.

Sudoku puzzles use printed fonts, but MNIST is handwritten -- a domain gap
that causes systematic misreads in sudokureader (e.g. a printed "7" without
a crossbar reads as a handwritten "2"). This module renders digits 0-9 in
common sans-serif fonts, with random size/rotation/position jitter, into
MNIST-format images (IMAGE_SIZE x IMAGE_SIZE float32 in [0, 1], white digit
on black, centered) so they can be mixed into training to close that gap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMAGE_SIZE = 28
DIGIT_PAD = 4
RENDER_SIZE = 128         # render large, then downscale -- avoids jagged glyphs
FONT_SIZES = (60, 70, 80, 90)
MAX_ROTATION_DEG = 12.0
MAX_SHIFT_PX = 2

MAX_FONTS = 16

# Searched in order; covers Windows dev machines and Linux CI runners (where
# fonts-dejavu-core / fonts-liberation are preinstalled under /usr/share/fonts).
FONT_SEARCH_DIRS = (
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
)

# Preferred for stylistic variety when present (mostly Windows system fonts);
# otherwise we fall back to whatever .ttf/.otf files we can find.
PREFERRED_FONT_NAMES = (
    "arial.ttf", "arialbd.ttf",
    "calibri.ttf", "calibrib.ttf",
    "verdana.ttf", "verdanab.ttf",
    "tahoma.ttf", "tahomabd.ttf",
    "segoeui.ttf", "segoeuib.ttf",
    "consola.ttf", "couri.ttf",
    "timesbd.ttf",
    "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf",
    "LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf",
)


def _discover_fonts() -> tuple[Path, ...]:
    """Locate usable TrueType/OpenType font files on this machine.

    Returns:
        Paths to font files -- PREFERRED_FONT_NAMES when any are present
        (for stylistic variety), otherwise up to MAX_FONTS of whatever
        .ttf/.otf files are found under FONT_SEARCH_DIRS.
    """
    by_name: dict[str, Path] = {}
    for directory in FONT_SEARCH_DIRS:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.suffix.lower() in (".ttf", ".otf"):
                by_name.setdefault(path.name, path)

    preferred = tuple(by_name[name] for name in PREFERRED_FONT_NAMES if name in by_name)
    if preferred:
        return preferred
    return tuple(sorted(by_name.values(), key=lambda p: p.name)[:MAX_FONTS])


FONT_PATHS = _discover_fonts()


def _render_digit(
    digit: int, font_path: Path, font_size: int, rng: np.random.Generator
) -> np.ndarray | None:
    """Render one digit centered on an IMAGE_SIZE x IMAGE_SIZE canvas, MNIST-style.

    Args:
        digit: The digit (0-9) to render.
        font_path: Path to a TrueType font file.
        font_size: Font size in points, applied on the RENDER_SIZE canvas.
        rng: Random generator driving rotation and position jitter.

    Returns:
        A float32 array of shape (IMAGE_SIZE, IMAGE_SIZE) in [0, 1] -- white
        ink on a black background, like a preprocessed sudokureader cell --
        or None if the glyph rendered no ink (degenerate font/size pairing).
    """
    font = ImageFont.truetype(str(font_path), font_size)
    canvas = Image.new("L", (RENDER_SIZE, RENDER_SIZE), color=0)
    draw = ImageDraw.Draw(canvas)
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pos = ((RENDER_SIZE - w) // 2 - bbox[0], (RENDER_SIZE - h) // 2 - bbox[1])
    draw.text(pos, text, fill=255, font=font)

    angle = float(rng.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG))
    canvas = canvas.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

    ink = np.asarray(canvas)
    ys, xs = np.nonzero(ink)
    if ys.size == 0:
        return None
    glyph = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    target = IMAGE_SIZE - 2 * DIGIT_PAD
    scale = target / max(glyph.shape)
    nh, nw = max(1, int(glyph.shape[0] * scale)), max(1, int(glyph.shape[1] * scale))
    glyph = np.asarray(Image.fromarray(glyph).resize((nw, nh), resample=Image.BILINEAR))

    dx = int(rng.integers(-MAX_SHIFT_PX, MAX_SHIFT_PX + 1))
    dy = int(rng.integers(-MAX_SHIFT_PX, MAX_SHIFT_PX + 1))
    ox = min(max((IMAGE_SIZE - nw) // 2 + dx, 0), IMAGE_SIZE - nw)
    oy = min(max((IMAGE_SIZE - nh) // 2 + dy, 0), IMAGE_SIZE - nh)

    out = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    out[oy:oy + nh, ox:ox + nw] = glyph
    return out.astype(np.float32) / 255.0


def generate_dataset(samples_per_digit: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Render a synthetic printed-digit dataset, balanced across digits 0-9.

    Args:
        samples_per_digit: How many images to render for each digit.
        seed: Seed controlling font/size/rotation/position jitter.

    Returns:
        (images, labels): images has shape (N, IMAGE_SIZE, IMAGE_SIZE) float32
        in [0, 1]; labels has shape (N,) with int64 values 0-9.
    """
    if not FONT_PATHS:
        raise FileNotFoundError(f"No usable fonts found under any of {FONT_SEARCH_DIRS}")

    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    labels: list[int] = []
    for digit in range(10):
        n = 0
        while n < samples_per_digit:
            font_path = FONT_PATHS[int(rng.integers(len(FONT_PATHS)))]
            font_size = int(rng.choice(FONT_SIZES))
            img = _render_digit(digit, font_path, font_size, rng)
            if img is None:
                continue
            images.append(img)
            labels.append(digit)
            n += 1

    return np.stack(images), np.array(labels, dtype=np.int64)
