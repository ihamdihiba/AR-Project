"""Text helpers: Arabic shaping, font loading and word wrapping (OpenCV can't draw Arabic)."""
import re
import sys
from functools import lru_cache

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import ImageFont

from . import config

_AR = re.compile(r"[\u0600-\u06FF]")

FONT_CANDIDATES = [
    config.ASSETS_DIR / "font.ttf",
    # Windows
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/arial.ttf",
    # macOS
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]


def has_arabic(s: str) -> bool:
    return bool(_AR.search(s))


def shape(s: str) -> str:
    """Return a string ready to draw left-to-right (Arabic reshaped + reordered)."""
    if has_arabic(s):
        return get_display(arabic_reshaper.reshape(s))
    return s


@lru_cache(maxsize=1)
def _font_path():
    from pathlib import Path

    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return str(p)
    print("WARNING: no Unicode font found. Put a font with Arabic + Latin glyphs "
          "(e.g. DejaVuSans.ttf) at assets/font.ttf", file=sys.stderr)
    return None


@lru_cache(maxsize=64)
def font(size: int):
    path = _font_path()
    if path:
        return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def text_width(s: str, size: int) -> float:
    return font(size).getlength(shape(s))


def wrap(text: str, size: int, max_w: int):
    """Greedy word wrap. Works on logical text; shaping happens at draw time."""
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if not cur or text_width(trial, size) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines
