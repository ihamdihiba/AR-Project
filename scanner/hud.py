"""Step 5: AR HUD drawing (cards, badges, buttons, reticle) + touchless dwell selection.

Shapes are drawn with OpenCV; all text goes through Pillow because cv2.putText cannot
render Arabic or accented characters.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import config
from .database import pick_section
from .textkit import font, shape, text_width, wrap

WHITE = (255, 255, 255)
YELLOW = (0, 230, 255)
GREEN = (80, 220, 80)
GRAY = (50, 50, 50)
SECTION_COLORS = {"dose": (200, 120, 30), "warnings": (40, 60, 220), "interactions": (170, 70, 150)}  # BGR

CARD = (560, 110, 520, 470)       # x, y, w, h
TAB_H = 56
BODY_SIZE = 30
LINE_H = 42
BODY_LINES = 6


class Canvas:
    """Draw shapes immediately, queue text, and render all text in one Pillow pass at flush()."""

    def __init__(self, frame):
        self.frame = frame
        self._texts = []

    def panel(self, x, y, w, h, color, alpha=0.8):
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.frame.shape[1], x + w), min(self.frame.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            return
        roi = self.frame[y0:y1, x0:x1]
        cv2.addWeighted(np.full_like(roi, color), alpha, roi, 1 - alpha, 0, roi)

    def text(self, s, x, y, size=28, color=WHITE, anchor="l"):
        self._texts.append((s, x, y, size, color, anchor))

    def flush(self):
        if not self._texts:
            return
        img = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        for s, x, y, size, color, anchor in self._texts:
            f = font(size)
            sh = shape(s)
            w = f.getlength(sh)
            if anchor == "c":
                x -= w / 2
            elif anchor == "r":
                x -= w
            d.text((x, y), sh, font=f, fill=(color[2], color[1], color[0]),
                   stroke_width=1, stroke_fill=(0, 0, 0))
        self.frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        self._texts.clear()


@dataclass
class Button:
    id: str
    x: int
    y: int
    w: int
    h: int
    label: str = ""
    key: str = ""                 # LABELS key, refreshed when the language changes
    icon: str = ""                # up | down | left | right
    size: int = 34
    active: bool = False
    color: tuple = (60, 160, 60)  # colour when active (BGR)

    def contains(self, p) -> bool:
        return p is not None and self.x <= p[0] <= self.x + self.w and self.y <= p[1] <= self.y + self.h


class DwellSelector:
    """Touchless 'tap': hold the fingertip over a button until the ring fills."""

    REPEATABLE = {"up", "down"}

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.current = None
        self.start = 0.0
        self.armed = True

    def update(self, buttons, point, now):
        """Returns (hovered_id, progress 0-1, triggered_id or None)."""
        hit = None
        if point:
            for b in buttons:
                if b.contains(point):
                    hit = b.id
                    break
        if hit != self.current:
            self.current, self.start, self.armed = hit, now, True
        if hit is None or not self.armed:
            return hit, 0.0, None
        progress = min(1.0, (now - self.start) / self.seconds)
        if progress >= 1.0:
            if hit in self.REPEATABLE:
                self.start = now - self.seconds * 0.4   # scroll buttons keep firing while held
            else:
                self.armed = False                       # must leave the button to trigger again
            return hit, 0.0, hit
        return hit, progress, None


def _icon(frame, b: Button):
    cx, cy, s = b.x + b.w // 2, b.y + b.h // 2, min(b.w, b.h) // 4
    pts = {
        "up": [(cx - s, cy + s // 2), (cx + s, cy + s // 2), (cx, cy - s)],
        "down": [(cx - s, cy - s // 2), (cx + s, cy - s // 2), (cx, cy + s)],
        "left": [(cx + s // 2, cy - s), (cx + s // 2, cy + s), (cx - s, cy)],
        "right": [(cx - s // 2, cy - s), (cx - s // 2, cy + s), (cx + s, cy)],
    }[b.icon]
    cv2.fillPoly(frame, [np.array(pts, np.int32)], WHITE)


def draw_button(cv: Canvas, b: Button, progress=0.0, hovered=False):
    color = b.color if b.active else ((100, 100, 100) if hovered else GRAY)
    cv.panel(b.x, b.y, b.w, b.h, color, 0.85)
    cv2.rectangle(cv.frame, (b.x, b.y), (b.x + b.w, b.y + b.h), WHITE, 4 if hovered else 2)
    if progress > 0:
        cv2.rectangle(cv.frame, (b.x, b.y + b.h - 12), (b.x + int(b.w * progress), b.y + b.h), GREEN, -1)
    if b.icon:
        _icon(cv.frame, b)
    else:
        cv.text(b.label, b.x + b.w // 2, b.y + (b.h - int(b.size * 1.25)) // 2, b.size, WHITE, "c")


def draw_reticle(frame, rect, color=YELLOW):
    x, y, w, h = rect
    arm, t = 50, 6
    for cx, cy, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * arm, cy), color, t)
        cv2.line(frame, (cx, cy), (cx, cy + dy * arm), color, t)
    mx, my = x + w // 2, y + h // 2
    cv2.line(frame, (mx - 14, my), (mx + 14, my), color, 2)
    cv2.line(frame, (mx, my - 14), (mx, my + 14), color, 2)


def card_lines(med, lang, section):
    """Wrapped bullet lines for the current section -> (lines, language_used)."""
    items, text_lang = pick_section(med, section, lang)
    max_w = CARD[2] - 2 * 24 - 16
    lines = []
    for item in items:
        lines += wrap("• " + item, BODY_SIZE, max_w)
    return lines, text_lang


def draw_card(cv: Canvas, med, lang, section, scroll, idle_msg, online_note=""):
    x, y, w, h = CARD
    color = SECTION_COLORS.get(section, GRAY) if med else GRAY
    cv.panel(x, y, w, h, (15, 15, 15), 0.82)
    cv2.rectangle(cv.frame, (x, y), (x + w, y + h), color, 4)
    if not med:
        cv.text(idle_msg, x + w // 2, y + h // 2 - 20, 32, WHITE, "c")
        return
    cv.text(med["name"], x + 24, y + TAB_H + 12, 36, WHITE)
    if med.get("generic"):
        cv.text(med["generic"], x + 24, y + TAB_H + 56, 24, (200, 200, 200))

    lines, text_lang = card_lines(med, lang, section)
    rtl = text_lang == "ar"
    body_y = y + TAB_H + 100
    for i, line in enumerate(lines[scroll:scroll + BODY_LINES]):
        if rtl:
            cv.text(line, x + w - 24 - 14, body_y + i * LINE_H, BODY_SIZE, WHITE, "r")
        else:
            cv.text(line, x + 24, body_y + i * LINE_H, BODY_SIZE, WHITE)
    if len(lines) > BODY_LINES:                       # scroll bar
        track_h = BODY_LINES * LINE_H
        thumb_h = max(30, int(track_h * BODY_LINES / len(lines)))
        top = body_y + int((track_h - thumb_h) * scroll / max(1, len(lines) - BODY_LINES))
        cv2.rectangle(cv.frame, (x + w - 18, body_y), (x + w - 10, body_y + track_h), (80, 80, 80), -1)
        cv2.rectangle(cv.frame, (x + w - 18, top), (x + w - 10, top + thumb_h), color, -1)
    if online_note:
        cv.text(online_note, x + w // 2, y + h - 30, 20, (0, 200, 255), "c")
