"""Modern iOS-style glass HUD for the touchless medicine scanner.
Clean TrueType single-pass text rendering with frosted glass panels and hand cursor.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config, textkit
from .database import pick_section

# --- Colors & Look & Layout ---
WHITE = (255, 255, 255)
ACCENT = (255, 122, 10)  # iOS blue (BGR)
INK = 24  # Dark tint
FROSTED_GLASS = (240, 240, 245)
PASTEL_LILAC = (235, 200, 215)
PASTEL_MINT = (195, 245, 185)

CARD = (290, 78, 700, 564)  # x, y, w, h
SEG_W, SEG_H = 560, 44
TABS = ["info", "dose", "warnings"]

WHEEL_CX, WHEEL_CY, WHEEL_STEP = 640, 402, 84
WHEEL_ROW_W, WHEEL_ROW_H = 380, 72

# Left-side wheel for Symptoms
SYM_WHEEL_CX, SYM_WHEEL_CY, SYM_WHEEL_STEP = 330, 390, 84
SYM_WHEEL_ROW_W, SYM_WHEEL_ROW_H = 460, 76

UI = {
    "overview": {"en": "Overview", "fr": "Aperçu", "ar": "نظرة عامة"},
    "uses": {"en": "What it's for", "fr": "À quoi ça sert", "ar": "دواعي الاستعمال"},
    "pause": {"en": "Pause", "fr": "Pause", "ar": "إيقاف مؤقت"},
    "listen": {"en": "Listen", "fr": "Écouter", "ar": "استماع"},
}


def t(key: str, lang: str) -> str:
    table = UI.get(key) or getattr(config, "LABELS", {}).get(key) or {}
    return table.get(lang) or table.get("en", "")


def tab_label(tab: str, lang: str) -> str:
    if tab == "info":
        return t("overview", lang)
    if tab == "dose":
        labels = getattr(config, "LABELS", {}).get("dose", {})
        return labels.get(lang, "Dose")
    labels = getattr(config, "LABELS", {}).get("warnings", {})
    return labels.get(lang, "Warnings")


def seg_rect():
    x, y, w, h = CARD
    return (x + (w - SEG_W) // 2, y + 22, SEG_W, SEG_H)


# --- Font Resolution ---
DIRS = [
    getattr(config, "ASSETS_DIR", Path("assets")),
    Path("C:/Windows/Fonts"),
    Path("/System/Library/Fonts/Supplemental"),
    Path("/System/Library/Fonts"),
    Path("/Library/Fonts"),
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/truetype/noto"),
]

LATIN = {
    "light": ["ui-light.ttf", "segoeuil.ttf", "seguisl.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"],
    "regular": ["ui-regular.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "NotoSans-Regular.ttf"],
    "semibold": ["ui-semibold.ttf", "seguisb.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "NotoSans-Bold.ttf"],
}

ARABIC = {
    "regular": ["segoeui.ttf", "tahoma.ttf", "arial.ttf", "DejaVuSans.ttf", "NotoSans-Regular.ttf"],
    "semibold": ["segoeuib.ttf", "tahomabd.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "NotoSans-Bold.ttf"],
}

_FONTS: dict = {}


def _find(names):
    for name in names:
        for d in DIRS:
            p = d / name
            if p.exists():
                return str(p)
    return None


def get_font(size: int, weight: str = "regular", arabic: bool = False):
    size = max(8, int(round(size)))
    key = (size, weight, arabic)
    f = _FONTS.get(key)
    if f is not None:
        return f
    if arabic:
        names = ARABIC["semibold" if weight in ("semibold", "bold") else "regular"]
    else:
        names = LATIN.get(weight, LATIN["regular"])
    path = _find(names)
    try:
        f = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.BASIC) if path else textkit.font(size)
    except Exception:
        f = textkit.font(size)
    _FONTS[key] = f
    return f


def text_w(s: str, size: int, weight: str = "regular") -> float:
    if not s:
        return 0.0
    return get_font(size, weight, textkit.has_arabic(s)).getlength(textkit.shape(s))


def fit(s: str, size: int, max_w: int, weight: str = "regular", min_size: int = 16) -> int:
    while size > min_size and text_w(s, size, weight) > max_w:
        size -= 1
    return size


# --- Glass Rendering Cache & Helpers ---
_CACHE: dict = {}


def _remember(key, value):
    if len(_CACHE) > 120:
        _CACHE.clear()
    _CACHE[key] = value
    return value


def _shape(w: int, h: int, r: int) -> dict:
    key = ("shape", w, h, r)
    s = _CACHE.get(key)
    if s is not None:
        return s
    yy, xx = np.mgrid[0:h, 0:w]
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    px = np.abs(xx + 0.5 - w / 2) - (w / 2 - r)
    py = np.abs(yy + 0.5 - h / 2) - (h / 2 - r)
    d = np.hypot(np.maximum(px, 0), np.maximum(py, 0)) + np.minimum(np.maximum(px, py), 0) - r
    cov = np.clip(0.5 - d, 0, 1).astype(np.float32)
    inner = (cov >= 0.999)[..., None]
    by, bx = np.nonzero((cov > 0.001) & (cov < 0.999))
    ring = np.clip(0.5 - d, 0, 1) * np.clip(d + 1.5, 0, 1)
    ring = ring * np.linspace(1.0, 0.35, h, dtype=np.float32)[:, None]
    ry, rx = np.nonzero(ring > 0.01)
    s = {
        "cov": cov,
        "inner": inner,
        "band": (by, bx, cov[by, bx][:, None]),
        "ring": (ry, rx, ring[ry, rx][:, None].astype(np.float32)),
    }
    return _remember(key, s)


def _paint(roi, s, src):
    np.copyto(roi, src, where=s["inner"])
    by, bx, c = s["band"]
    if len(by):
        o = roi[by, bx].astype(np.float32)
        n = src[by, bx].astype(np.float32)
        roi[by, bx] = (o * (1 - c) + n * c).astype(np.uint8)


def _edge(roi, s, strength):
    ry, rx, a = s["ring"]
    o = roi[ry, rx].astype(np.float32)
    roi[ry, rx] = (o + (255 - o) * (a * strength)).astype(np.uint8)


def _blur(roi):
    h, w = roi.shape[:2]
    small = cv2.resize(np.ascontiguousarray(roi), (max(2, w // 5), max(2, h // 5)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 4.0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _shadow(frame, x, y, w, h, r):
    pad, dy = 40, 14
    key = ("shadow", w, h, r)
    m = _CACHE.get(key)
    if m is None:
        big = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
        big[pad:pad + h, pad:pad + w] = _shape(w, h, r)["cov"]
        big = cv2.GaussianBlur(big, (0, 0), 16)
        mult = (255 * (1 - 0.40 * big)).astype(np.uint8)
        m = _remember(key, cv2.merge([mult, mult, mult]))
    H, W = frame.shape[:2]
    x0, y0 = x - pad, y - pad + dy
    x1, y1 = x0 + w + 2 * pad, y0 + h + 2 * pad
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    roi = np.ascontiguousarray(frame[cy0:cy1, cx0:cx1])
    mm = np.ascontiguousarray(m[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0])
    frame[cy0:cy1, cx0:cx1] = cv2.multiply(roi, mm, scale=1 / 255.0)


def glass(frame, x, y, w, h, r=None, tint=0.46, light=0.07, border=1.0, blur=True, shadow=False):
    x, y, w, h = int(x), int(y), int(w), int(h)
    H, W = frame.shape[:2]
    if w < 4 or h < 4 or x < 0 or y < 0 or x + w > W or y + h > H:
        return
    r = int(min(h // 2, w // 2, 34 if r is None else r))
    s = _shape(w, h, r)
    if shadow:
        _shadow(frame, x, y, w, h, r)
    roi = frame[y:y + h, x:x + w]
    base = _blur(roi) if blur else roi
    a = (1 - tint) * (1 - light) * 0.94
    b = INK * tint * (1 - light) + 255 * light
    src = cv2.convertScaleAbs(base, alpha=a, beta=b)
    _paint(roi, s, src)
    if border >= 0:
        _edge(roi, s, border * 0.55)


def solid(frame, x, y, w, h, color, r=None, hi=0.2, shadow=True):
    x, y, w, h = int(x), int(y), int(w), int(h)
    H, W = frame.shape[:2]
    if w < 4 or h < 4 or x < 0 or y < 0 or x + w > W or y + h > H:
        return
    r = int(min(h // 2, w // 2, 34 if r is None else r))
    s = _shape(w, h, r)
    if shadow:
        _shadow(frame, x, y, w, h, r)
    gkey = ("grad", w, h, color, round(hi, 2))
    src = _CACHE.get(gkey)
    if src is None:
        g = np.linspace(1, 0, h, dtype=np.float32)[:, None, None]
        col = np.array(color, np.float32)[None, None, :]
        src = _remember(gkey, (np.clip(col + (255 - col) * hi * g, 0, 255).astype(np.uint8) * np.ones((1, w, 1), np.uint8)))
    roi = frame[y:y + h, x:x + w]
    _paint(roi, s, src)
    _edge(roi, s, 0.35)


def _sweep_fill(frame, x, y, w, h, r, progress):
    fw = int(w * progress)
    if fw < 2:
        return
    cov = _shape(w, h, r)["cov"][:, :fw, None]
    roi = frame[y:y + h, x:x + fw]
    roi[:] = (roi + (255 - roi) * (0.30 * cov)).astype(np.uint8)


def _white(frame, x, y, m):
    h, w = m.shape
    roi = np.ascontiguousarray(frame[y:y + h, x:x + w])
    m3 = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    add = cv2.multiply(cv2.bitwise_not(roi), m3, scale=1 / 255.0)
    frame[y:y + h, x:x + w] = cv2.add(roi, add)


# --- Clean Single-Pass Text & Layer System ---
class Layer:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.img = Image.new("L", (w, h), 0)
        self.d = ImageDraw.Draw(self.img)
        self.shp = np.zeros((h, w), np.uint8)
        self.glow = None
        self.gd = None
        self.fade = (0, 0)

    @staticmethod
    def _put(d, s, x, y, f, fill, anchor, size):
        try:
            d.text((x, y), s, font=f, fill=fill, anchor={"l": "lm", "c": "mm", "r": "rm"}[anchor])
        except Exception:
            w = f.getlength(s)
            tx = x - (w / 2 if anchor == "c" else w if anchor == "r" else 0)
            d.text((tx, y - size / 2), s, font=f, fill=fill)

    def text(self, s, x, y, size=20, weight="regular", alpha=1.0, anchor="l", glow=0.0):
        if not s or alpha <= 0.01:
            return
        f = get_font(size, weight, textkit.has_arabic(s))
        sh = textkit.shape(s)
        ox, oy = x - self.x, y - self.y
        self._put(self.d, sh, ox, oy, f, int(255 * min(1.0, alpha)), anchor, size)
        if glow > 0:
            if self.glow is None:
                self.glow = Image.new("L", (self.w, self.h), 0)
                self.gd = ImageDraw.Draw(self.glow)
            self._put(self.gd, sh, ox, oy, f, int(255 * min(1.0, alpha * glow)), anchor, size)

    def _c(self, a):
        return int(255 * max(0.0, min(1.0, a)))

    def _p(self, p):
        return (int(round(p[0] - self.x)), int(round(p[1] - self.y)))

    def line(self, p1, p2, w=2, a=1.0):
        cv2.line(self.shp, self._p(p1), self._p(p2), self._c(a), int(w), cv2.LINE_AA)

    def circle(self, c, r, w=-1, a=1.0):
        cv2.circle(self.shp, self._p(c), int(r), self._c(a), int(w), cv2.LINE_AA)

    def arc(self, c, r, start, end, w=2, a=1.0):
        cv2.ellipse(self.shp, self._p(c), (int(r), int(r)), 0, start, end, self._c(a), int(w), cv2.LINE_AA)

    def poly(self, pts, a=1.0):
        cv2.fillConvexPoly(self.shp, np.array([self._p(p) for p in pts], np.int32), self._c(a), cv2.LINE_AA)

    def cap_line(self, p1, p2, w=2, a=1.0):
        self.line(p1, p2, w, a)
        self.circle(p1, w / 2, -1, a)
        self.circle(p2, w / 2, -1, a)


class Canvas:
    def __init__(self, frame):
        self.frame = frame
        self.H, self.W = frame.shape[:2]
        self.main = Layer(0, 0, self.W, self.H)
        self.windows = []

    def text(self, s, x, y, size=20, color=None, anchor="l", bold=False, weight=None, alpha=1.0, glow=0.0):
        self.main.text(s, x, y, size, weight or ("semibold" if bold else "regular"), alpha, anchor, glow)

    def window(self, x, y, w, h, fade_top=0, fade_bottom=0):
        L = Layer(int(x), int(y), int(w), int(h))
        L.fade = (fade_top, fade_bottom)
        self.windows.append(L)
        return L

    def flush(self):
        fr = self.frame
        g = self.main.glow
        if g is not None:
            bb = g.getbbox()
            if bb:
                pad = 36
                x0, y0 = max(bb[0] - pad, 0), max(bb[1] - pad, 0)
                x1, y1 = min(bb[2] + pad, self.W), min(bb[3] + pad, self.H)
                arr = np.asarray(g.crop((x0, y0, x1, y1)))
                arr = cv2.GaussianBlur(arr, (0, 0), 9)
                arr = cv2.convertScaleAbs(arr, alpha=2.4)
                _white(fr, x0, y0, arr)
        _white(fr, 0, 0, np.maximum(np.asarray(self.main.img), self.main.shp))
        for L in self.windows:
            a = np.maximum(np.asarray(L.img), L.shp)
            ft, fb = L.fade
            if ft or fb:
                ramp = np.ones(L.h, np.float32)
                if ft:
                    ramp[:ft] = np.linspace(0, 1, ft)
                if fb:
                    ramp[-fb:] = np.minimum(ramp[-fb:], np.linspace(1, 0, fb))
                a = (a * ramp[:, None]).astype(np.uint8)
            _white(fr, L.x, L.y, a)


@dataclass
class Button:
    id: str
    x: int
    y: int
    w: int
    h: int
    label: str = ""
    size: int = 20
    active: bool = False
    color: tuple = WHITE
    kind: str = "glass"
    icon: str = ""

    def contains(self, p) -> bool:
        return p is not None and self.x <= p[0] <= self.x + self.w and self.y <= p[1] <= self.y + self.h


class DwellSelector:
    def __init__(self, seconds: float = 0.9):
        self.seconds = seconds
        self.current = None
        self.start = 0.0
        self.armed = True

    def update(self, buttons, point, now):
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
            self.armed = False
            return hit, 0.0, hit
        return hit, progress, None


ICON_W = {"shutter": 24, "spinner": 24, "play": 18, "pause": 18, "back": 12, "check": 18, "plus": 18, "": 0}


def _icon(L: Layer, name: str, cx: float, cy: float):
    if name == "shutter":
        L.circle((cx, cy), 11, w=2)
        L.circle((cx, cy), 6)
    elif name == "spinner":
        a = int(time.time() * 520) % 360
        L.arc((cx, cy), 10, a, a + 270, w=3)
    elif name == "play":
        L.poly([(cx - 6, cy - 9), (cx - 6, cy + 9), (cx + 9, cy)])
    elif name == "pause":
        L.cap_line((cx - 5, cy - 7), (cx - 5, cy + 7), w=5)
        L.cap_line((cx + 5, cy - 7), (cx + 5, cy + 7), w=5)
    elif name == "back":
        L.cap_line((cx + 3, cy - 8), (cx - 4, cy), w=3)
        L.cap_line((cx - 4, cy), (cx + 3, cy + 8), w=3)
    elif name == "check":
        L.cap_line((cx - 7, cy), (cx - 2, cy + 6), w=3)
        L.cap_line((cx - 2, cy + 6), (cx + 7, cy - 6), w=3)


def draw_button(cv: Canvas, b: Button, hover=False, progress=0.0, label=None, icon=None):
    if b.kind in ("hit", "tab"):
        return
    fr = cv.frame
    label = b.label if label is None else label
    icon = b.icon if icon is None else icon
    r = b.h // 2
    if b.kind == "primary":
        solid(fr, b.x, b.y, b.w, b.h, ACCENT, hi=0.34 if hover else 0.22)
    elif b.kind == "chip":
        glass(fr, b.x, b.y, b.w, b.h, r=r, tint=0.6, light=0.18 if hover else 0.08, shadow=True)
    else:
        glass(fr, b.x, b.y, b.w, b.h, r=r, tint=0.46, light=0.16 if hover else 0.07, shadow=True)

    if progress > 0:
        _sweep_fill(fr, b.x, b.y, b.w, b.h, min(r, b.h // 2, b.w // 2), progress)

    size = 19 if b.size == 20 else b.size
    tw = text_w(label, size, "semibold")
    iw = ICON_W.get(icon, 0)
    gap = 10 if iw else 0
    gx = b.x + (b.w - (iw + gap + tw)) / 2
    cy = b.y + b.h / 2
    if iw:
        _icon(cv.main, icon, gx + iw / 2, cy)
    cv.main.text(label, gx + iw + gap, cy, size, "semibold", 1.0, "l")


def draw_cursor(frame, finger, progress, pinching=False):
    if not finger:
        return
    x, y = int(finger[0]), int(finger[1])
    H, W = frame.shape[:2]
    R = 34
    x0, y0, x1, y1 = max(x - R, 0), max(y - R, 0), min(x + R, W), min(y + R, H)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    over = roi.copy()
    c = (x - x0, y - y0)
    cv2.circle(over, c, 20, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.addWeighted(over, 0.16, roi, 0.84, 0, roi)
    over = roi.copy()
    cur_color = (0, 230, 255) if pinching else WHITE
    cv2.circle(over, c, 19, cur_color, 2, cv2.LINE_AA)
    cv2.circle(over, c, 7, cur_color, -1, cv2.LINE_AA)
    cv2.addWeighted(over, 0.92, roi, 0.08, 0, roi)
    if progress > 0:
        cv2.ellipse(roi, c, (19, 19), -90, 0, int(360 * progress), ACCENT, 4, cv2.LINE_AA)


# --- Screen Phases ---
def draw_island(cv: Canvas, title="Touchless Medicine AR"):
    w, h = 360, 40
    glass(cv.frame, (cv.W - w) // 2, 16, w, h, r=20, tint=0.72, light=0.0, border=0.6)
    cv.text(title, cv.W // 2, 36, size=17, weight="semibold", anchor="c")


def draw_toast(cv: Canvas, text: str):
    if not text:
        return
    size, max_w = 16, 440
    fwords = text.split()
    lines, cur = [], ""
    for wd in fwords:
        trial = (cur + " " + wd).strip()
        if cur and text_w(trial, size) > max_w:
            lines.append(cur)
            cur = wd
        else:
            cur = trial
    if cur:
        lines.append(cur)
    lines = lines[:2]
    w = int(max(text_w(l, size) for l in lines)) + 52
    h = len(lines) * 24 + 24
    x, y = cv.W - 36 - w, cv.H - 28 - h
    glass(cv.frame, x, y, w, h, r=min(h // 2, 28), tint=0.58, shadow=True)
    for i, l in enumerate(lines):
        cv.text(l, x + w // 2, y + 24 + i * 24, size=size, anchor="c")


def _dim_outside(fr, rect, r, strength):
    x, y, w, h = rect
    H, W = fr.shape[:2]
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return
    dark = cv2.convertScaleAbs(fr, alpha=1 - strength)
    cov = _shape(w, h, r)["cov"][..., None]
    roi = fr[y:y + h, x:x + w]
    dark[y:y + h, x:x + w] = (roi * cov + dark[y:y + h, x:x + w] * (1 - cov)).astype(np.uint8)
    fr[:] = dark


def _sweep_line(fr, rect, r, yy):
    x, y, w, h = rect
    y0, y1 = max(yy - 56, y), min(yy + 56, y + h)
    if y1 <= y0:
        return
    rows = np.arange(y0, y1, dtype=np.float32)
    prof = (0.55 * np.exp(-((rows - yy) / 17.0) ** 2))[:, None, None]
    cov = _shape(w, h, r)["cov"][y0 - y:y1 - y, :, None]
    roi = fr[y0:y1, x:x + w]
    roi[:] = (roi + (255 - roi) * prof * cov).astype(np.uint8)


def draw_scan_phase(cv: Canvas, rect, scanning=False, t_now=None):
    x, y, w, h = rect
    fr = cv.frame
    R = 44
    _dim_outside(fr, rect, R, 0.40)
    L = cv.main
    arm, th = 34, 5
    for (cx, cy, a0) in ((x + R, y + R, 180), (x + w - R, y + R, 270), (x + w - R, y + h - R, 0), (x + R, y + h - R, 90)):
        L.arc((cx, cy), R, a0, a0 + 90, w=1, a=0.28)
    L.line((x + R, y), (x + w - R, y), 1, 0.28)
    L.line((x + R, y + h), (x + w - R, y + h), 1, 0.28)
    L.line((x, y + R), (x, y + h - R), 1, 0.28)
    L.line((x + w, y + R), (x + w, y + h - R), 1, 0.28)
    L.arc((x + R, y + R), R, 180, 270, w=th)
    L.arc((x + w - R, y + R), R, 270, 360, w=th)
    L.arc((x + w - R, y + h - R), R, 0, 90, w=th)
    L.arc((x + R, y + h - R), R, 90, 180, w=th)
    for p1, p2 in (
        ((x + R, y), (x + R + arm, y)),
        ((x, y + R), (x, y + R + arm)),
        ((x + w - R, y), (x + w - R - arm, y)),
        ((x + w, y + R), (x + w, y + R + arm)),
        ((x + w - R, y + h), (x + w - R - arm, y + h)),
        ((x + w, y + h - R), (x + w, y + h - R - arm)),
        ((x + R, y + h), (x + R + arm, y + h)),
        ((x, y + h - R), (x, y + h - R - arm)),
    ):
        L.cap_line(p1, p2, w=th)
    if scanning:
        tt = time.time() if t_now is None else t_now
        ph = (tt * 0.9) % 2.0
        pos = ph if ph <= 1 else 2 - ph
        _sweep_line(fr, rect, R, int(y + 24 + pos * (h - 48)))
    cx = x + w // 2
    pw, ph = 440, 70
    px, py = cx - pw // 2, y + h + 18
    glass(fr, px, py, pw, ph, r=32, tint=0.46, shadow=True)
    title = getattr(config, "LABELS", {}).get("scanning", {}).get("en", "Scanning...").replace("...", "...") if scanning else "Place the medicine box here"
    cv.text(title, cx, py + 27, size=21, weight="semibold", anchor="c", glow=0.35)
    cv.text("Box, blister pack or pills with visible writing", cx, py + 51, size=14, alpha=0.65, anchor="c")


def draw_language_select_phase(cv: Canvas, med_name: str, wheel_pos: float, order, hover=None):
    fr = cv.frame
    cx = WHEEL_CX
    pw, ph = 480, 452
    px, py = cx - pw // 2, WHEEL_CY - 250
    glass(fr, px, py, pw, ph, r=46, tint=0.5, light=0.05, shadow=True)
    cv.text("Select Language", cx, py + 38, size=14, weight="semibold", alpha=0.55, anchor="c")
    size = fit(med_name, 32, pw - 90, "semibold")
    cv.text(med_name, cx, py + 78, size=size, weight="semibold", anchor="c", glow=0.4)
    glass(
        fr,
        cx - WHEEL_ROW_W // 2,
        WHEEL_CY - WHEEL_ROW_H // 2,
        WHEEL_ROW_W,
        WHEEL_ROW_H,
        r=26,
        tint=0.0,
        light=0.24 if hover == "wheel_pick" else 0.16,
        border=0.5,
        blur=False,
    )
    names = dict(getattr(config, "LANGS", [("ar", "العربية"), ("en", "English"), ("fr", "Français")]))
    for i, code in enumerate(order):
        d = i - wheel_pos
        ad = abs(d)
        if ad >= 2:
            continue
        k = min(ad, 1.0)
        alpha = 1.0 - 0.66 * k if ad <= 1 else 0.34 * (2 - ad)
        cv.text(
            names.get(code, code),
            cx,
            WHEEL_CY + d * WHEEL_STEP,
            size=int(round(44 - 14 * k)),
            weight="semibold" if ad < 0.5 else "regular",
            alpha=alpha,
            anchor="c",
            glow=0.5 * (1 - k),
        )
    cv.text("Swipe up or down, then hold on a language", cx, py + ph - 36, size=14, alpha=0.55, anchor="c")


# --- Cylindrical Wheel Symptom Selector (Matches Language Wheel Look) ---
def draw_symptom_wheel_phase(cv: Canvas, lang: str, symptoms: list, wheel_pos: float, selected: list, dragging_item: dict | None, finger: tuple | None, hover=None):
    fr = cv.frame

    # 1. Left Big Card: The Cylindrical Symptom Wheel
    cx = SYM_WHEEL_CX
    pw, ph = 520, 520
    px, py = cx - pw // 2, SYM_WHEEL_CY - 250
    glass(fr, px, py, pw, ph, r=46, tint=0.5, light=0.05, shadow=True)
    header = {"en": "Symptom Picker", "fr": "Choix des Symptômes", "ar": "اختيار الأعراض"}.get(lang, "Symptoms")
    cv.text(header, cx, py + 38, size=15, weight="semibold", alpha=0.55, anchor="c")
    cv.text("Swipe or tap to scroll • Pinch center to drag", cx, py + 68, size=13, alpha=0.55, anchor="c")

    # Center Active Glass Reticle (Focus Capsule)
    glass(
        fr,
        cx - SYM_WHEEL_ROW_W // 2,
        SYM_WHEEL_CY - SYM_WHEEL_ROW_H // 2,
        SYM_WHEEL_ROW_W,
        SYM_WHEEL_ROW_H,
        r=26,
        tint=0.0,
        light=0.24 if hover == "sym_pick" else 0.16,
        border=0.8,
        blur=False,
    )

    # Render Cylindrical items with smooth alpha & size decay
    for i, sym in enumerate(symptoms):
        d = i - wheel_pos
        ad = abs(d)
        if ad >= 2.3:
            continue
        k = min(ad, 1.0)
        alpha = 1.0 - 0.66 * k if ad <= 1 else max(0.12, 0.34 * (2.2 - ad))
        name = sym.get(lang, sym["en"])
        
        # If currently being dragged from the center, show as ghost
        if dragging_item and dragging_item["id"] == sym["id"] and ad < 0.6:
            alpha *= 0.35

        cv.text(
            name,
            cx,
            SYM_WHEEL_CY + d * SYM_WHEEL_STEP,
            size=int(round(34 - 10 * k)),
            weight="semibold" if ad < 0.5 else "regular",
            alpha=alpha,
            anchor="c",
            glow=0.5 * (1 - k),
        )

    # 2. Right Card: Selection Basket (Max 3)
    rx, ry, rw, rh = 620, py, 580, ph
    glass(fr, rx, ry, rw, rh, r=46, tint=0.5, light=0.05, shadow=True)
    header_right = {"en": f"Selected Basket ({len(selected)}/3)", "fr": f"Panier Sélectionné ({len(selected)}/3)", "ar": f"سلة الأعراض ({len(selected)}/3)"}.get(lang, "Basket")
    cv.text(header_right, rx + 40, ry + 38, size=18, weight="semibold", anchor="l", glow=0.3)
    cv.text("Drag & drop symptoms here with pinch", rx + 40, ry + 68, size=13, alpha=0.55, anchor="l")

    for i in range(3):
        by = ry + 104 + i * 115
        bw, bh = rw - 80, 92
        bx = rx + 40
        glass(fr, bx, by, bw, bh, r=24, tint=0.34, border=0.5, blur=False)
        if i < len(selected):
            item = selected[i]
            solid(fr, bx + 6, by + 6, bw - 12, bh - 12, ACCENT, hi=0.18, shadow=False)
            cv.text(item.get(lang, item["en"]), bx + 30, by + bh // 2, size=20, weight="semibold", anchor="l")
        else:
            empty_txt = {"en": "— Empty Slot —", "fr": "— Emplacement vide —", "ar": "— مكان فارغ —"}.get(lang, "Empty")
            cv.text(empty_txt, bx + bw // 2, by + bh // 2, size=17, alpha=0.35, anchor="c")

    # Dragged Ghost Floating under finger
    if dragging_item and finger:
        fx, fy = finger
        dw, dh = 260, 68
        dx, dy = fx - dw // 2, fy - dh // 2
        glass(fr, dx, dy, dw, dh, r=20, tint=0.75, light=0.3, border=1.5, shadow=True)
        cv.text(dragging_item.get(lang, dragging_item["en"]), fx, fy, size=18, weight="semibold", anchor="c", glow=0.5)


def draw_recommendations_phase(cv: Canvas, lang: str, meds: list):
    fr = cv.frame
    cx, cy, cw, ch = 200, 80, 880, 560
    glass(fr, cx, cy, cw, ch, r=46, tint=0.52, shadow=True)
    title = {"en": "Recommended Medicines", "fr": "Médicaments Recommandés", "ar": "الأدوية المقترحة"}.get(lang, "Recommendations")
    cv.text(title, cx + cw // 2, cy + 44, size=24, weight="semibold", anchor="c", glow=0.4)
    hint = {"en": "Hover over a medicine card to see complete medical info", "fr": "Pointez une carte pour voir les informations complètes", "ar": "وجه إصبعك نحو أي دواء لعرض تفاصيله"}.get(lang, "Hover to inspect")
    cv.text(hint, cx + cw // 2, cy + 74, size=14, alpha=0.6, anchor="c")

    if not meds:
        no_med = {"en": "No matching medicines found for selected symptoms.", "fr": "Aucun médicament trouvé pour ces symptômes.", "ar": "لم يتم العثور على أدوية مطابقة لهذه الأعراض."}.get(lang, "None")
        cv.text(no_med, cx + cw // 2, cy + 240, size=18, alpha=0.7, anchor="c")


# --- Tab Content & Layout ---
def _items(med, section, lang):
    try:
        items, used = pick_section(med, section, lang)
        return [i for i in items if isinstance(i, str)], used
    except Exception:
        return [], lang


def _blocks(med, lang, tab):
    out = []
    if tab == "info":
        out.append(("title", med.get("name", ""), lang))
        if med.get("generic"):
            out.append(("subtitle", med["generic"].title(), lang))
        uses, ul = _items(med, "uses", lang)
        if uses:
            out.append(("label", t("uses", lang), lang))
            out += [("item", u, ul) for u in uses]
    elif tab == "dose":
        labels = getattr(config, "LABELS", {}).get("dose", {})
        out.append(("title", labels.get(lang, "Dose"), lang))
        items, tl = _items(med, "dose", lang)
        out += [("item", i, tl) for i in items]
    else:
        labels = getattr(config, "LABELS", {}).get("warnings", {})
        out.append(("title", labels.get(lang, "Warnings"), lang))
        items, tl = _items(med, "warnings", lang)
        out += [("item", i, tl) for i in items]
        inter, il = _items(med, "interactions", lang)
        if inter:
            inter_labels = getattr(config, "LABELS", {}).get("interactions", {})
            out.append(("label", inter_labels.get(lang, "Interactions"), lang))
            out += [("item", i, il) for i in inter]
    return out


def _wrap(text, size, weight, max_w):
    f = get_font(size, weight, textkit.has_arabic(text))
    lines, cur = [], ""
    for wd in text.split():
        trial = (cur + " " + wd).strip()
        if cur and f.getlength(textkit.shape(trial)) > max_w:
            lines.append(cur)
            cur = wd
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines or [""]


class Layout:
    def __init__(self, ops, height):
        self.ops = ops
        self.height = height


def _layout(med, lang, tab, width) -> Layout:
    key = ("layout", id(med), lang, tab, width)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] is med:
        return hit[1]
    blocks = _blocks(med, lang, tab)
    ops, y = [], 0.0
    for idx, (kind, text, tl) in enumerate(blocks):
        rtl = (tl == "ar")
        if kind == "title":
            for ln in _wrap(text, 36, "semibold", width - 8):
                ops.append(("t", ln, rtl, y + 22, 36, "semibold", 1.0))
                y += 44
            y += 6
        elif kind == "subtitle":
            for ln in _wrap(text, 20, "regular", width - 8):
                ops.append(("t", ln, rtl, y + 14, 20, "regular", 0.7))
                y += 28
            y += 14
        elif kind == "label":
            y += 14
            ops.append(("t", text if rtl else text.upper(), rtl, y + 11, 14, "semibold", 0.55))
            y += 22 + 6
        else:
            y += 12
            wrapped = _wrap(text, 22, "regular", width - 36)
            for l_idx, ln in enumerate(wrapped):
                bullet = ("• " if l_idx == 0 else "  ") if not rtl else ("• " if l_idx == 0 else "")
                full_line = bullet + ln
                ops.append(("t", full_line, rtl, y + 16, 22, "regular", 0.96))
                y += 32
            y += 12
        if idx + 1 < len(blocks) and blocks[idx + 1][0] == "item":
            ops.append(("sep", y))
    lay = Layout(ops, y)
    _remember(key, (med, lay))
    return lay


def _draw_ops(win: Layer, lay: Layout, ox, oy, ww, alpha, vis_top, vis_h):
    for op in lay.ops:
        if op[0] == "sep":
            yy = oy + op[1]
            if vis_top - 4 <= yy <= vis_top + vis_h + 4:
                win.line((ox + 4, yy), (ox + ww - 4, yy), 1, 0.16 * alpha)
            continue
        _, txt, rtl, yc, size, wt, al = op
        yy = oy + yc
        if yy < vis_top - 40 or yy > vis_top + vis_h + 40:
            continue
        if rtl:
            win.text(txt, ox + ww - 4, yy, size, wt, al * alpha, "r")
        else:
            win.text(txt, ox + 4, yy, size, wt, al * alpha, "l")


def draw_info_phase(cv: Canvas, med, lang, page_pos, tab_idx, scroll):
    fr = cv.frame
    x, y, w, h = CARD
    glass(fr, x, y, w, h, r=50, tint=0.5, light=0.05, shadow=True)
    sx, sy, sw, sh = seg_rect()
    glass(fr, sx, sy, sw, sh, r=sh // 2, tint=0.30, light=0.0, border=0.4, blur=False)
    segw = (sw - 8) / 3
    glass(
        fr,
        int(sx + 4 + page_pos * segw),
        sy + 4,
        int(segw),
        sh - 8,
        r=(sh - 8) // 2,
        tint=0.0,
        light=0.26,
        border=0.8,
        blur=False,
    )
    for i, tab in enumerate(TABS):
        near = 1 - min(1.0, abs(i - page_pos))
        cv.text(
            tab_label(tab, lang),
            sx + 4 + segw * (i + 0.5),
            sy + sh / 2,
            size=17,
            weight="semibold",
            alpha=0.55 + 0.45 * near,
            anchor="c",
        )
    wx, wy, ww, wh = x + 36, y + 100, w - 72, h - 100 - 58
    win = cv.window(wx, wy, ww, wh, fade_top=10, fade_bottom=34)
    max_scroll = 0
    for i, tab in enumerate(TABS):
        d = i - page_pos
        if abs(d) >= 1:
            continue
        lay = _layout(med, lang, tab, ww)
        sc = scroll if i == tab_idx else 0
        a = 1 - min(1.0, abs(d)) ** 1.4
        _draw_ops(win, lay, wx + int(d * ww), wy - int(sc), ww, a, wy, wh)
        if i == tab_idx:
            max_scroll = max(0, int(lay.height - wh + 10))
    if max_scroll > 0:
        th = max(44, int(wh * wh / max(wh, lay.height)))
        ty = wy + int((scroll / max_scroll) * (wh - th)) if max_scroll else wy
        cv.main.cap_line((x + w - 16, ty + 2), (x + w - 16, ty + th - 2), w=4, a=0.38)
    return max_scroll