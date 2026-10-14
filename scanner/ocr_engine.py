"""Steps 2-3: snapshot preprocessing and OCR."""
import os
import re
import shutil

import cv2
import pytesseract

from . import config

if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
elif not shutil.which("tesseract"):
    for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            break

_STOPWORDS = {
    "tablet", "tablets", "comprime", "comprimes", "boite", "capsule", "capsules", "dose", "doses",
    "pharmacie", "laboratoire", "laboratories", "medicament", "usage", "orale", "voie", "adult",
    "adults", "adulte", "adultes", "store", "conserver", "keep", "children", "enfants", "film",
    "pelliculé", "pellicule", "each", "contains", "contient", "pack", "box", "made", "from",
}


def preprocess(snapshot):
    """BGR crop -> (grayscale+bilateral image, black/white thresholded image)."""
    gray = cv2.cvtColor(snapshot, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)       # smooth texture, keep letter edges
    bw = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)  # pure black / pure white
    return smooth, bw


def read_text(snapshot) -> str:
    smooth, bw = preprocess(snapshot)
    cfg = "--oem 3 --psm 11"
    texts = []
    for img in (bw, smooth):
        try:
            texts.append(pytesseract.image_to_string(img, lang=config.OCR_LANG, config=cfg))
        except pytesseract.TesseractError:
            # French language data missing -> fall back to English only
            texts.append(pytesseract.image_to_string(img, lang="eng", config=cfg))
    return "\n".join(texts)


def candidate_names(text: str, limit: int = 6):
    """Word-like tokens that could be a brand/generic name (used for the online lookup)."""
    seen, out = set(), []
    for tok in re.findall(r"[A-Za-zÀ-ÿ]{5,}", text):
        low = tok.lower()
        if low in _STOPWORDS or low in seen:
            continue
        seen.add(low)
        out.append(tok)
        if len(out) >= limit:
            break
    return out
