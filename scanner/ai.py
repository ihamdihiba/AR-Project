"""Optional AI fallback (Gemini): identifies a medicine from OCR text and writes simple EN/FR/AR bullets.

Enabled only if GEMINI_API_KEY is set. Output is flagged as "online info" in the HUD
because AI text must always be verified with a pharmacist.
"""
from __future__ import annotations

import json
import os

from . import config

SECTIONS = ("dose", "warnings", "interactions")
LANGS = ("en", "fr", "ar")


def available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def _valid(d) -> bool:
    if not isinstance(d, dict) or not d.get("name"):
        return False
    for sec in SECTIONS:
        block = d.get(sec)
        if not isinstance(block, dict):
            return False
        for lang in LANGS:
            items = block.get(lang)
            if not isinstance(items, list) or not items or not all(isinstance(i, str) for i in items):
                return False
    return True


def build_medicine(ocr_text: str, fda_rec_text: dict | None = None):
    """Return a medicine dict (same shape as medicines.json) or None."""
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    source = ""
    if fda_rec_text:
        source = "\n\nOFFICIAL LABEL EXCERPTS (English):\n" + json.dumps(fda_rec_text, ensure_ascii=False)
    prompt = (
        "You help elderly people understand their medicines.\n"
        "Below is noisy OCR text read from a medicine box. Identify the medicine. "
        "If you cannot identify a real medicine with confidence, return {\"name\": null}.\n"
        "Otherwise return ONLY JSON with this exact shape:\n"
        '{"name": str, "generic": str, '
        '"dose": {"en": [str], "fr": [str], "ar": [str]}, '
        '"warnings": {"en": [str], "fr": [str], "ar": [str]}, '
        '"interactions": {"en": [str], "fr": [str], "ar": [str]}}\n'
        "Rules: 2 to 4 short, simple bullets per section and language (max 20 words each). "
        "Base doses and warnings only on the label excerpts or on well-established standard "
        "adult information for that exact medicine and strength; never invent numbers. "
        "If a dose is unknown, say to follow the doctor's or pharmacist's instructions."
        f"\n\nOCR TEXT:\n{ocr_text[:1500]}{source}"
    )
    resp = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    data = json.loads(resp.text)
    if not _valid(data):
        return None
    data["id"] = "ai_" + str(data["name"]).lower().replace(" ", "_")
    data.setdefault("generic", "")
    return data
