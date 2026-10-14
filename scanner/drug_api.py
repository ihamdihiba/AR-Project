"""Online fallback using the free openFDA drug label API (listed under Health in public-apis).

No API key needed. Data is English only; see ai.py to get FR/AR versions.
"""
import re

import requests

FDA_URL = "https://api.fda.gov/drug/label.json"

_SECTION_FIELDS = {
    "dose": ["dosage_and_administration"],
    "warnings": ["boxed_warning", "warnings", "warnings_and_cautions"],
    "interactions": ["drug_interactions"],
}


def lookup(name: str):
    """Return the first openFDA label record for a brand or generic name, or None."""
    for field in ("openfda.brand_name", "openfda.generic_name"):
        try:
            r = requests.get(FDA_URL, params={"search": f'{field}:"{name}"', "limit": 1}, timeout=8)
        except requests.RequestException:
            return None
        if r.status_code == 200:
            results = r.json().get("results") or []
            if results:
                return results[0]
    return None


def _bullets(chunks, n=4, maxlen=170):
    text = " ".join(chunks or [])
    text = re.sub(r"\b[A-Z][A-Z &/]{6,}\b", " ", text)       # drop SHOUTED headings
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for s in sentences:
        s = s.strip()
        if len(s) < 15:
            continue
        out.append(s if len(s) <= maxlen else s[: maxlen - 1].rstrip() + "…")
        if len(out) >= n:
            break
    return out


def fda_text(rec: dict) -> dict:
    """Raw English excerpts per section (used to feed the AI)."""
    out = {}
    for sec, fields in _SECTION_FIELDS.items():
        chunks = []
        for f in fields:
            chunks += rec.get(f, [])
        out[sec] = " ".join(chunks)[:2500]
    return out


def to_medicine(rec: dict):
    """Convert an openFDA record into the same shape as an entry of medicines.json (English only)."""
    ofda = rec.get("openfda", {})
    name = (ofda.get("brand_name") or ofda.get("generic_name") or ["Unknown"])[0].title()
    generic = (ofda.get("generic_name") or [""])[0].title()
    med = {"id": "openfda_" + name.lower().replace(" ", "_"), "name": name, "generic": generic}
    for sec, fields in _SECTION_FIELDS.items():
        chunks = []
        for f in fields:
            chunks += rec.get(f, [])
        items = _bullets(chunks) or ["Not provided in the source. Ask your pharmacist."]
        med[sec] = {"en": items}
    return med
