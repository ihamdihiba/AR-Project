"""Local medicine database + fuzzy matching of noisy OCR text (Step 4)."""
import json
import re
import unicodedata

from rapidfuzz import fuzz, process

from . import config


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9؀-ۿ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def pick_section(med: dict, section: str, lang: str):
    """Return (items, language_actually_used). Falls back to English if a translation is missing."""
    sec = med.get(section, {})
    if sec.get(lang):
        return sec[lang], lang
    return sec.get("en", []), "en"


class MedicineDB:
    def __init__(self, path=config.DATA_FILE):
        with open(path, encoding="utf-8") as f:
            self.meds = {m["id"]: m for m in json.load(f)}
        self.alias_to_id = {}
        for m in self.meds.values():
            for alias in [m["name"], m.get("generic", "")] + m.get("aliases", []):
                n = normalize(alias)
                if n:
                    self.alias_to_id[n] = m["id"]
        self.keys = list(self.alias_to_id)

    @staticmethod
    def _chunks(text: str):
        out = set()
        for line in text.splitlines():
            n = normalize(line)
            if len(n) < 4:
                continue
            out.add(n)
            words = [w for w in n.split() if len(w) >= 3]
            for i, w in enumerate(words):
                if len(w) >= 4:
                    out.add(w)
                if i + 1 < len(words):
                    out.add(w + " " + words[i + 1])
        return out

    def find(self, ocr_text: str):
        """Return (medicine dict or None, score 0-100)."""
        best_score, best_id = 0.0, None
        for chunk in self._chunks(ocr_text):
            res = process.extractOne(chunk, self.keys, scorer=fuzz.ratio,
                                     score_cutoff=config.FUZZY_THRESHOLD)
            if res and res[1] > best_score:
                best_score, best_id = res[1], self.alias_to_id[res[0]]
        return (self.meds[best_id], best_score) if best_id else (None, 0.0)
