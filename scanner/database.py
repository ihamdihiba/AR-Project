"""Local medicine database + fuzzy matching of noisy OCR text (Step 4)."""
import json
import re
import unicodedata
from rapidfuzz import fuzz, process
from . import config

SYMPTOMS_CATALOG = [
    {"id": "headache", "en": "Headache", "fr": "Maux de tête", "ar": "صداع"},
    {"id": "fever", "en": "Fever", "fr": "Fièvre", "ar": "حمى"},
    {"id": "toothache", "en": "Toothache", "fr": "Mal de dents", "ar": "ألم الأسنان"},
    {"id": "body_aches", "en": "Body Aches", "fr": "Courbatures", "ar": "آلام الجسم"},
    {"id": "joint_pain", "en": "Joint Pain", "fr": "Douleurs articulaires", "ar": "آلام المفاصل"},
    {"id": "cramps", "en": "Cramps & Spasms", "fr": "Crampes & Spasmes", "ar": "تقلصات وتشنجات"},
    {"id": "throat_infection", "en": "Sore Throat", "fr": "Maux de gorge", "ar": "التهاب الحلق"},
    {"id": "ear_pain", "en": "Ear Pain", "fr": "Douleur d'oreille", "ar": "ألم الأذن"},
    {"id": "chest_infection", "en": "Chest Infection", "fr": "Infection pulmonaire", "ar": "التهاب الصدر"},
    {"id": "high_blood_sugar", "en": "High Blood Sugar", "fr": "Hyperglycémie", "ar": "ارتفاع سكر الدم"}
]


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\u0600-\u06FF ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def pick_section(med: dict, section: str, lang: str):
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
        best_score, best_id = 0.0, None
        for chunk in self._chunks(ocr_text):
            res = process.extractOne(
                chunk, self.keys, scorer=fuzz.ratio, score_cutoff=config.FUZZY_THRESHOLD
            )
            if res and res[1] > best_score:
                best_score, best_id = res[1], self.alias_to_id[res[0]]
        return (self.meds[best_id], best_score) if best_id else (None, 0.0)

    def recommend_for_symptoms(self, selected_symptom_ids: list):
        """Find medicines matching the chosen symptoms sorted by highest match score."""
        scored = []
        target_set = set(selected_symptom_ids)
        for med in self.meds.values():
            med_syms = set(med.get("symptoms", []))
            common = target_set.intersection(med_syms)
            if common:
                scored.append((med, len(common)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored]