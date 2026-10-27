
import re
import cv2
import easyocr


_reader = easyocr.Reader(['en', 'fr'], gpu=False)

_STOPWORDS = {
    "tablet", "tablets", "comprime", "comprimes", "boite", "capsule", "capsules", 
    "dose", "doses", "pharmacie", "laboratoire", "laboratories", "medicament", 
    "usage", "orale", "voie", "adult", "adults", "adulte", "adultes", "store", 
    "conserver", "keep", "children", "enfants", "film", "pack", "box"
}

def read_text(snapshot) -> str:
    rgb = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
    results = _reader.readtext(rgb, detail=0, paragraph=True)
    extracted = " ".join(results)
    print(f"\n[OCR DEBUG] Detected Raw Text: '{extracted}'")
    return extracted

def candidate_names(text: str, limit: int = 6):
    seen, out = set(), []
    for tok in re.findall(r"[A-Za-zÀ-ÿ]{4,}", text):
        low = tok.lower()
        if low in _STOPWORDS or low in seen:
            continue
        seen.add(low)
        out.append(tok)
        if len(out) >= limit:
            break
    return out