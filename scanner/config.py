import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "medicines.json"
ASSETS_DIR = BASE_DIR / "assets"

CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720


RETICLE = (80, 140, 440, 440)

LANGS = [("en", "English"), ("fr", "Français"), ("ar", "العربية")]
SECTIONS = ["dose", "warnings", "interactions"]

DWELL_SECONDS = 1.2       
FUZZY_THRESHOLD = 80       
OCR_LANG = "eng+fra"
SPEECH_RATE = 150
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

LABELS = {
    "dose": {"en": "Dosage", "fr": "Posologie", "ar": "الجرعة"},
    "warnings": {"en": "Warnings", "fr": "Alertes", "ar": "تحذيرات"},
    "interactions": {"en": "Interactions", "fr": "Interactions", "ar": "التداخلات"},
    "aim": {
        "en": "Place the medicine box in the frame",
        "fr": "Placez la boîte dans le cadre",
        "ar": "ضع علبة الدواء داخل الإطار",
    },
    "scanning": {"en": "Scanning...", "fr": "Analyse en cours...", "ar": "جارٍ التحليل..."},
    "notfound": {
        "en": "Medicine not recognised. Try again.",
        "fr": "Médicament non reconnu. Réessayez.",
        "ar": "لم يتم التعرف على الدواء. حاول مجددًا.",
    },
    "hint": {
        "en": "Point a finger at a button and hold",
        "fr": "Pointez un bouton du doigt et maintenez",
        "ar": "وجّه إصبعك نحو زر وابقَ ثابتًا",
    },
    "disclaimer": {
        "en": "Information only. Ask your doctor or pharmacist.",
        "fr": "À titre informatif. Demandez conseil à votre médecin ou pharmacien.",
        "ar": "للمعلومات فقط. استشر طبيبك أو الصيدلي.",
    },
    "scan": {"en": "Scan", "fr": "Scanner", "ar": "مسح"},
    "listen": {"en": "Listen", "fr": "Écouter", "ar": "استمع"},
    "online": {
        "en": "Online info - verify with your pharmacist",
        "fr": "Infos en ligne - vérifiez avec votre pharmacien",
        "ar": "معلومات من الإنترنت - تحقق مع الصيدلي",
    },
    "novoice": {
        "en": "No voice installed for this language",
        "fr": "Aucune voix installée pour cette langue",
        "ar": "لا يوجد صوت مثبّت لهذه اللغة",
    },
}
