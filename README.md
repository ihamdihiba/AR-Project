# Assistive AR Medication Scanner

Point a webcam at a medicine box. The app reads the box with OCR, finds the medicine in a local
database (even with smudged ink), and floats a high-contrast info card next to the box with the
**dosage**, **warnings** and **interactions**. Everything can be controlled **touchlessly**: point a
finger in mid-air at a button and hold it for about one second. The card can be read aloud in
**English, French or Arabic**, which helps elderly users.

> Information only. This is a learning project, not medical advice. Always check with a doctor or pharmacist.

## How it works (6 steps)

1. **Capture** (`opencv-python`): webcam stream with an alignment reticle.
2. **Preprocess**: grayscale, bilateral filter, adaptive threshold (`scanner/ocr_engine.py`).
3. **OCR** (`pytesseract`): reads brand name and dose.
4. **Fuzzy match** (`rapidfuzz`): matches against `data/medicines.json`, so "Dolipran 1000m" still finds "Doliprane 1000 mg".
5. **AR HUD + hand tracking** (`opencv` + `mediapipe`): floating card, tabs, language menu, scroll arrows; fingertip dwell = tap.
6. **Speech** (`pyttsx3`): offline text to speech in the selected language.

If the medicine is **not** in the JSON database: the app asks the free **openFDA** drug label API
(English text). If you also set `GEMINI_API_KEY`, an AI model turns it into simple EN/FR/AR bullets.
Online results are marked "verify with your pharmacist" on the card.

## Setup

1. Install **Python 3.10, 3.11 or 3.12** (MediaPipe does not support the newest Python versions yet).
2. Install **Tesseract OCR** (the program, not just the Python package):
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki (tick **French** in the language list during install)
   - macOS: `brew install tesseract tesseract-lang`
   - Linux: `sudo apt install tesseract-ocr tesseract-ocr-fra`
3. In the project folder (VS Code: **Terminal > New Terminal**):

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
python main.py
```

### Voices (for read-aloud)
`pyttsx3` uses the voices installed on your computer. English usually works out of the box.
For **French** and **Arabic**, add the voice packs: Windows **Settings > Time & language > Speech > Add voices**.
If a voice is missing, the app shows "No voice installed for this language" instead of speaking garbage.

### Arabic text on screen
OpenCV cannot draw Arabic, so the app uses Pillow with a font. It finds Segoe UI / Arial (Windows),
Arial Unicode (macOS) or DejaVu Sans (Linux) automatically. If text shows as squares, copy any font
that has Arabic and Latin letters (e.g. DejaVuSans.ttf) to `assets/font.ttf`.

### Optional AI fallback
Copy `.env.example` to `.env` and add your own `GEMINI_API_KEY`. Never commit `.env` (it is in `.gitignore`).

## Controls

| Action | Finger (mid-air) | Keyboard / mouse |
|---|---|---|
| Scan the box in the reticle | hold on **Scan** | SPACE |
| Choose language (also reads aloud) | hold on **English / Français / العربية** | L (next language) |
| Switch Dosage / Warnings / Interactions | hold on a tab or the ◀ ▶ buttons | N / P |
| Scroll long text | hold on ▲ ▼ | U / D |
| Read aloud / stop | hold on **Listen** | S |
| Quit | | Q or ESC |

You can also click any button with the mouse, which is handy for testing without a hand in front of the camera.

## Add more medicines
Open `data/medicines.json` and copy one entry. Fill in `name`, `aliases` (every spelling or brand you want to match),
and the `dose`, `warnings`, `interactions` lists in `en`, `fr` and `ar`. Have the content checked by a pharmacist.

## Publish to GitHub
In the VS Code terminal, inside this folder:

```
git init
git add .
git commit -m "AR medication scanner"
git branch -M main
git remote add origin https://github.com/YOUR-NAME/YOUR-REPO.git
git push -u origin main
```

## Project layout

```
main.py                 app loop (camera, hand tapping, actions)
scanner/config.py       settings and all UI text in EN / FR / AR
scanner/ocr_engine.py   preprocessing + Tesseract OCR
scanner/database.py     medicines.json + fuzzy matching
scanner/drug_api.py     openFDA online lookup
scanner/ai.py           optional Gemini fallback (FR/AR bullets)
scanner/pipeline.py     background scan thread
scanner/hands.py        MediaPipe fingertip tracking
scanner/hud.py          cards, buttons, reticle, dwell selection
scanner/speaker.py      offline speech worker
scanner/textkit.py      Arabic shaping, fonts, word wrap
data/medicines.json     local medicine database
```
