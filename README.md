# Touchless Medication AR Scanner

A computer vision project developed for our semester coursework at ENSA Tanger. The goal is to make reading medication boxes touchless and accessible using a simple webcam. 

The application scans medicine packaging, extracts the brand name offline, and matches it with a local database to show indications, dosage, and warnings. It also includes an interactive symptom checker and speaks the results aloud in Arabic, French, or English. Everything is controlled using hand gestures so the user never has to touch the keyboard or mouse.

---

## What It Does

- **Touchless UI Navigation:** Uses Google MediaPipe to track hand landmarks. Users can click buttons by hovering their index finger (dwell click) or pick up symptom cards using a pinch gesture.
- **Offline OCR Scanning:** Uses EasyOCR to read the text directly from the camera feed. To keep the camera stream moving smoothly, OCR processing runs on a background thread so the interface doesn't freeze while reading text.
- **Ignore-Hand Zone:** When you hold a medication box in front of the camera, your holding hand is inside the scan box. We filter out any landmarks detected inside that box so the scanner doesn't confuse your grip with a button click.
- **Fuzzy Name Matching:** Medication boxes often have shiny packaging, curved text, or harsh glare. We run RapidFuzz to match the detected text against our database aliases even if EasyOCR misreads a few letters.
- **Voice Output:** Uses `pyttsx3` to read instructions and warnings out loud in English, French, or Arabic.

---

## Project Structure

```text
Project AR/
├── main.py                 # Main entry point and camera display loop
├── scanner/
│   ├── config.py           # Window dimensions, coordinates, and thresholds
│   ├── database.py         # Local dictionary of medicines, symptoms, and advice
│   ├── hud.py              # Custom UI drawing routines (frosted glass overlays, buttons)
│   ├── ocr_engine.py       # Background worker running EasyOCR
│   ├── pipeline.py         # Text cleanup and RapidFuzz matching logic
│   ├── voice.py            # Text-to-speech helper
│   └── gestures/
│       └── hands.py        # MediaPipe hand tracking and pinch/dwell logic
└── requirements.txt        # Python libraries needed to run the app
```

---

## Tested Hardware & Performance

We measured real-time performance on a standard laptop running on CPU (no dedicated GPU):

- **Display & UI:** Runs at around **10–12 FPS** ($85\text{–}100\text{ ms}$ per frame). The main workload on the CPU is running the multi-pass Gaussian blur for the frosted-glass UI cards alongside hand tracking.
- **Hand Tracking:** MediaPipe takes roughly $35\text{–}50\text{ ms}$ per frame for dual-hand detection.
- **Text Recognition:** EasyOCR takes about $1.2\text{–}1.8\text{ s}$ per scan on CPU. Because it runs on a separate worker thread, the live video feed continues running without stuttering while it thinks.
- **Matching:** RapidFuzz lookup takes under $2\text{ ms}$.

---

## Setup & Running

### Requirements
- Python 3.12
- A working webcam

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ihamdihiba/Project-AR.git
   cd "Project AR"
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Run the program:**
   ```powershell
   python main.py
   ```

---

## How to Use It

1. **Scan a Medicine Box:** Hold the box inside the central viewfinder box. Hover your index finger over the **"Hold to Scan"** button for about a second. The app will extract the name, announce it, and display the dosage and safety instructions.
2. **Symptom Checker:** Open the symptoms screen. Pinch your thumb and index finger together over any symptom card to grab it, then drag and drop it into your basket to view recommended treatments.
3. **Change Language:** Hover over the language toggle at the top bar to switch between Arabic, French, and English.
4. **Quit:** Hover over the exit button on the top right, or press `ESC` / `Q` on your keyboard.

---

## Authors

- **Hiba HAMDI**
- **Maissae ACHERNAN**
- **Wiame BENACUER**
- **Riham DRIOUECH**

*National School of Applied Sciences of Tangier (ENSAT)*  