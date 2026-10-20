"""Assistive AR Medication Scanner - main app.

Controls
  Finger (mid-air): point at a button and hold ~1 second to "tap" it
  Mouse click     : tap a button (handy for testing without a hand)
  SPACE scan | S listen | L next language | N / P next / previous section | U / D scroll | Q or ESC quit
"""
import time

import cv2

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scanner import config, hud
from scanner.config import LABELS
from scanner.database import MedicineDB, pick_section
from scanner.hands import HandTracker, NullTracker

from scanner.pipeline import Scanner
from scanner.speaker import Speaker

W, H = config.FRAME_W, config.FRAME_H
RX, RY, RW, RH = config.RETICLE


class App:
    def __init__(self):
        self.lang = "en"
        self.section = "dose"
        self.scroll = 0
        self.db = MedicineDB()
        self.scanner = Scanner(self.db)
        self.speaker = Speaker()
        try:
            self.hands = HandTracker()
        except Exception as e:  # noqa: BLE001
            print("Hand tracking unavailable (", e, ") - use mouse/keyboard instead.")
            self.hands = NullTracker()
        self.selector = hud.DwellSelector(config.DWELL_SECONDS)
        self.buttons = self._build_buttons()
        self.pending = []
        self.notice, self.notice_until = "", 0.0
        self.last_crop = None
        self._seen_stamp = 0.0

    # ---------- layout ----------
    def _build_buttons(self):
        b = []
        for i, (code, name) in enumerate(config.LANGS):            # floating language menu
            b.append(hud.Button(f"lang:{code}", 1110, 110 + i * 100, 150, 80, label=name, size=26))
        cx, cy, cw, _ = hud.CARD
        tab_w = cw // 3
        for i, sec in enumerate(config.SECTIONS):                  # tabs on the card
            b.append(hud.Button(f"tab:{sec}", cx + i * tab_w, cy, tab_w if i < 2 else cw - 2 * tab_w,
                                hud.TAB_H, key=sec, size=24, color=hud.SECTION_COLORS[sec]))
        b.append(hud.Button("up", 1110, 420, 150, 70, icon="up"))
        b.append(hud.Button("down", 1110, 510, 150, 70, icon="down"))
        b.append(hud.Button("scan", 80, 625, 220, 80, key="scan", color=(0, 140, 255)))
        b.append(hud.Button("listen", 330, 625, 220, 80, key="listen", color=(0, 140, 255)))
        b.append(hud.Button("prev", 560, 625, 100, 80, icon="left"))
        b.append(hud.Button("next", 680, 625, 100, 80, icon="right"))
        return b

    # ---------- actions ----------
    def notify(self, msg, seconds=3.0):
        self.notice, self.notice_until = msg, time.time() + seconds

    def max_scroll(self):
        med = self.scanner.state.medicine
        if not med:
            return 0
        lines, _ = hud.card_lines(med, self.lang, self.section)
        return max(0, len(lines) - hud.BODY_LINES)

    def speak_current(self):
        med = self.scanner.state.medicine
        if med is None:
            text, lang = LABELS["aim"][self.lang], self.lang
        else:
            items, lang = pick_section(med, self.section, self.lang)
            text = f"{med['name']}. {LABELS[self.section][lang]}. " + " ".join(items)
        if not self.speaker.speak(text, lang):
            self.notify(LABELS["novoice"][self.lang])

    def start_scan(self):
        if self.last_crop is None:
            return
        snapshot = cv2.flip(self.last_crop, 1)     # undo the mirror so the text reads correctly
        self.scanner.start(snapshot)

    def handle(self, action):
        if action.startswith("lang:"):
            self.lang = action.split(":")[1]
            self.scroll = 0
            self.speak_current()
        elif action.startswith("tab:"):
            self.section, self.scroll = action.split(":")[1], 0
        elif action in ("prev", "next"):
            i = config.SECTIONS.index(self.section) + (1 if action == "next" else -1)
            self.section, self.scroll = config.SECTIONS[i % len(config.SECTIONS)], 0
        elif action == "up":
            self.scroll = max(0, self.scroll - 1)
        elif action == "down":
            self.scroll = min(self.max_scroll(), self.scroll + 1)
        elif action == "scan":
            self.start_scan()
        elif action == "listen":
            if self.speaker.speaking:
                self.speaker.stop()
            else:
                self.speak_current()

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for b in self.buttons:
                if b.contains((x, y)):
                    self.pending.append(b.id)
                    break

    # ---------- drawing ----------
    def draw(self, frame, finger, hover, progress):
        L = self.lang
        st = self.scanner.state
        cv = hud.Canvas(frame)

        cv.panel(0, 0, W, 52, (0, 0, 0), 0.7)
        cv.text(LABELS["disclaimer"][L], W // 2, 12, 22, hud.WHITE, "c")

        ret_color = hud.GREEN if st.status == "found" else (hud.YELLOW if st.status != "scanning" else (0, 140, 255))
        hud.draw_reticle(frame, config.RETICLE, ret_color)
        cv.text(LABELS["aim"][L], RX + RW // 2, RY - 42, 26, hud.YELLOW, "c")

        status = ""
        if st.status == "scanning":
            status = LABELS["scanning"][L]
        elif st.status == "notfound":
            status = LABELS["notfound"][L]
        elif st.status == "error":
            status = st.detail
        if time.time() < self.notice_until:
            status = self.notice
        if status:
            cv.text(status, RX + RW // 2, RY + RH + 6, 22, (0, 200, 255), "c")

        online = LABELS["online"][L] if st.medicine and st.source in ("ai", "openfda") else ""
        hud.draw_card(cv, st.medicine, L, self.section, self.scroll, LABELS["aim"][L], online)

        for b in self.buttons:
            if b.key:
                b.label = LABELS[b.key][L]
            if b.id.startswith("lang:"):
                b.active = b.id.endswith(L)
            elif b.id.startswith("tab:"):
                b.active = b.id.endswith(self.section)
            hud.draw_button(cv, b, progress if b.id == hover else 0.0, b.id == hover)

        cv.text(LABELS["hint"][L], 1030, 655, 22, hud.WHITE, "c")   # sits right of the bottom buttons
        cv.flush()

        if finger:                                      # fingertip cursor + dwell ring
            cv2.circle(frame, finger, 16, hud.YELLOW, -1)
            cv2.circle(frame, finger, 18, (0, 0, 0), 2)
            if progress > 0:
                cv2.ellipse(frame, finger, (32, 32), -90, 0, int(360 * progress), hud.GREEN, 6)

    # ---------- main loop ----------
    def run(self):
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        if not cap.isOpened():
            print("Cannot open the camera. Try CAMERA_INDEX = 1 in scanner/config.py")
            return
        win = "Medication Scanner"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self._on_mouse)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)               # mirror: moving right on screen = moving right in life
                if frame.shape[1] != W or frame.shape[0] != H:
                    frame = cv2.resize(frame, (W, H))
                self.last_crop = frame[RY:RY + RH, RX:RX + RW].copy()   # clean snapshot, before any HUD

                finger = self.hands.update(frame)
                hover, progress, triggered = self.selector.update(self.buttons, finger, time.time())
                if triggered:
                    self.pending.append(triggered)
                while self.pending:
                    self.handle(self.pending.pop(0))

                st = self.scanner.state                   # new result -> show Dosage and read it aloud
                if st.status == "found" and st.stamp != self._seen_stamp:
                    self._seen_stamp = st.stamp
                    self.section, self.scroll = "dose", 0
                    self.speak_current()
                elif st.status in ("notfound", "error"):
                    self._seen_stamp = st.stamp

                self.draw(frame, finger, hover, progress)
                cv2.imshow(win, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                keymap = {ord(" "): "scan", ord("s"): "listen", ord("n"): "next",
                          ord("p"): "prev", ord("u"): "up", ord("d"): "down"}
                if key in keymap:
                    self.pending.append(keymap[key])
                elif key == ord("l"):
                    codes = [c for c, _ in config.LANGS]
                    self.pending.append("lang:" + codes[(codes.index(self.lang) + 1) % len(codes)])
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            self.speaker.close()


if __name__ == "__main__":
    App().run()
