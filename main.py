import math
import time
import cv2

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scanner import config, hud
from scanner.database import MedicineDB, pick_section
from scanner.hands import HandTracker, NullTracker

from scanner.pipeline import Scanner
from scanner.speaker import Speaker

W, H = config.FRAME_W, config.FRAME_H
# Medicine scanning area on the left
RX, RY, RW, RH = 100, 110, 480, 480

LANG_ORDER = ["ar", "en", "fr"]
TABS = getattr(hud, "TABS", ["info", "dose", "warnings"])

class App:
    def __init__(self):
        self.state = "SCAN"
        self.lang = "en"
        self.wheel_idx = LANG_ORDER.index("en")
        self.wheel_pos = float(self.wheel_idx)
        self.tab_idx = 0
        self.page_pos = 0.0
        self.scroll_target = 0.0
        self.scroll_pos = 0.0
        self.max_scroll = 0
        self.db = MedicineDB()
        self.scanner = Scanner(self.db)
        self.speaker = Speaker()

        try:
            self.hands = HandTracker()
        except Exception as e:
            print(f"Hand tracker fallback to Null: {e}")
            self.hands = NullTracker()

        self.selector = hud.DwellSelector(0.9)
        self.buttons = []
        self.pending = []
        self.last_crop = None
        self._prev_finger = None
        self._swipe_cooldown = 0.0
        self._seen_stamp = 0.0
        self._t = time.time()
        self._notice = ""
        self._notice_until = 0.0

        self._refresh_buttons()

    def _refresh_buttons(self):
        self.buttons.clear()

        if self.state == "SCAN":
            self.buttons.append(hud.Button(
                "scan", 920, 150, 260, 60,
                label="Hold to Scan", kind="glass", icon="shutter"
            ))

        elif self.state == "SELECT_LANG":
            self.buttons.append(hud.Button(
                "reset", 36, 24, 160, 44,
                label="New Scan", size=17, kind="chip", icon="back"
            ))
            cx, cy, st = hud.WHEEL_CX, hud.WHEEL_CY, hud.WHEEL_STEP
            rw, rh = hud.WHEEL_ROW_W, hud.WHEEL_ROW_H
            if self.wheel_idx > 0:
                self.buttons.append(hud.Button("wheel_prev", cx - rw // 2, cy - st - rh // 2, rw, rh, kind="hit"))
            self.buttons.append(hud.Button("wheel_pick", cx - rw // 2, cy - rh // 2, rw, rh, kind="hit"))
            if self.wheel_idx < len(LANG_ORDER) - 1:
                self.buttons.append(hud.Button("wheel_next", cx - rw // 2, cy + st - rh // 2, rw, rh, kind="hit"))

        elif self.state == "SHOW_INFO":
            # Top-left back button
            self.buttons.append(hud.Button(
                "reset", 36, 24, 160, 44,
                label="New Scan", size=17, kind="chip", icon="back"
            ))
            
            # Segmented tabs across the top of the card
            sx, sy, sw, sh = hud.seg_rect()
            for i in range(len(TABS)):
                self.buttons.append(hud.Button(f"tab_{i}", sx + i * sw // 3, sy, sw // 3, sh, kind="tab"))

            # Clean floating Listen capsule positioned at upper-right (chest level)
            self.buttons.append(hud.Button(
                "listen", 1020, 130, 200, 56,
                label="Listen", kind="primary", icon="play"
            ))

    def notify(self, text, seconds=3.0):
        self._notice = text
        self._notice_until = time.time() + seconds

    def speak_current(self):
        med = self.scanner.state.medicine
        if not med:
            return
        tab = TABS[self.tab_idx]
        if tab == "info":
            parts = [med["name"] + "."]
            if med.get("generic"):
                parts.append(med["generic"] + ".")
            uses, lang = pick_section(med, "uses", self.lang)
            parts += uses
        elif tab == "dose":
            items, lang = pick_section(med, "dose", self.lang)
            labels = getattr(config, "LABELS", {}).get("dose", {})
            parts = [labels.get(lang, "Dose") + "."] + items
        else:
            items, lang = pick_section(med, "warnings", self.lang)
            labels = getattr(config, "LABELS", {}).get("warnings", {})
            parts = [labels.get(lang, "Warnings") + "."] + items
            inter, _ = pick_section(med, "interactions", lang)
            if inter:
                inter_labels = getattr(config, "LABELS", {}).get("interactions", {})
                parts += [inter_labels.get(lang, "Interactions") + "."] + inter

        if not self.speaker.speak(" ".join(parts), lang):
            novoice = getattr(config, "LABELS", {}).get("novoice", {})
            self.notify(novoice.get(self.lang, novoice.get("en", "No voice available")))

    def _go_tab(self, idx):
        idx = max(0, min(len(TABS) - 1, idx))
        if idx == self.tab_idx:
            return
        self.tab_idx = idx
        self.scroll_target = 0.0
        self.scroll_pos = 0.0
        if self.speaker.speaking:
            self.speak_current()

    def handle(self, action):
        if action == "scan":
            if self.last_crop is not None:
                self.scanner.start(cv2.flip(self.last_crop, 1))

        elif action in ("lang_ar", "lang_en", "lang_fr"):
            self.lang = action.replace("lang_", "")
            self.state = "SHOW_INFO"
            self.tab_idx = 0
            self.page_pos = 0.0
            self.scroll_target = self.scroll_pos = 0.0
            self._refresh_buttons()

        elif action == "wheel_prev" or (action == "swipe_down" and self.state == "SELECT_LANG"):
            self.wheel_idx = max(0, self.wheel_idx - 1)
            self._refresh_buttons()

        elif action == "wheel_next" or (action == "swipe_up" and self.state == "SELECT_LANG"):
            self.wheel_idx = min(len(LANG_ORDER) - 1, self.wheel_idx + 1)
            self._refresh_buttons()

        elif action == "wheel_pick":
            self.handle("lang_" + LANG_ORDER[self.wheel_idx])

        elif action.startswith("tab_"):
            if self.state == "SHOW_INFO":
                self._go_tab(int(action.split("_")[1]))

        elif action == "swipe_left":
            if self.state == "SHOW_INFO":
                self._go_tab(self.tab_idx + 1)

        elif action == "swipe_right":
            if self.state == "SHOW_INFO":
                self._go_tab(self.tab_idx - 1)

        elif action == "swipe_up":
            if self.state == "SHOW_INFO":
                self.scroll_target = min(self.max_scroll, self.scroll_target + 120)

        elif action == "swipe_down":
            if self.state == "SHOW_INFO":
                self.scroll_target = max(0.0, self.scroll_target - 120)

        elif action == "listen":
            if self.speaker.speaking:
                self.speaker.stop()
            else:
                self.speak_current()

        elif action == "reset":
            if self.speaker.speaking:
                self.speaker.stop()
            self.state = "SCAN"
            self._refresh_buttons()

    def check_gestures(self, finger):
        now = time.time()
        if not finger or now < self._swipe_cooldown:
            self._prev_finger = finger
            return

        if self._prev_finger:
            dx = finger[0] - self._prev_finger[0]
            dy = finger[1] - self._prev_finger[1]
            if self.state == "SHOW_INFO" and abs(dx) > 55 and abs(dy) < 40:
                self.handle("swipe_left" if dx < 0 else "swipe_right")
                self._swipe_cooldown = now + 0.55
            elif self.state in ("SELECT_LANG", "SHOW_INFO") and abs(dy) > 42 and abs(dx) <= 40:
                self.handle("swipe_up" if dy < 0 else "swipe_down")
                self._swipe_cooldown = now + 0.55

        self._prev_finger = finger

    def tick(self, dt):
        k = 1.0 - math.exp(-dt * 12)
        self.wheel_pos += (self.wheel_idx - self.wheel_pos) * k
        self.page_pos += (self.tab_idx - self.page_pos) * k
        self.scroll_target = max(0.0, min(float(self.max_scroll), self.scroll_target))
        self.scroll_pos += (self.scroll_target - self.scroll_pos) * k
        for name, target in (("wheel_pos", self.wheel_idx), ("page_pos", self.tab_idx), ("scroll_pos", self.scroll_target)):
            if abs(getattr(self, name) - target) < 0.002:
                setattr(self, name, float(target))

    def draw(self, frame, finger, hover, progress):
        cv = hud.Canvas(frame)
        st = self.scanner.state
        now = time.time()

        if self.state == "SCAN":
            hud.draw_scan_phase(cv, (RX, RY, RW, RH), scanning=(st.status == "scanning"))
        elif self.state == "SELECT_LANG":
            med_name = st.medicine["name"] if st.medicine else "Medicine"
            hud.draw_language_select_phase(cv, med_name, self.wheel_pos, LANG_ORDER, hover)
        elif self.state == "SHOW_INFO":
            self.max_scroll = hud.draw_info_phase(cv, st.medicine, self.lang, self.page_pos, self.tab_idx, self.scroll_pos)

        hud.draw_island(cv)

        for b in self.buttons:
            label, icon = b.label, b.icon
            if b.id == "listen":
                speaking = self.speaker.speaking
                label = hud.t("pause" if speaking else "listen", self.lang)
                icon = "pause" if speaking else "play"
            elif b.id == "scan" and st.status == "scanning":
                label, icon = "Scanning...", "spinner"
            is_hover = (b.id == hover)
            hud.draw_button(cv, b, is_hover, progress if is_hover else 0.0, label, icon)

        if self.state == "SCAN":
            hint = getattr(config, "LABELS", {}).get("hint", {}).get("en", "Hold your hand up to the button")
            cv.text(hint, 920 + 130, 150 + 60 + 26, size=14, alpha=0.7, anchor="c")

        notice = ""
        if now < self._notice_until:
            notice = self._notice
        elif self.state == "SCAN" and st.status in ("notfound", "error") and now - st.stamp < 5.0:
            notfound_label = getattr(config, "LABELS", {}).get("notfound", {}).get("en", "Medicine not recognized")
            notice = notfound_label if st.status == "notfound" else (st.detail or "Scan error")
        hud.draw_toast(cv, notice)

        cv.flush()
        hud.draw_cursor(frame, finger, progress)

    def run(self):
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        win = "Medication Scanner"
        cv2.namedWindow(win)

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                if frame.shape[1] != W or frame.shape[0] != H:
                    frame = cv2.resize(frame, (W, H))

                self.last_crop = frame[RY:RY + RH, RX:RX + RW].copy()

                # Ignore hand inside medicine box when in SCAN state
                ignore_box = (RX, RY, RW, RH) if self.state == "SCAN" else None
                finger = self.hands.update(frame, ignore_box=ignore_box)

                self.check_gestures(finger)

                now = time.time()
                hover, progress, triggered = self.selector.update(self.buttons, finger, now)
                if triggered:
                    self.pending.append(triggered)

                while self.pending:
                    self.handle(self.pending.pop(0))

                st = self.scanner.state
                if st.status == "found" and st.stamp != self._seen_stamp:
                    self._seen_stamp = st.stamp
                    self.state = "SELECT_LANG"
                    self.wheel_idx = LANG_ORDER.index("en")
                    self.wheel_pos = float(self.wheel_idx)
                    self._refresh_buttons()

                self.tick(min(0.1, now - self._t))
                self._t = now

                self.draw(frame, finger, hover, progress)
                cv2.imshow(win, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            self.speaker.close()

if __name__ == "__main__":
    App().run()