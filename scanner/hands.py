"""Step 5 (part): mid-air fingertip tracking with MediaPipe Hands."""
import time

import cv2


class HandTracker:
    """Returns the index fingertip position in pixels, smoothed. None if no hand is visible."""

    def __init__(self):
        import mediapipe as mp  # imported here so the app still runs (mouse/keyboard) without it

        self._hands = mp.solutions.hands.Hands(
            max_num_hands=1, model_complexity=0,
            min_detection_confidence=0.6, min_tracking_confidence=0.5,
        )
        self._pt = None
        self._last_seen = 0.0

    def update(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._hands.process(rgb)
        now = time.time()
        if res.multi_hand_landmarks:
            tip = res.multi_hand_landmarks[0].landmark[8]   # index fingertip
            raw = (tip.x * w, tip.y * h)
            if self._pt is None:
                self._pt = raw
            else:                                            # smooth the jitter (important for shaky hands)
                a = 0.5
                self._pt = (a * raw[0] + (1 - a) * self._pt[0], a * raw[1] + (1 - a) * self._pt[1])
            self._last_seen = now
        elif now - self._last_seen > 0.4:
            self._pt = None
        return None if self._pt is None else (int(self._pt[0]), int(self._pt[1]))

    def close(self):
        self._hands.close()


class NullTracker:
    """Used when mediapipe is not installed: the app falls back to mouse + keyboard."""

    def update(self, frame_bgr):
        return None

    def close(self):
        pass
