"""Touchless hand tracking using MediaPipe Tasks API with multi-hand arbitration."""
from pathlib import Path
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self):
        model_path = Path(__file__).resolve().parent.parent / "hand_landmarker.task"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing at: {model_path}. Run the download command.")

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,  # Track both hands so MediaPipe does not flicker between them
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._pt = None
        self._last_seen = 0.0

    def update(self, frame_bgr, ignore_box=None):
        """
        Processes frame and returns index fingertip coordinates.
        If ignore_box (x, y, w, h) is provided, hands holding the medicine inside that box are ignored.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(mp_image)
        now = time.time()

        if res.hand_landmarks:
            valid_fingertips = []

            for hand_lms in res.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                tip = pts[8]   # Index fingertip
                wrist = pts[0] # Wrist


                if ignore_box is not None:
                    bx, by, bw, bh = ignore_box
                    
                    in_box = (bx <= tip[0] <= bx + bw and by <= tip[1] <= by + bh) or \
                             (bx <= wrist[0] <= bx + bw and by <= wrist[1] <= by + bh)
                    if in_box:
                        continue

                valid_fingertips.append(tip)

           
            if not valid_fingertips:
                all_tips = [int(hand_lms[8].x * w) for hand_lms in res.hand_landmarks]
                best_idx = int(np.argmax(all_tips)) if len(all_tips) > 1 else 0
                raw = (int(res.hand_landmarks[best_idx][8].x * w), int(res.hand_landmarks[best_idx][8].y * h))
            else:
               
                valid_fingertips.sort(key=lambda p: p[0], reverse=True)
                raw = valid_fingertips[0]

            if self._pt is None:
                self._pt = raw
            else:
                a = 0.55
                self._pt = (a * raw[0] + (1 - a) * self._pt[0], a * raw[1] + (1 - a) * self._pt[1])
            self._last_seen = now
        elif now - self._last_seen > 0.35:
            self._pt = None

        return None if self._pt is None else (int(self._pt[0]), int(self._pt[1]))

    def close(self):
        self.detector.close()


class NullTracker:
    def update(self, frame_bgr, ignore_box=None):
        return None

    def close(self):
        pass