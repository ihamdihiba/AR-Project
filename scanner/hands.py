from pathlib import Path
import math
import time
import cv2
import numpy as np
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
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._pt = None
        self.last_seen = 0.0
        self.is_pinching = False

    def update(self, frame_bgr, ignore_box=None):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(mp_image)
        now = time.time()

        if res.hand_landmarks:
            valid_fingertips = []
            valid_landmarks = []

            for hand_lms in res.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                tip = pts[8]
                wrist = pts[0]

                if ignore_box is not None:
                    bx, by, bw, bh = ignore_box
                    in_box = (bx <= tip[0] <= bx + bw and by <= tip[1] <= by + bh) or \
                             (bx <= wrist[0] <= bx + bw and by <= wrist[1] <= by + bh)
                    if in_box:
                        continue
                
                valid_fingertips.append(tip)
                valid_landmarks.append(hand_lms)

            if not valid_fingertips:
                all_tips = [int(hand_lms[8].x * w) for hand_lms in res.hand_landmarks]
                best_idx = int(np.argmax(all_tips)) if len(all_tips) > 1 else 0
                chosen_lms = res.hand_landmarks[best_idx]
                raw = (int(chosen_lms[8].x * w), int(chosen_lms[8].y * h))
            else:
                sorted_pairs = sorted(zip(valid_fingertips, valid_landmarks), key=lambda p: p[0][0], reverse=True)
                raw = sorted_pairs[0][0]
                chosen_lms = sorted_pairs[0][1]

            # Pinch detection (Thumb Tip 4 to Index Tip 8)
            t_x, t_y = int(chosen_lms[4].x * w), int(chosen_lms[4].y * h)
            i_x, i_y = int(chosen_lms[8].x * w), int(chosen_lms[8].y * h)
            pinch_dist = math.hypot(i_x - t_x, i_y - t_y)
            self.is_pinching = (pinch_dist < 44)

            a = 0.55
            if self._pt is None:
                self._pt = raw
            else:
                self._pt = (a * raw[0] + (1 - a) * self._pt[0], a * raw[1] + (1 - a) * self._pt[1])
            self.last_seen = now

        elif now - self.last_seen > 0.35:
            self._pt = None
            self.is_pinching = False

        return None if self._pt is None else (int(self._pt[0]), int(self._pt[1]))

    def close(self):
        self.detector.close()


class NullTracker:
    def __init__(self):
        self.is_pinching = False

    def update(self, frame_bgr, ignore_box=None):
        return None

    def close(self):
        pass