"""
Blink Detector Module
Detects eye blinks using Eye Aspect Ratio (EAR).
- Normal blink detection
- Abnormal blink pattern detection (too fast, too slow, eyes closed)
"""

import numpy as np
import time


# MediaPipe landmark indices for EAR calculation
# Left eye: top-bottom pairs and corner pairs
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_TOP2 = 158
LEFT_EYE_BOTTOM2 = 153
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_TOP2 = 385
RIGHT_EYE_BOTTOM2 = 380
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

EAR_THRESHOLD = 0.21          # Below this = eye closed
BLINK_CONSEC_FRAMES = 2       # Frames needed to count a blink
EYES_CLOSED_THRESHOLD = 15    # Sustained closed frames = suspicious
RAPID_BLINK_WINDOW = 5.0      # seconds
RAPID_BLINK_COUNT = 10        # blinks in window = suspicious


def _ear(landmarks, top, bottom, top2, bottom2, left, right):
    """Eye Aspect Ratio calculation."""
    v1 = np.linalg.norm(landmarks[top][:2] - landmarks[bottom][:2])
    v2 = np.linalg.norm(landmarks[top2][:2] - landmarks[bottom2][:2])
    h = np.linalg.norm(landmarks[left][:2] - landmarks[right][:2])
    if h < 1e-6:
        return 0.3
    return (v1 + v2) / (2.0 * h)


class BlinkDetector:
    def __init__(self):
        self._closed_frames = 0
        self._blink_count = 0
        self._blink_timestamps = []
        self._frame_counter = 0
        self._ear_history = []

    def analyze(self, frame, face_result: dict) -> dict:
        """
        Analyze blinks from facial landmarks.
        """
        self._frame_counter += 1
        landmarks = face_result.get("primary_landmarks")

        if landmarks is None or not face_result.get("face_detected"):
            return {
                "left_ear": 0.3,
                "right_ear": 0.3,
                "avg_ear": 0.3,
                "eyes_closed": False,
                "blink_count": self._blink_count,
                "abnormal_blink": False,
                "status": "no_face"
            }

        left_ear = _ear(landmarks,
                        LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                        LEFT_EYE_TOP2, LEFT_EYE_BOTTOM2,
                        LEFT_EYE_LEFT, LEFT_EYE_RIGHT)

        right_ear = _ear(landmarks,
                         RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                         RIGHT_EYE_TOP2, RIGHT_EYE_BOTTOM2,
                         RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)

        avg_ear = (left_ear + right_ear) / 2.0
        eyes_closed = avg_ear < EAR_THRESHOLD

        self._ear_history.append(avg_ear)
        if len(self._ear_history) > 30:
            self._ear_history.pop(0)

        # Blink detection (rising edge of EAR)
        if eyes_closed:
            self._closed_frames += 1
        else:
            if self._closed_frames >= BLINK_CONSEC_FRAMES:
                self._blink_count += 1
                self._blink_timestamps.append(time.time())
            self._closed_frames = 0

        # Prune old blink timestamps
        now = time.time()
        self._blink_timestamps = [t for t in self._blink_timestamps
                                   if now - t < RAPID_BLINK_WINDOW]

        # Abnormality detection
        sustained_closed = self._closed_frames > EYES_CLOSED_THRESHOLD
        rapid_blink = len(self._blink_timestamps) > RAPID_BLINK_COUNT

        abnormal = sustained_closed or rapid_blink

        if sustained_closed:
            status = "eyes_closed_sustained"
        elif rapid_blink:
            status = "rapid_blinking"
        elif eyes_closed:
            status = "blinking"
        else:
            status = "normal"

        return {
            "left_ear": float(left_ear),
            "right_ear": float(right_ear),
            "avg_ear": float(avg_ear),
            "eyes_closed": eyes_closed,
            "closed_frames": self._closed_frames,
            "blink_count": self._blink_count,
            "blinks_in_window": len(self._blink_timestamps),
            "abnormal_blink": abnormal,
            "status": status
        }
