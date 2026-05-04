"""
Mouth Detector Module
Detects mouth opening (talking/whispering) using facial landmarks.
Uses Mouth Aspect Ratio (MAR).
"""

import numpy as np


# MediaPipe landmark indices for mouth
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_TOP2 = 312
MOUTH_BOTTOM2 = 82
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# Additional inner lip points
UPPER_LIP_TOP = 0
LOWER_LIP_BOTTOM = 17
UPPER_LIP_MID = 13
LOWER_LIP_MID = 14

MAR_THRESHOLD = 0.5           # Mouth Aspect Ratio threshold
SUSTAINED_OPEN_FRAMES = 8     # Frames mouth must be open to flag


def _mar(landmarks):
    """Mouth Aspect Ratio."""
    v1 = np.linalg.norm(landmarks[MOUTH_TOP][:2] - landmarks[MOUTH_BOTTOM][:2])
    v2 = np.linalg.norm(landmarks[UPPER_LIP_TOP][:2] - landmarks[LOWER_LIP_BOTTOM][:2])
    h = np.linalg.norm(landmarks[MOUTH_LEFT][:2] - landmarks[MOUTH_RIGHT][:2])
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


class MouthDetector:
    def __init__(self):
        self._open_frames = 0
        self._talk_events = 0
        self._mar_history = []

    def analyze(self, frame, face_result: dict) -> dict:
        """
        Analyze mouth state.
        """
        landmarks = face_result.get("primary_landmarks")

        if landmarks is None or not face_result.get("face_detected"):
            return {
                "mar": 0.0,
                "mouth_open": False,
                "talking": False,
                "talk_events": self._talk_events
            }

        mar = _mar(landmarks)
        self._mar_history.append(mar)
        if len(self._mar_history) > 30:
            self._mar_history.pop(0)

        mouth_open = mar > MAR_THRESHOLD

        if mouth_open:
            self._open_frames += 1
        else:
            if self._open_frames >= SUSTAINED_OPEN_FRAMES:
                self._talk_events += 1
            self._open_frames = 0

        # Calculate mouth openness percentage
        avg_mar = np.mean(self._mar_history) if self._mar_history else 0
        openness_pct = min(100, int((mar / 1.0) * 100))

        # Get mouth landmark points for drawing
        mouth_pts = None
        try:
            mouth_pts = np.array([
                landmarks[MOUTH_LEFT][:2],
                landmarks[MOUTH_RIGHT][:2],
                landmarks[MOUTH_TOP][:2],
                landmarks[MOUTH_BOTTOM][:2],
            ], dtype=np.int32)
        except Exception:
            pass

        return {
            "mar": float(mar),
            "avg_mar": float(avg_mar),
            "mouth_open": self._open_frames > SUSTAINED_OPEN_FRAMES,
            "talking": self._open_frames > SUSTAINED_OPEN_FRAMES,
            "open_frames": self._open_frames,
            "talk_events": self._talk_events,
            "openness_pct": openness_pct,
            "mouth_pts": mouth_pts
        }
