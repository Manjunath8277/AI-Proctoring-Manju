"""
Eye Tracker Module
Tracks eye gaze direction using MediaPipe facial landmarks.
- Gaze direction estimation (left, right, center, up, down)
- Looking-away detection
"""

import numpy as np
import cv2


# MediaPipe FaceMesh landmark indices for iris and eye corners
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_CORNER_L = 33
LEFT_EYE_CORNER_R = 133
RIGHT_EYE_CORNER_L = 362
RIGHT_EYE_CORNER_R = 263


class EyeTracker:
    def __init__(self):
        self._look_away_counter = 0
        self._gaze_history = []
        self._history_max = 10

    def _get_iris_position(self, landmarks, iris_ids, corner_l, corner_r):
        """
        Returns normalized horizontal and vertical iris position [0,1]
        relative to the eye bounding box.
        """
        iris = np.mean([landmarks[i][:2] for i in iris_ids], axis=0)
        corner_left = landmarks[corner_l][:2]
        corner_right = landmarks[corner_r][:2]

        eye_width = np.linalg.norm(corner_right - corner_left)
        if eye_width < 1e-6:
            return 0.5, 0.5

        # Horizontal ratio
        h_ratio = np.linalg.norm(iris - corner_left) / eye_width

        # Vertical: use top and bottom landmarks of the eye
        return h_ratio, 0.5  # vertical gaze simplified

    def analyze(self, frame: np.ndarray, face_result: dict) -> dict:
        """
        Analyze gaze direction.
        Returns dict with gaze info.
        """
        landmarks = face_result.get("primary_landmarks")

        if landmarks is None or not face_result.get("face_detected"):
            return {
                "gaze_direction": "unknown",
                "looking_away": False,
                "left_ratio": 0.5,
                "right_ratio": 0.5,
                "gaze_point": None
            }

        # Has iris landmarks (refine_landmarks=True needed)
        has_iris = len(landmarks) > 472

        if has_iris:
            l_ratio, _ = self._get_iris_position(
                landmarks, LEFT_IRIS, LEFT_EYE_CORNER_L, LEFT_EYE_CORNER_R
            )
            r_ratio, _ = self._get_iris_position(
                landmarks, RIGHT_IRIS, RIGHT_EYE_CORNER_L, RIGHT_EYE_CORNER_R
            )
        else:
            l_ratio, r_ratio = 0.5, 0.5

        avg_ratio = (l_ratio + r_ratio) / 2.0

        # Determine gaze direction
        if avg_ratio < 0.35:
            gaze_direction = "LEFT"
            looking_away = True
        elif avg_ratio > 0.65:
            gaze_direction = "RIGHT"
            looking_away = True
        else:
            gaze_direction = "CENTER"
            looking_away = False

        # Smooth over history
        self._gaze_history.append(looking_away)
        if len(self._gaze_history) > self._history_max:
            self._gaze_history.pop(0)

        sustained_look_away = sum(self._gaze_history) > (self._history_max * 0.6)

        if sustained_look_away:
            self._look_away_counter = min(self._look_away_counter + 1, 60)
        else:
            self._look_away_counter = max(self._look_away_counter - 1, 0)

        # Gaze point estimation (center of both irises)
        if has_iris:
            left_iris_pt = np.mean([landmarks[i][:2] for i in LEFT_IRIS], axis=0)
            right_iris_pt = np.mean([landmarks[i][:2] for i in RIGHT_IRIS], axis=0)
            gaze_point = tuple(((left_iris_pt + right_iris_pt) / 2).astype(int))
        else:
            gaze_point = None

        return {
            "gaze_direction": gaze_direction,
            "looking_away": self._look_away_counter > 15,
            "left_ratio": float(l_ratio),
            "right_ratio": float(r_ratio),
            "avg_ratio": float(avg_ratio),
            "gaze_point": gaze_point,
            "look_away_counter": self._look_away_counter
        }
