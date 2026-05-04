"""
Head Pose Estimator Module
Estimates head orientation (yaw, pitch, roll) using facial landmarks
and OpenCV solvePnP.
- Detects if head is turned left/right/up/down
- Flags suspicious head movements
"""

import numpy as np
import cv2


# 3D model points for head pose (standard face model)
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),     # Nose tip (1)
    (0.0,   -330.0, -65.0),    # Chin (152)
    (-225.0, 170.0, -135.0),   # Left eye left corner (33)
    (225.0,  170.0, -135.0),   # Right eye right corner (263)
    (-150.0, -150.0, -125.0),  # Left mouth corner (61)
    (150.0,  -150.0, -125.0),  # Right mouth corner (291)
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
LM_INDICES = [1, 152, 33, 263, 61, 291]

# Thresholds (degrees)
YAW_THRESHOLD = 20     # Left/Right rotation
PITCH_THRESHOLD = 15   # Up/Down tilt
ROLL_THRESHOLD = 20    # Head tilt


class HeadPoseEstimator:
    def __init__(self):
        self._suspicious_counter = 0
        self._pose_history = []

    def estimate(self, frame: np.ndarray, face_result: dict) -> dict:
        """
        Estimate head pose and detect suspicious orientation.
        """
        landmarks = face_result.get("primary_landmarks")
        h, w = frame.shape[:2]

        if landmarks is None or not face_result.get("face_detected"):
            return {
                "yaw": 0, "pitch": 0, "roll": 0,
                "direction": "unknown",
                "suspicious_pose": False,
                "nose_point": None
            }

        # Extract 2D landmark points
        image_points = np.array([
            landmarks[idx][:2] for idx in LM_INDICES
        ], dtype=np.float64)

        # Camera internals (estimated)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vec, translation_vec = cv2.solvePnP(
            MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {
                "yaw": 0, "pitch": 0, "roll": 0,
                "direction": "unknown",
                "suspicious_pose": False,
                "nose_point": None
            }

        # Convert rotation vector to angles
        rmat, _ = cv2.Rodrigues(rotation_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = angles[0]  # Up/Down
        yaw   = angles[1]  # Left/Right
        roll  = angles[2]  # Tilt

        # Determine direction
        direction = "CENTER"
        suspicious = False

        if abs(yaw) > YAW_THRESHOLD:
            direction = "LEFT" if yaw < 0 else "RIGHT"
            suspicious = True
        elif abs(pitch) > PITCH_THRESHOLD:
            direction = "UP" if pitch > 0 else "DOWN"
            suspicious = True
        elif abs(roll) > ROLL_THRESHOLD:
            direction = "TILTED"
            suspicious = True

        # Smooth suspicious detection
        self._pose_history.append(suspicious)
        if len(self._pose_history) > 15:
            self._pose_history.pop(0)
        sustained_suspicious = sum(self._pose_history) > 8

        # Nose projection for drawing
        nose_end_point2d = None
        try:
            nose_end = np.array([[0.0, 0.0, 1000.0]])
            nose_proj, _ = cv2.projectPoints(
                nose_end, rotation_vec, translation_vec, camera_matrix, dist_coeffs
            )
            nose_tip = tuple(image_points[0].astype(int))
            nose_end_pt = tuple(nose_proj[0][0].astype(int))
            nose_end_point2d = (nose_tip, nose_end_pt)
        except Exception:
            pass

        return {
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "direction": direction,
            "suspicious_pose": sustained_suspicious,
            "nose_direction_line": nose_end_point2d,
            "rotation_vec": rotation_vec,
            "translation_vec": translation_vec
        }
