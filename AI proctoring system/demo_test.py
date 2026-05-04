"""
Demo / Test Script
Tests all modules without a webcam using synthetic frames.
Run: python demo_test.py
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def create_test_frame(width=640, height=480):
    """Create a blank test frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (40, 40, 50)
    return frame


def test_imports():
    print("\n[TEST] Checking imports...")
    errors = []

    try:
        import cv2
        print(f"  ✔ OpenCV {cv2.__version__}")
    except ImportError as e:
        errors.append(f"OpenCV: {e}")
        print(f"  ✖ OpenCV: {e}")

    try:
        import mediapipe as mp
        print(f"  ✔ MediaPipe {mp.__version__}")
    except ImportError as e:
        errors.append(f"MediaPipe: {e}")
        print(f"  ✖ MediaPipe: {e}")

    try:
        import numpy as np
        print(f"  ✔ NumPy {np.__version__}")
    except ImportError as e:
        errors.append(f"NumPy: {e}")
        print(f"  ✖ NumPy: {e}")

    try:
        from ultralytics import YOLO
        print(f"  ✔ Ultralytics (YOLOv8)")
    except ImportError as e:
        print(f"  ⚠ Ultralytics not found (YOLO detection disabled): {e}")

    return len(errors) == 0


def test_detectors():
    print("\n[TEST] Testing detector modules...")
    frame = create_test_frame()

    # Test face detector
    try:
        from detectors.face_detector import FaceDetector
        fd = FaceDetector()
        result = fd.detect(frame)
        print(f"  ✔ FaceDetector: face_detected={result['face_detected']}")
    except Exception as e:
        print(f"  ✖ FaceDetector: {e}")

    # Test eye tracker
    try:
        from detectors.eye_tracker import EyeTracker
        et = EyeTracker()
        result = et.analyze(frame, {"face_detected": False, "primary_landmarks": None})
        print(f"  ✔ EyeTracker: gaze={result['gaze_direction']}")
    except Exception as e:
        print(f"  ✖ EyeTracker: {e}")

    # Test blink detector
    try:
        from detectors.blink_detector import BlinkDetector
        bd = BlinkDetector()
        result = bd.analyze(frame, {"face_detected": False, "primary_landmarks": None})
        print(f"  ✔ BlinkDetector: abnormal={result['abnormal_blink']}")
    except Exception as e:
        print(f"  ✖ BlinkDetector: {e}")

    # Test mouth detector
    try:
        from detectors.mouth_detector import MouthDetector
        md = MouthDetector()
        result = md.analyze(frame, {"face_detected": False, "primary_landmarks": None})
        print(f"  ✔ MouthDetector: mouth_open={result['mouth_open']}")
    except Exception as e:
        print(f"  ✖ MouthDetector: {e}")

    # Test head pose
    try:
        from detectors.head_pose import HeadPoseEstimator
        hp = HeadPoseEstimator()
        result = hp.estimate(frame, {"face_detected": False, "primary_landmarks": None})
        print(f"  ✔ HeadPoseEstimator: direction={result['direction']}")
    except Exception as e:
        print(f"  ✖ HeadPoseEstimator: {e}")


def test_utils():
    print("\n[TEST] Testing utility modules...")

    try:
        from utils.logger import ViolationLogger
        logger = ViolationLogger(log_dir="logs", exam_id="TEST_001")
        logger.log_violation("NO_FACE")
        logger.log_violation("PHONE_DETECTED")
        stats = logger.get_stats()
        logger.generate_report()
        print(f"  ✔ ViolationLogger: stats={stats}")
    except Exception as e:
        print(f"  ✖ ViolationLogger: {e}")

    try:
        from utils.alert_manager import AlertManager
        am = AlertManager()
        am.trigger("Test alert", severity="warning")
        alerts = am.get_active_alerts()
        print(f"  ✔ AlertManager: {len(alerts)} active alert(s)")
    except Exception as e:
        print(f"  ✖ AlertManager: {e}")

    try:
        from utils.display import DisplayManager
        dm = DisplayManager()
        frame = create_test_frame()
        out = dm.render(
            frame=frame,
            face_result={"face_detected": False, "face_count": 0,
                         "multiple_faces": False, "face_boxes": [],
                         "landmarks_list": [], "primary_landmarks": None},
            eye_result={"gaze_direction": "unknown", "looking_away": False,
                        "gaze_point": None, "look_away_counter": 0},
            blink_result={"avg_ear": 0.3, "eyes_closed": False,
                          "blink_count": 0, "abnormal_blink": False, "status": "normal"},
            mouth_result={"mar": 0.0, "mouth_open": False, "talk_events": 0},
            head_result={"yaw": 0, "pitch": 0, "roll": 0,
                         "direction": "CENTER", "suspicious_pose": False},
            object_result={"detections": [], "violations": [], "person_count": 0},
            alerts=[],
            fps=30.0,
            exam_id="TEST_001",
            stats={}
        )
        print(f"  ✔ DisplayManager: rendered frame shape={out.shape}")
    except Exception as e:
        print(f"  ✖ DisplayManager: {e}")


def main():
    print("=" * 55)
    print("  AI EXAM PROCTOR - DEPENDENCY & MODULE TEST")
    print("=" * 55)

    ok = test_imports()
    test_detectors()
    test_utils()

    print("\n" + "=" * 55)
    if ok:
        print("  ✔ Core dependencies OK. Run: python main.py")
    else:
        print("  ✖ Some dependencies missing. See above.")
    print("=" * 55)


if __name__ == "__main__":
    main()
