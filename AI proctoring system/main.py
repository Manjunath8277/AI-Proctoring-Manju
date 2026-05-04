"""
AI-Based Online Exam Proctoring System
Main entry point - Run this file to start proctoring
"""

import cv2
import time
import argparse
import sys
from datetime import datetime

from detectors.face_detector import FaceDetector
from detectors.eye_tracker import EyeTracker
from detectors.blink_detector import BlinkDetector
from detectors.mouth_detector import MouthDetector
from detectors.head_pose import HeadPoseEstimator
from detectors.object_detector import ObjectDetector
from utils.logger import ViolationLogger
from utils.alert_manager import AlertManager
from utils.display import DisplayManager


def parse_args():
    parser = argparse.ArgumentParser(description="AI Exam Proctoring System")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for log files")
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO object detection")
    parser.add_argument("--no-sound", action="store_true", help="Disable audio alerts")
    parser.add_argument("--exam-id", type=str, default=f"EXAM_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help="Exam session ID")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("   AI EXAM PROCTORING SYSTEM")
    print("=" * 60)
    print(f"  Exam ID   : {args.exam_id}")
    print(f"  Camera    : {args.camera}")
    print(f"  Log Dir   : {args.log_dir}")
    print(f"  YOLO      : {'Disabled' if args.no_yolo else 'Enabled'}")
    print("=" * 60)
    print("  Press 'Q' to quit | Press 'S' to save screenshot")
    print("=" * 60)

    # Initialize logger
    logger = ViolationLogger(log_dir=args.log_dir, exam_id=args.exam_id)

    # Initialize alert manager (with sound)
    alert_manager = AlertManager(sound_enabled=not args.no_sound)

    # Initialize detectors
    print("\n[INIT] Loading detectors...")
    face_detector = FaceDetector()
    eye_tracker = EyeTracker()
    blink_detector = BlinkDetector()
    mouth_detector = MouthDetector()
    head_pose = HeadPoseEstimator()
    display_manager = DisplayManager()

    object_detector = None
    if not args.no_yolo:
        try:
            object_detector = ObjectDetector(model_path=args.yolo_model)
            print("[INIT] YOLO object detector loaded.")
        except Exception as e:
            print(f"[WARN] YOLO failed to load: {e}. Object detection disabled.")

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\n[INFO] Proctoring session started. Press Q to quit.\n")
    logger.log_event("SESSION_START", "Proctoring session started.")

    frame_count = 0
    fps_time = time.time()
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera.")
                break

            frame_count += 1

            # FPS calculation
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_time)
                fps_time = time.time()

            # ── Detection pipeline ──────────────────────────────────────
            violations = []

            # 1. Face Detection
            face_result = face_detector.detect(frame)

            # 2. Eye Tracking + Blink Detection
            eye_result = eye_tracker.analyze(frame, face_result)
            blink_result = blink_detector.analyze(frame, face_result)

            # 3. Mouth Detection
            mouth_result = mouth_detector.analyze(frame, face_result)

            # 4. Head Pose Estimation
            head_result = head_pose.estimate(frame, face_result)

            # 5. Object Detection (YOLO)
            object_result = {"detections": [], "violations": []}
            if object_detector:  # Run every frame; persistence counter handles smoothing
                object_result = object_detector.detect(frame)

            # ── Aggregate violations ─────────────────────────────────────
            if face_result.get("no_face_sustained", False):
                violations.append("NO_FACE")
                alert_manager.trigger("⚠ NO FACE DETECTED", severity="critical")

            if face_result.get("multiple_faces", False):
                violations.append("MULTIPLE_PERSONS")
                alert_manager.trigger("⚠ MULTIPLE PERSONS DETECTED", severity="critical")

            if eye_result.get("looking_away", False):
                violations.append("LOOKING_AWAY")
                alert_manager.trigger("⚠ LOOKING AWAY FROM SCREEN", severity="warning")

            if blink_result.get("abnormal_blink", False):
                violations.append("ABNORMAL_BLINK")
                alert_manager.trigger("⚠ ABNORMAL BLINKING DETECTED", severity="warning")

            if mouth_result.get("mouth_open", False):
                violations.append("MOUTH_OPEN")
                alert_manager.trigger("⚠ TALKING DETECTED", severity="info")

            if head_result.get("suspicious_pose", False):
                violations.append(f"HEAD_POSE_{head_result.get('direction', 'UNKNOWN').upper()}")
                alert_manager.trigger(f"⚠ HEAD TURNED {head_result.get('direction','').upper()}", severity="warning")

            for v in object_result.get("violations", []):
                violations.append(v)
                if v == "PHONE_DETECTED":
                    alert_manager.trigger("🚨 MOBILE PHONE DETECTED!", severity="critical")
                elif v == "BOOK_DETECTED":
                    alert_manager.trigger("⚠ BOOK/NOTES DETECTED", severity="warning")

            # Log violations
            for v in violations:
                logger.log_violation(v)

            # ── Render frame ─────────────────────────────────────────────
            output_frame = display_manager.render(
                frame=frame,
                face_result=face_result,
                eye_result=eye_result,
                blink_result=blink_result,
                mouth_result=mouth_result,
                head_result=head_result,
                object_result=object_result,
                alerts=alert_manager.get_active_alerts(),
                fps=fps,
                exam_id=args.exam_id,
                stats=logger.get_stats()
            )

            cv2.imshow("AI Exam Proctoring System", output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n[INFO] Quit signal received.")
                break
            elif key == ord('s') or key == ord('S'):
                screenshot_path = f"logs/screenshot_{datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(screenshot_path, output_frame)
                print(f"[INFO] Screenshot saved: {screenshot_path}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        logger.log_event("SESSION_END", "Proctoring session ended.")
        logger.generate_report()
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Session report saved. Goodbye.")


if __name__ == "__main__":
    main()
