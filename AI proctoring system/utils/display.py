"""
Display Manager
Renders all detection overlays, HUD, and alerts onto the webcam frame.
"""

import cv2
import numpy as np
import time
from typing import List, Optional


# Color palette (BGR)
COLORS = {
    "green":    (0, 220, 0),
    "red":      (0, 0, 220),
    "orange":   (0, 165, 255),
    "yellow":   (0, 220, 220),
    "blue":     (220, 100, 0),
    "cyan":     (220, 220, 0),
    "white":    (255, 255, 255),
    "gray":     (150, 150, 150),
    "dark":     (20, 20, 20),
    "bg":       (15, 15, 30),
    "panel":    (30, 30, 50),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.45
FONT_MED = 0.55
FONT_LARGE = 0.7
THICKNESS = 1
THICKNESS_BOLD = 2

YOLO_COLORS = {
    "cell phone": (0, 0, 255),
    "person":     (0, 255, 0),
    "book":       (0, 165, 255),
    "laptop":     (255, 0, 255),
    "tablet":     (255, 165, 0),
}


def draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=-1, alpha=0.6):
    """Draw a semi-transparent rounded rectangle."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    r = radius
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.circle(overlay, (x1 + r, y1 + r), r, color, thickness)
    cv2.circle(overlay, (x2 - r, y1 + r), r, color, thickness)
    cv2.circle(overlay, (x1 + r, y2 - r), r, color, thickness)
    cv2.circle(overlay, (x2 - r, y2 - r), r, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def put_text_with_bg(img, text, pos, font_scale=FONT_MED, color=COLORS["white"],
                     bg_color=COLORS["dark"], thickness=THICKNESS, padding=4, alpha=0.7):
    """Put text with a semi-transparent background."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = pos
    overlay = img.copy()
    cv2.rectangle(overlay, (x - padding, y - th - padding),
                  (x + tw + padding, y + baseline + padding), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, pos, FONT, font_scale, color, thickness, cv2.LINE_AA)


class DisplayManager:
    def __init__(self):
        self._frame_count = 0
        self._status_history = []

    def render(self, frame, face_result, eye_result, blink_result,
               mouth_result, head_result, object_result, alerts, fps,
               exam_id, stats) -> np.ndarray:
        """Render only camera-based detection overlays onto the frame."""
        output = frame.copy()
        h, w = output.shape[:2]
        self._frame_count += 1

        # ── Draw face bounding boxes ────────────────────────────────────
        self._draw_face_boxes(output, face_result)

        # ── Draw YOLO detections ────────────────────────────────────────
        self._draw_yolo(output, object_result)

        # ── Draw nose direction line ────────────────────────────────────
        self._draw_head_pose(output, head_result)

        # ── Draw gaze point ────────────────────────────────────────────
        self._draw_gaze(output, eye_result)

        # ── Alert overlay ──────────────────────────────────────────────
        self._draw_alerts(output, alerts, w, h)

        # ── Violation border flash ─────────────────────────────────────
        critical = any(a.severity == "critical" for a in alerts)
        if critical and (self._frame_count % 20 < 10):
            cv2.rectangle(output, (0, 0), (w - 1, h - 1), COLORS["red"], 6)

        return output

    # ── Internal helpers ────────────────────────────────────────────────

    def _draw_header(self, img, fps, exam_id, w):
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 45), COLORS["bg"], -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

        cv2.putText(img, "AI EXAM PROCTOR", (10, 30), FONT, FONT_LARGE,
                    COLORS["cyan"], THICKNESS_BOLD, cv2.LINE_AA)
        exam_txt = f"ID: {exam_id}"
        cv2.putText(img, exam_txt, (w // 2 - 100, 30), FONT, FONT_SMALL,
                    COLORS["gray"], THICKNESS, cv2.LINE_AA)
        fps_txt = f"FPS: {fps:.1f}"
        (tw, _), _ = cv2.getTextSize(fps_txt, FONT, FONT_MED, THICKNESS)
        cv2.putText(img, fps_txt, (w - tw - 10, 30), FONT, FONT_MED,
                    COLORS["green"], THICKNESS, cv2.LINE_AA)

    def _draw_status_panel(self, img, face_r, eye_r, blink_r, mouth_r, head_r, h):
        panel_w = 230
        panel_x = 5
        panel_y = 55

        overlay = img.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + 260), COLORS["panel"], -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        cv2.putText(img, "DETECTION STATUS", (panel_x + 5, panel_y + 18),
                    FONT, FONT_SMALL, COLORS["cyan"], THICKNESS_BOLD, cv2.LINE_AA)

        rows = [
            ("Face", face_r.get("face_detected", False),
             f"{face_r.get('face_count', 0)} face(s)"),
            ("Multi-Person", face_r.get("multiple_faces", False),
             "ALERT" if face_r.get("multiple_faces") else "OK"),
            ("Gaze", eye_r.get("looking_away", False),
             eye_r.get("gaze_direction", "?")),
            ("Blink", blink_r.get("abnormal_blink", False),
             f"EAR:{blink_r.get('avg_ear', 0):.2f}"),
            ("Mouth", mouth_r.get("mouth_open", False),
             f"MAR:{mouth_r.get('mar', 0):.2f}"),
            ("Head", head_r.get("suspicious_pose", False),
             head_r.get("direction", "?")),
        ]

        for i, (label, is_bad, value) in enumerate(rows):
            y = panel_y + 38 + i * 35
            color = COLORS["red"] if is_bad else COLORS["green"]
            indicator = "✖" if is_bad else "✔"
            cv2.putText(img, f"{indicator} {label}", (panel_x + 8, y),
                        FONT, FONT_SMALL, color, THICKNESS, cv2.LINE_AA)
            cv2.putText(img, value, (panel_x + 130, y),
                        FONT, FONT_SMALL, COLORS["gray"], THICKNESS, cv2.LINE_AA)

        # Head pose angles
        y = panel_y + 38 + len(rows) * 35
        cv2.putText(img,
                    f"Y:{head_r.get('yaw', 0):.0f}° "
                    f"P:{head_r.get('pitch', 0):.0f}° "
                    f"R:{head_r.get('roll', 0):.0f}°",
                    (panel_x + 5, y),
                    FONT, 0.38, COLORS["gray"], THICKNESS, cv2.LINE_AA)

    def _draw_stats_panel(self, img, stats, w, h):
        if not stats:
            return
        panel_w = 210
        panel_x = w - panel_w - 5
        panel_y = 55
        panel_h = min(30 + len(stats) * 20 + 20, 200)

        overlay = img.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), COLORS["panel"], -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        cv2.putText(img, "VIOLATIONS LOG", (panel_x + 5, panel_y + 16),
                    FONT, FONT_SMALL, COLORS["orange"], THICKNESS_BOLD, cv2.LINE_AA)

        for i, (k, v) in enumerate(list(stats.items())[:8]):
            y = panel_y + 34 + i * 20
            short_k = k.replace("HEAD_POSE_", "HEAD_").replace("_DETECTED", "")[:18]
            cv2.putText(img, f"{short_k}", (panel_x + 5, y),
                        FONT, 0.38, COLORS["white"], THICKNESS, cv2.LINE_AA)
            cv2.putText(img, str(v), (panel_x + panel_w - 25, y),
                        FONT, 0.38, COLORS["yellow"], THICKNESS, cv2.LINE_AA)

    def _draw_alerts(self, img, alerts, w, h):
        if not alerts:
            return
        start_y = h - 30 - len(alerts) * 42
        for i, alert in enumerate(alerts):
            y = start_y + i * 42
            box_w = 440
            x = (w - box_w) // 2

            overlay = img.copy()
            cv2.rectangle(overlay, (x - 4, y - 28), (x + box_w, y + 10),
                          alert.color, -1)
            cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)

            # Border
            cv2.rectangle(img, (x - 4, y - 28), (x + box_w, y + 10),
                          COLORS["white"], 1)

            cv2.putText(img, alert.message, (x + 5, y),
                        FONT, FONT_MED, COLORS["white"], THICKNESS_BOLD, cv2.LINE_AA)

    def _draw_face_boxes(self, img, face_result):
        for i, (x1, y1, x2, y2) in enumerate(face_result.get("face_boxes", [])):
            color = COLORS["red"] if i > 0 else COLORS["green"]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"Person {i + 1}" if i > 0 else "You"
            put_text_with_bg(img, label, (x1, y1 - 6),
                              color=color, bg_color=COLORS["dark"])

    def _draw_yolo(self, img, object_result):
        for det in object_result.get("detections", []):
            cls = det["class"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["bbox"]
            color = YOLO_COLORS.get(cls, COLORS["yellow"])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls} {conf:.0%}"
            put_text_with_bg(img, label, (x1, y1 - 6),
                              color=color, bg_color=COLORS["dark"])

    def _draw_head_pose(self, img, head_result):
        line = head_result.get("nose_direction_line")
        if line:
            start, end = line
            cv2.arrowedLine(img, start, end, COLORS["cyan"], 2, tipLength=0.3)

    def _draw_gaze(self, img, eye_result):
        gaze_pt = eye_result.get("gaze_point")
        if gaze_pt:
            cv2.circle(img, gaze_pt, 4, COLORS["yellow"], -1)
            cv2.circle(img, gaze_pt, 8, COLORS["yellow"], 1)
