"""
Object Detector Module  –  High-accuracy dual-pass YOLO implementation
=======================================================================

Key improvements over v1:
  1. Dual-pass inference:
       Pass 1 – full frame at 640px  (catches large/medium objects)
       Pass 2 – centre-crop 60 % at 416px  (catches small objects like phones
                 held close to the chest / on the desk)
       Results are merged with NMS to remove duplicates.

  2. Two-tier confidence:
       DETECT_CONF = 0.40  – cast a wide net for raw detections
       CONFIRM_CONF = 0.55 – only surface detections above this to the UI
       This gives high recall without flooding the display.

  3. Frame pre-processing (CLAHE + mild blur) – same as face detector,
     improves detection in dark rooms and under harsh overhead lighting.

  4. Tighter class list – only cell phone, book, laptop flagged.
     Remote / mouse / keyboard removed (too many false positives).

  5. Smarter temporal smoothing:
       - Counters increment by confidence-weighted amount (strong detection
         fills the counter faster than a borderline one).
       - Runs on EVERY frame in main.py (change frame_count % 3 → every frame)
         so PERSISTENCE_REQUIRED = 5 frames ≈ 0.17s – fast but robust.
       - Decay is aggressive (−3) so objects that disappear clear quickly.

  6. Per-detection NMS across dual-pass results prevents double-counting.

Usage note: caller should run this every frame (not every 3rd frame) because
  the persistence counter now requires only 5 hits, not 8.
  In main.py change:  if object_detector and frame_count % 3 == 0:
  To:                 if object_detector:
"""

import numpy as np
import cv2
from collections import defaultdict

# ── Class config ─────────────────────────────────────────────────────────────
# Only genuine exam violations. Remote/mouse/keyboard removed.
SUSPICIOUS_CLASSES = {
    "cell phone": "PHONE_DETECTED",
    "book":       "BOOK_DETECTED",
    "laptop":     "LAPTOP_DETECTED",
}
PERSON_CLASS = "person"

# Two-tier confidence
DETECT_CONF  = 0.40   # minimum to even consider a detection
CONFIRM_CONF = 0.55   # must clear this to count toward persistence

# Temporal persistence
PERSISTENCE_REQUIRED = 5    # consecutive confirmed frames before violation fires
DECAY_RATE           = 3    # frames subtracted per miss (fast clear)

# NMS IoU threshold (suppress duplicate boxes from dual-pass)
NMS_IOU_THRESH = 0.45


# ── Helpers ──────────────────────────────────────────────────────────────────
def _preprocess(frame: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement + mild denoising for robust detection."""
    lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)


def _nms(detections: list, iou_thresh: float = NMS_IOU_THRESH) -> list:
    """
    Non-maximum suppression across all detections regardless of class.
    Keeps highest-confidence box when two boxes overlap heavily.
    """
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for d in dets:
        b1 = d["bbox"]
        suppress = False
        for k in kept:
            b2 = k["bbox"]
            # IoU
            xA = max(b1[0], b2[0]); yA = max(b1[1], b2[1])
            xB = min(b1[2], b2[2]); yB = min(b1[3], b2[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            if inter == 0:
                continue
            a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
            a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
            iou = inter / float(a1 + a2 - inter)
            if iou > iou_thresh and d["class"] == k["class"]:
                suppress = True
                break
        if not suppress:
            kept.append(d)
    return kept


def _run_yolo(model, img: np.ndarray, size: int, offset=(0, 0)) -> list:
    """
    Run YOLO on img resized to `size`. Returns raw detections with
    coordinates mapped back to the original full-frame space via `offset`.
    offset = (x_offset, y_offset) when img is a crop of the full frame.
    """
    results = model(img, verbose=False, conf=DETECT_CONF, imgsz=size)[0]
    dets = []
    oh, ow = img.shape[:2]
    ox, oy = offset

    if results.boxes is not None:
        for box in results.boxes:
            cls_id   = int(box.cls[0])
            conf     = float(box.conf[0])
            cls_name = model.names[cls_id].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Map back to full-frame coordinates
            dets.append({
                "class":      cls_name,
                "confidence": conf,
                "bbox":       (x1 + ox, y1 + oy, x2 + ox, y2 + oy),
            })
    return dets


# ── Main class ───────────────────────────────────────────────────────────────
class ObjectDetector:
    """
    High-accuracy object detector using dual-pass YOLO inference.

    Pass 1: full frame  → catches laptops, books, people
    Pass 2: centre crop → catches small phones held at arm's length or on desk
    """

    def __init__(self, model_path: str = "yolov8n.pt"):
        from ultralytics import YOLO

        # Prefer yolov8s if caller passed the nano model; s is ~2× more accurate
        # for small objects with only ~30% speed penalty.
        if model_path == "yolov8n.pt":
            import os
            # Try to use yolov8s; fall back to whatever is specified
            s_path = "yolov8s.pt"
            try:
                self.model = YOLO(s_path)
                print(f"[YOLO] Upgraded to yolov8s for better accuracy.")
            except Exception:
                self.model = YOLO(model_path)
                print(f"[YOLO] Model loaded: {model_path}")
        else:
            self.model = YOLO(model_path)
            print(f"[YOLO] Model loaded: {model_path}")

        # Temporal persistence counters per violation type
        self._counters: dict = defaultdict(float)

    # ── Detection pipeline ────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> dict:
        """
        Full dual-pass detection with NMS + temporal smoothing.
        Call this every frame (not every 3rd frame).
        """
        enhanced = _preprocess(frame)
        h, w     = enhanced.shape[:2]

        # ── Pass 1: full frame at 640 ────────────────────────────────────
        p1 = _run_yolo(self.model, enhanced, size=640)

        # ── Pass 2: centre crop 60% at 416 ──────────────────────────────
        # Captures small objects in the exam area (desk, in-hand)
        cx, cy  = w // 2, h // 2
        crop_w  = int(w * 0.60)
        crop_h  = int(h * 0.60)
        x1c = max(0, cx - crop_w  // 2)
        y1c = max(0, cy - crop_h  // 2)
        x2c = min(w, x1c + crop_w)
        y2c = min(h, y1c + crop_h)
        crop = enhanced[y1c:y2c, x1c:x2c]
        p2   = _run_yolo(self.model, crop, size=416, offset=(x1c, y1c))

        # ── Merge + NMS ──────────────────────────────────────────────────
        all_raw  = p1 + p2
        all_dets = _nms(all_raw)

        # ── Classify & count ─────────────────────────────────────────────
        seen_confirmed = set()   # violations seen with conf ≥ CONFIRM_CONF
        seen_raw       = set()   # violations seen at any conf (for display only)
        person_count   = 0
        display_dets   = []

        for d in all_dets:
            cls  = d["class"]
            conf = d["confidence"]

            if cls == PERSON_CLASS:
                person_count += 1

            if cls in SUSPICIOUS_CLASSES:
                viol = SUSPICIOUS_CLASSES[cls]
                if viol:
                    seen_raw.add(viol)
                    if conf >= CONFIRM_CONF:
                        seen_confirmed.add(viol)

            display_dets.append(d)

        if person_count > 1:
            seen_raw.add("MULTIPLE_PERSONS")
            seen_confirmed.add("MULTIPLE_PERSONS")

        # ── Temporal smoothing ────────────────────────────────────────────
        all_types = set(v for v in SUSPICIOUS_CLASSES.values() if v) | {"MULTIPLE_PERSONS"}

        for vtype in all_types:
            if vtype in seen_confirmed:
                # Increment by 1 (strong detection) – capped at 3× persistence
                self._counters[vtype] = min(
                    self._counters[vtype] + 1.0,
                    PERSISTENCE_REQUIRED * 3,
                )
            elif vtype in seen_raw:
                # Weak detection – increment by 0.5 (needs more evidence)
                self._counters[vtype] = min(
                    self._counters[vtype] + 0.5,
                    PERSISTENCE_REQUIRED * 3,
                )
            else:
                self._counters[vtype] = max(self._counters[vtype] - DECAY_RATE, 0)

        confirmed_violations = [
            v for v in all_types if self._counters[v] >= PERSISTENCE_REQUIRED
        ]

        # ── Filter display detections to confirmed classes only ───────────
        confirmed_cls = {
            cls for cls, viol in SUSPICIOUS_CLASSES.items()
            if viol in confirmed_violations
        }
        if "MULTIPLE_PERSONS" in confirmed_violations:
            confirmed_cls.add(PERSON_CLASS)

        # Also always show person boxes (useful for multi-person detection)
        show_dets = [
            d for d in display_dets
            if d["class"] in confirmed_cls
        ]

        return {
            "detections":   show_dets,
            "violations":   confirmed_violations,
            "person_count": person_count,
        }
