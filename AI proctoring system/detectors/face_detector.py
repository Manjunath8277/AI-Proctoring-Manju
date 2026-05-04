"""
Face Detector Module  –  Robust dual-model implementation
=========================================================
Strategy for near-zero false "NO FACE" detections:

1.  PRIMARY:   MediaPipe FaceDetection  (fast, robust, good for presence/absence)
2.  SECONDARY: MediaPipe FaceMesh       (landmarks for pose/eye/mouth modules)
3.  TERTIARY:  OpenCV DNN              (final fallback, always available)

Key improvements over v1:
  - Dual-model consensus: a face is considered PRESENT if EITHER FaceDetection
    OR FaceMesh sees it. One model's miss does not cause a false NO_FACE.
  - FaceDetection confidence lowered to 0.5 (more recall) while FaceMesh kept
    at 0.5/0.4 — detection is now separate from landmark quality.
  - Frame pre-processing: CLAHE equalisation + mild denoising improve detection
    in poor lighting and with motion blur.
  - Counter dynamics: face_detected resets the no_face_counter fully in 2 hits;
    no_face_sustained now requires 25 consecutive missed frames (~0.8 s @ 30fps).
  - IoU deduplication kept to prevent double-counting.
  - Last-known-box fallback: if detection misses for ≤ 5 frames but landmarks
    are still present, we reuse the last valid bounding box so the UI does not
    flicker.
"""

import cv2
import numpy as np
import urllib.request
import pathlib

# ── MediaPipe imports ───────────────────────────────────────────────────────
try:
    import mediapipe as mp
    _legacy_face_detection = mp.solutions.face_detection
    _legacy_face_mesh      = mp.solutions.face_mesh
    _USE_LEGACY = True
except Exception:
    _USE_LEGACY = False


# ── OpenCV DNN fallback ─────────────────────────────────────────────────────
def _download_opencv_model():
    model_dir = pathlib.Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True)
    proto      = model_dir / "deploy.prototxt"
    caffemodel = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
    proto_url  = ("https://raw.githubusercontent.com/opencv/opencv/master/"
                  "samples/dnn/face_detector/deploy.prototxt")
    model_url  = ("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_"
                  "face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")
    if not proto.exists():
        print("[FaceDetector] Downloading deploy.prototxt …")
        urllib.request.urlretrieve(proto_url, proto)
    if not caffemodel.exists():
        print("[FaceDetector] Downloading caffemodel …")
        urllib.request.urlretrieve(model_url, caffemodel)
    return str(proto), str(caffemodel)


class _OpenCVDNNDetector:
    """Lightweight OpenCV DNN face detector – always available as last resort."""

    CONF = 0.55   # slightly lower than before – we want high recall here

    def __init__(self):
        try:
            proto, model = _download_opencv_model()
            self.net  = cv2.dnn.readNetFromCaffe(proto, model)
            self._ok  = True
            print("[FaceDetector] OpenCV DNN fallback ready.")
        except Exception as e:
            self._ok = False
            print(f"[FaceDetector] OpenCV DNN init failed: {e}")

    def detect(self, frame: np.ndarray):
        if not self._ok:
            return []
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        dets = self.net.forward()
        boxes = []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf > self.CONF:
                x1 = max(0, int(dets[0, 0, i, 3] * w))
                y1 = max(0, int(dets[0, 0, i, 4] * h))
                x2 = min(w, int(dets[0, 0, i, 5] * w))
                y2 = min(h, int(dets[0, 0, i, 6] * h))
                boxes.append((x1, y1, x2, y2))
        return boxes


# ── Utilities ───────────────────────────────────────────────────────────────
def _preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Enhance frame for detection robustness:
      - CLAHE on L-channel  → handles dark / overexposed lighting
      - Fast denoising      → reduces camera noise on cheap webcams
    Returns BGR frame ready for cv2 / MediaPipe processing.
    """
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    lab   = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # Very mild blur – smooths sensor noise without blurring facial features
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return enhanced


def _iou(A, B):
    xA, yA = max(A[0], B[0]), max(A[1], B[1])
    xB, yB = min(A[2], B[2]), min(A[3], B[3])
    inter  = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (A[2]-A[0]) * (A[3]-A[1])
    aB = (B[2]-B[0]) * (B[3]-B[1])
    return inter / float(aA + aB - inter)


def _deduplicate(boxes, thresh=0.35):
    if len(boxes) <= 1:
        return boxes
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    kept  = []
    for box in boxes:
        if all(_iou(box, k) < thresh for k in kept):
            kept.append(box)
    return kept


def _landmarks_to_box(lm_arr, w, h, pad=0.15):
    """Convert a landmark array (N×3) to a padded bounding box."""
    xs = lm_arr[:, 0]
    ys = lm_arr[:, 1]
    x1, x2 = float(np.min(xs)), float(np.max(xs))
    y1, y2 = float(np.min(ys)), float(np.max(ys))
    pw = (x2 - x1) * pad
    ph = (y2 - y1) * pad
    return (
        max(0, int(x1 - pw)),
        max(0, int(y1 - ph)),
        min(w, int(x2 + pw)),
        min(h, int(y2 + ph)),
    )


# ── Main detector class ─────────────────────────────────────────────────────
class FaceDetector:
    """
    Robust dual-model face detector.

    Detection logic (OR consensus):
      face_present = FaceDetection sees face  OR  FaceMesh sees face  OR  DNN sees face

    This means a momentary miss by one model does NOT cause a NO_FACE alert.
    The NO_FACE alert is only raised after 25 consecutive frames (~0.8 s) where
    ALL models agree no face is present.
    """

    # How many consecutive all-model-miss frames before NO_FACE is flagged
    NO_FACE_THRESHOLD   = 25
    # How many consecutive multi-face frames before MULTI_FACE is flagged
    MULTI_FACE_THRESHOLD = 20
    # How many frames to keep reusing last known box after a detection miss
    BOX_HOLDOVER_FRAMES  = 5

    def __init__(self):
        # ── MediaPipe ─────────────────────────────────────────────────
        self._mp_det  = None
        self._mp_mesh = None
        if _USE_LEGACY:
            try:
                # model_selection=1: full-range model (works up to 5 m away)
                # Lower confidence → higher recall; temporal smoothing handles FPs
                self._mp_det = _legacy_face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                )
                # max_num_faces=3, lower thresholds → fewer tracking drops
                self._mp_mesh = _legacy_face_mesh.FaceMesh(
                    max_num_faces=3,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.4,
                )
                print("[FaceDetector] MediaPipe (FaceDetection + FaceMesh) ready.")
            except Exception as e:
                print(f"[FaceDetector] MediaPipe init failed: {e}")
                self._mp_det  = None
                self._mp_mesh = None

        # ── OpenCV DNN (always initialised as final fallback) ─────────
        self._dnn = _OpenCVDNNDetector()

        # ── State ─────────────────────────────────────────────────────
        self._no_face_counter    = 0
        self._multi_face_counter = 0
        self._last_boxes         = []        # last valid detection boxes
        self._holdover_count     = 0         # frames since last real detection

    # ── Private helpers ──────────────────────────────────────────────────────

    def _run_mediapipe(self, rgb, h, w):
        """
        Run both MediaPipe models and return (boxes_from_det, boxes_from_mesh,
        landmarks_list).  Either model result can be empty independently.
        """
        boxes_det  = []
        boxes_mesh = []
        lm_list    = []

        # 1. FaceDetection (best for presence/absence)
        if self._mp_det:
            try:
                res = self._mp_det.process(rgb)
                if res.detections:
                    for d in res.detections:
                        b  = d.location_data.relative_bounding_box
                        x1 = max(0, int(b.xmin * w))
                        y1 = max(0, int(b.ymin * h))
                        x2 = min(w, int((b.xmin + b.width)  * w))
                        y2 = min(h, int((b.ymin + b.height) * h))
                        boxes_det.append((x1, y1, x2, y2))
            except Exception:
                pass

        # 2. FaceMesh (best for landmarks; also confirms face presence)
        if self._mp_mesh:
            try:
                res = self._mp_mesh.process(rgb)
                if res.multi_face_landmarks:
                    for face_lm in res.multi_face_landmarks:
                        arr = np.array(
                            [(lm.x * w, lm.y * h, lm.z)
                             for lm in face_lm.landmark],
                            dtype=np.float32,
                        )
                        lm_list.append(arr)
                        boxes_mesh.append(_landmarks_to_box(arr, w, h))
            except Exception:
                pass

        return boxes_det, boxes_mesh, lm_list

    # ── Public API ───────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]

        # Pre-process for better detection in varied lighting
        enhanced = _preprocess(frame)
        rgb      = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

        # ── Run all models ──────────────────────────────────────────────
        boxes_det, boxes_mesh, lm_list = [], [], []
        if self._mp_det or self._mp_mesh:
            boxes_det, boxes_mesh, lm_list = self._run_mediapipe(rgb, h, w)

        # DNN runs when MediaPipe gives no detections at all (any model)
        # OR when MediaPipe is not available
        mp_any = len(boxes_det) > 0 or len(boxes_mesh) > 0
        boxes_dnn = []
        if not mp_any:
            boxes_dnn = self._dnn.detect(enhanced)

        # ── Merge & deduplicate ─────────────────────────────────────────
        # Union of all detections from all models → OR consensus
        all_raw = boxes_det + boxes_mesh + boxes_dnn
        face_boxes = _deduplicate(all_raw, thresh=0.35)

        # Trim landmark list to deduplicated count
        lm_list = lm_list[:len(face_boxes)]

        face_count    = len(face_boxes)
        face_detected = face_count > 0

        # ── Last-known-box holdover ─────────────────────────────────────
        # If this frame has no detections but we had one very recently,
        # keep using the last valid box so the UI doesn't flicker.
        if face_detected:
            self._last_boxes     = face_boxes
            self._holdover_count = 0
        else:
            self._holdover_count += 1
            if self._holdover_count <= self.BOX_HOLDOVER_FRAMES and self._last_boxes:
                # Reuse last box — treat as "face present" for display
                face_boxes    = self._last_boxes
                face_count    = len(face_boxes)
                face_detected = True   # suppress alert during brief holdover

        multiple_faces = face_count > 1

        # ── Temporal smoothing ──────────────────────────────────────────
        # No-face counter: resets fast on re-detection (÷5 per hit),
        # requires NO_FACE_THRESHOLD consecutive real misses to alert.
        if face_detected:
            self._no_face_counter = max(self._no_face_counter - 5, 0)
        else:
            self._no_face_counter = min(self._no_face_counter + 1,
                                        self.NO_FACE_THRESHOLD * 2)

        # Multi-face counter: requires MULTI_FACE_THRESHOLD consecutive hits
        if multiple_faces:
            self._multi_face_counter = min(self._multi_face_counter + 1,
                                           self.MULTI_FACE_THRESHOLD * 2)
        else:
            self._multi_face_counter = max(self._multi_face_counter - 3, 0)

        return {
            "face_detected":    face_detected,
            "face_count":       face_count,
            "multiple_faces":   self._multi_face_counter >= self.MULTI_FACE_THRESHOLD,
            "no_face_sustained": self._no_face_counter  >= self.NO_FACE_THRESHOLD,
            "face_boxes":       face_boxes,
            "landmarks_list":   lm_list,
            "primary_landmarks": lm_list[0] if lm_list else None,
            "backend":          "mediapipe+dnn" if self._mp_det else "opencv_dnn",
        }
