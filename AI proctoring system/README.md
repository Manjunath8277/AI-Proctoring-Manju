# 🎓 AI-Based Online Exam Proctoring System

A real-time AI-powered exam proctoring system using Python, OpenCV, MediaPipe, and YOLOv8.

---

## 📋 Features

| Feature | Description |
|---|---|
| **Face Detection** | Detects presence/absence of face using MediaPipe |
| **Eye Tracking** | Gaze direction estimation (Left/Right/Center) |
| **Blink Detection** | Eye Aspect Ratio (EAR) blink counting & abnormal pattern detection |
| **Mouth Detection** | Mouth Aspect Ratio (MAR) talking/whispering detection |
| **Head Pose Estimation** | Yaw/Pitch/Roll via solvePnP – flags suspicious head turns |
| **Object Detection** | YOLOv8 detects phone, books, multiple persons, laptop, tablet |
| **Multiple Person Detection** | Alerts if more than one person is visible |
| **Violation Logging** | CSV log + JSON session report with risk score |
| **Real-Time Alerts** | On-screen banner alerts with severity levels |
| **Live HUD** | FPS counter, detection status panel, violation stats |

---

## 🛠 Requirements

- Python 3.10+
- Webcam

### Python packages

```
opencv-python>=4.8.0
mediapipe>=0.10.0
ultralytics>=8.0.0
numpy>=1.24.0
```

---

## 🚀 Installation

### 1. Clone / extract the project

```bash
cd exam_proctor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

On first run, YOLOv8 will automatically download `yolov8n.pt` (~6MB).

---

## ▶️ Usage

### Basic run (all features):
```bash
python main.py
```

### Options:
```bash
python main.py --camera 0          # Camera index (default: 0)
python main.py --no-yolo           # Disable YOLO (faster, no object detection)
python main.py --exam-id EXAM_001  # Set custom exam session ID
python main.py --log-dir logs      # Set log output directory
```

### Keyboard shortcuts:
| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot |

---

## 📁 Project Structure

```
exam_proctor/
│
├── main.py                     ← Entry point
│
├── detectors/
│   ├── face_detector.py        ← MediaPipe face + mesh detection
│   ├── eye_tracker.py          ← Iris gaze tracking
│   ├── blink_detector.py       ← EAR-based blink detection
│   ├── mouth_detector.py       ← MAR-based mouth/talking detection
│   ├── head_pose.py            ← solvePnP head orientation
│   └── object_detector.py      ← YOLOv8 object detection
│
├── utils/
│   ├── logger.py               ← CSV + JSON violation logging
│   ├── alert_manager.py        ← On-screen alert system
│   └── display.py              ← Frame rendering and HUD
│
├── logs/                       ← Auto-created; stores CSV + JSON reports
│
└── requirements.txt
```

---

## 🔍 Detection Details

### Face Detection
- Uses **MediaPipe FaceMesh** (refine_landmarks=True) for 478 landmarks
- No-face alert triggers after 10 consecutive frames

### Eye Tracking
- Uses iris landmark positions (landmark IDs 468–477)
- Horizontal iris ratio < 0.35 → looking LEFT; > 0.65 → looking RIGHT
- Smoothed over 10-frame rolling window

### Blink Detection
- **Eye Aspect Ratio (EAR)** = (vertical eye distance) / (horizontal eye distance)
- EAR < 0.21 → eyes closed
- Sustained closed > 15 frames → suspicious
- > 10 blinks in 5 seconds → rapid blinking alert

### Mouth Detection
- **Mouth Aspect Ratio (MAR)** using lip landmarks
- MAR > 0.5 for 8+ frames → talking alert

### Head Pose Estimation
- **OpenCV solvePnP** with 6 facial anchor points
- Yaw > ±20° → LEFT/RIGHT alert
- Pitch > ±15° → UP/DOWN alert

### Object Detection (YOLOv8)
- Runs on every 3rd frame for performance
- Detects: cell phone, book, laptop, tablet, remote, keyboard, scissors
- Multiple-person detection from COCO `person` class

---

## 📊 Logging & Reports

After each session, logs are saved in `logs/`:

| File | Contents |
|------|----------|
| `EXAM_<id>_<ts>.csv` | Timestamped violation log |
| `EXAM_<id>_<ts>_report.json` | Full JSON summary with risk score |

### Risk Levels:
- **LOW**: Risk score < 20
- **MEDIUM**: Risk score 20–50
- **HIGH**: Risk score > 50

Risk score is calculated as:
- CRITICAL violation × 10 (phone, multiple persons, no face)
- WARNING violation × 3 (looking away, head pose, book)
- INFO violation × 1 (mouth open)

---

## ⚠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| Camera not found | Try `--camera 1` or check webcam drivers |
| MediaPipe import error | `pip install mediapipe==0.10.14` |
| YOLO download fails | Run with `--no-yolo` to skip |
| Slow performance | Use `--no-yolo` or reduce resolution |
| macOS camera permission | Grant Terminal/Python camera access in System Preferences |

---

## 🧪 Technology Stack

| Component | Technology |
|-----------|-----------|
| Video Capture | OpenCV |
| Face/Eye/Mouth | MediaPipe FaceMesh |
| Head Pose | OpenCV solvePnP |
| Object Detection | Ultralytics YOLOv8n |
| Language | Python 3.10+ |

---

## 📝 License

For educational and research use only. Not intended for production deployment without proper privacy review and consent mechanisms.
