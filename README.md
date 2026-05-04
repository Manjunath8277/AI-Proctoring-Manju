# 🎓 AI Proctoring System

An AI-based online exam proctoring system that uses **Computer Vision (OpenCV + YOLOv8)** to detect cheating behaviors in real-time during online exams.

---

## 🚀 Features

* 👤 **Face Detection** (ensures student presence)
* 👥 **Multiple Person Detection** (detects cheating via extra people)
* 📱 **Mobile Phone Detection** (identifies unauthorized devices)
* 🧠 **Head Pose Estimation** (detects looking away)
* 🔊 **Audio Alerts System** (warning notifications)
* 📊 **Live Logging System** (CSV + JSON reports)
* 📸 **Screenshot Capture on Suspicious Activity**

---

## 🛠️ Tech Stack

* Python 🐍
* OpenCV
* YOLOv8 (Ultralytics)
* PyTorch
* NumPy
* Computer Vision & DNN Models

---

## 📁 Project Structure

```
AI proctoring system/
│── main.py
│── requirements.txt
│── yolov8s.pt
│── detectors/
│   ├── face_detector.py
│   ├── object_detector.py
│   ├── eye_tracker.py
│   ├── head_pose.py
│   ├── mouth_detector.py
│   └── models/
│── utils/
│   ├── logger.py
│   ├── alert_manager.py
│   ├── sound_manager.py
│   └── display.py
│── logs/
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-proctoring-system.git
cd ai-proctoring-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install required system dependency (IMPORTANT)

Download and install:

👉 [https://aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)

---

## ▶️ Run the Project

```bash
python main.py
```

---

## 🧠 How It Works

1. Webcam captures live video feed
2. Face detection ensures student presence
3. YOLO detects suspicious objects (mobile, earbuds, etc.)
4. Eye & head tracking monitor attention
5. System logs every suspicious activity
6. Alerts are triggered in real-time

---

## 📊 Output

* Real-time detection window
* CSV logs of exam activity
* JSON reports for analysis
* Screenshot evidence of violations

---

## ⚠️ Limitations

* Requires good lighting for accurate detection
* Performance depends on webcam quality
* YOLO model may require GPU for best speed

---

## 🚀 Future Improvements

* Student behavior analytics
* Face recognition login system

---

## 👨‍💻 Author

**Manjunath A**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to contribute!
