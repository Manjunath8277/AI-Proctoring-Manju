"""
Violation Logger Module
Records all violations and events to:
  - Console (terminal)
  - CSV log file
  - Session summary report
"""

import os
import csv
import json
import time
from datetime import datetime
from collections import defaultdict


VIOLATION_MESSAGES = {
    "NO_FACE":           "No face detected in frame",
    "MULTIPLE_PERSONS":  "Multiple persons detected",
    "LOOKING_AWAY":      "User looking away from screen",
    "ABNORMAL_BLINK":    "Abnormal blinking pattern",
    "MOUTH_OPEN":        "Mouth open / possible talking",
    "HEAD_POSE_LEFT":    "Head turned LEFT",
    "HEAD_POSE_RIGHT":   "Head turned RIGHT",
    "HEAD_POSE_UP":      "Head turned UP",
    "HEAD_POSE_DOWN":    "Head turned DOWN",
    "HEAD_POSE_TILTED":  "Head tilted",
    "PHONE_DETECTED":    "Mobile phone detected!",
    "BOOK_DETECTED":     "Book/notes detected",
    "LAPTOP_DETECTED":   "Secondary laptop detected",
    "TABLET_DETECTED":   "Tablet detected",
    "SESSION_START":     "Session started",
    "SESSION_END":       "Session ended",
}

VIOLATION_SEVERITY = {
    "NO_FACE":           "CRITICAL",
    "MULTIPLE_PERSONS":  "CRITICAL",
    "PHONE_DETECTED":    "CRITICAL",
    "LOOKING_AWAY":      "WARNING",
    "ABNORMAL_BLINK":    "WARNING",
    "HEAD_POSE_LEFT":    "WARNING",
    "HEAD_POSE_RIGHT":   "WARNING",
    "HEAD_POSE_UP":      "WARNING",
    "HEAD_POSE_DOWN":    "WARNING",
    "HEAD_POSE_TILTED":  "WARNING",
    "BOOK_DETECTED":     "WARNING",
    "LAPTOP_DETECTED":   "WARNING",
    "TABLET_DETECTED":   "WARNING",
    "MOUTH_OPEN":        "INFO",
}

# Minimum seconds between logging the same violation
VIOLATION_COOLDOWN = {
    "NO_FACE":          3,
    "MULTIPLE_PERSONS": 5,
    "LOOKING_AWAY":     3,
    "ABNORMAL_BLINK":   10,
    "MOUTH_OPEN":       5,
    "HEAD_POSE_LEFT":   3,
    "HEAD_POSE_RIGHT":  3,
    "HEAD_POSE_UP":     3,
    "HEAD_POSE_DOWN":   3,
    "PHONE_DETECTED":   5,
    "BOOK_DETECTED":    10,
}


class ViolationLogger:
    def __init__(self, log_dir: str = "logs", exam_id: str = "EXAM"):
        self.log_dir = log_dir
        self.exam_id = exam_id
        os.makedirs(log_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"{exam_id}_{ts}.csv")
        self.report_path = os.path.join(log_dir, f"{exam_id}_{ts}_report.json")

        self._violation_counts = defaultdict(int)
        self._last_logged = {}
        self._events = []
        self._start_time = time.time()

        # Init CSV
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "elapsed_sec", "event_type", "severity", "message"])

        print(f"[LOG] Logging to: {self.csv_path}")

    def log_violation(self, violation_type: str):
        """Log a violation with cooldown deduplication."""
        now = time.time()
        cooldown = VIOLATION_COOLDOWN.get(violation_type, 2)

        if now - self._last_logged.get(violation_type, 0) < cooldown:
            return  # Skip duplicate within cooldown window

        self._last_logged[violation_type] = now
        self._violation_counts[violation_type] += 1

        message = VIOLATION_MESSAGES.get(violation_type, violation_type)
        severity = VIOLATION_SEVERITY.get(violation_type, "INFO")
        elapsed = now - self._start_time
        ts = datetime.now().strftime("%H:%M:%S")

        # Terminal output with color codes
        color = {"CRITICAL": "\033[91m", "WARNING": "\033[93m", "INFO": "\033[94m"}.get(severity, "")
        reset = "\033[0m"
        print(f"  {color}[{severity}]{reset} [{ts}] {message}")

        # CSV write
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, f"{elapsed:.1f}", violation_type, severity, message])

        self._events.append({
            "time": ts, "elapsed": elapsed,
            "type": violation_type, "severity": severity, "message": message
        })

    def log_event(self, event_type: str, message: str):
        """Log a non-violation system event."""
        ts = datetime.now().strftime("%H:%M:%S")
        elapsed = time.time() - self._start_time
        print(f"  [EVENT] [{ts}] {message}")

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, f"{elapsed:.1f}", event_type, "INFO", message])

    def get_stats(self) -> dict:
        """Return current violation statistics."""
        return dict(self._violation_counts)

    def generate_report(self):
        """Generate a JSON session summary report."""
        duration = time.time() - self._start_time
        total_violations = sum(self._violation_counts.values())

        # Risk score (weighted)
        risk_score = 0
        weights = {"CRITICAL": 10, "WARNING": 3, "INFO": 1}
        for vtype, count in self._violation_counts.items():
            sev = VIOLATION_SEVERITY.get(vtype, "INFO")
            risk_score += count * weights.get(sev, 1)

        risk_level = (
            "HIGH" if risk_score > 50
            else "MEDIUM" if risk_score > 20
            else "LOW"
        )

        report = {
            "exam_id": self.exam_id,
            "start_time": datetime.fromtimestamp(self._start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": round(duration, 1),
            "total_violations": total_violations,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "violation_counts": dict(self._violation_counts),
            "events": self._events[-50:]  # Last 50 events
        }

        with open(self.report_path, "w") as f:
            json.dump(report, f, indent=2)

        print("\n" + "=" * 60)
        print("  EXAM SESSION REPORT")
        print("=" * 60)
        print(f"  Exam ID        : {self.exam_id}")
        print(f"  Duration       : {duration:.0f}s")
        print(f"  Total Violations: {total_violations}")
        print(f"  Risk Score     : {risk_score}")
        print(f"  Risk Level     : {risk_level}")
        print("  Violations by type:")
        for vtype, count in sorted(self._violation_counts.items()):
            print(f"    {vtype:<25} : {count}")
        print(f"\n  Report saved: {self.report_path}")
        print("=" * 60)
