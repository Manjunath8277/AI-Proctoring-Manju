"""
Alert Manager
Manages on-screen alerts with severity levels, expiry, and audio alerts.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

from utils.sound_manager import SoundManager


ALERT_DURATION = {
    "critical": 4.0,
    "warning":  3.0,
    "info":     2.0,
}

ALERT_COLORS = {
    "critical": (0, 0, 220),    # Red (BGR)
    "warning":  (0, 165, 255),  # Orange
    "info":     (200, 200, 0),  # Cyan-ish
}

# Map alert messages to violation types for targeted sounds
MESSAGE_TO_VIOLATION = {
    "MOBILE PHONE DETECTED":     "PHONE_DETECTED",
    "PHONE DETECTED":            "PHONE_DETECTED",
    "MULTIPLE PERSONS DETECTED": "MULTIPLE_PERSONS",
    "NO FACE DETECTED":          "NO_FACE",
    "LOOKING AWAY FROM SCREEN":  "LOOKING_AWAY",
    "HEAD TURNED LEFT":          "HEAD_POSE_LEFT",
    "HEAD TURNED RIGHT":         "HEAD_POSE_RIGHT",
    "BOOK/NOTES DETECTED":       "BOOK_DETECTED",
    "BOOK DETECTED":             "BOOK_DETECTED",
}


def _message_to_violation(message: str) -> Optional[str]:
    """Fuzzy-match alert message to a violation type key."""
    msg_upper = message.upper()
    for keyword, vtype in MESSAGE_TO_VIOLATION.items():
        if keyword in msg_upper:
            return vtype
    return None


@dataclass
class Alert:
    message: str
    severity: str
    violation_type: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    @property
    def is_active(self) -> bool:
        duration = ALERT_DURATION.get(self.severity, 3.0)
        return (time.time() - self.created_at) < duration

    @property
    def color(self):
        return ALERT_COLORS.get(self.severity, (200, 200, 200))

    @property
    def alpha(self) -> float:
        """Fade out effect."""
        duration = ALERT_DURATION.get(self.severity, 3.0)
        elapsed = time.time() - self.created_at
        remaining = max(0, duration - elapsed)
        return min(1.0, remaining / 0.5)


class AlertManager:
    def __init__(self, sound_enabled: bool = True):
        self._alerts: List[Alert] = []
        self._last_triggered: dict = {}
        self._trigger_cooldown = 1.5  # seconds
        self._sound = SoundManager(enabled=sound_enabled)

    def trigger(self, message: str, severity: str = "warning",
                violation_type: str = None):
        """
        Trigger a new alert.
        - Displays on-screen banner
        - Plays audio beep pattern matching the violation/severity
        """
        now = time.time()
        key = f"{severity}:{message}"
        if now - self._last_triggered.get(key, 0) < self._trigger_cooldown:
            return
        self._last_triggered[key] = now

        # Resolve violation type from message if not provided
        vtype = violation_type or _message_to_violation(message)

        alert = Alert(message=message, severity=severity, violation_type=vtype)
        self._alerts.append(alert)

        # 🔊 Play audio alert (non-blocking background thread)
        self._sound.play_alert(message=message, severity=severity,
                               violation_type=vtype)

    def get_active_alerts(self) -> List[Alert]:
        self._alerts = [a for a in self._alerts if a.is_active]
        order = {"critical": 0, "warning": 1, "info": 2}
        return sorted(self._alerts, key=lambda a: order.get(a.severity, 3))[:5]

    def set_sound_enabled(self, enabled: bool):
        self._sound.enabled = enabled
