"""
Sound Manager
Cross-platform audio alerts for exam proctoring violations.
Works on Windows (winsound), macOS (afplay/say), Linux (paplay/aplay/beep).

Each severity level and violation type has a distinct sound pattern.
"""

import sys
import time
import threading
import os


# ── Sound profiles ──────────────────────────────────────────────────────────
# (frequency_hz, duration_ms) sequences for Windows winsound.Beep
# For other platforms, number of beeps / pattern conveys severity.

SOUND_PROFILES = {
    "critical": [          # 3 rapid high beeps  🚨
        (1200, 200),
        (1200, 200),
        (1200, 200),
    ],
    "warning": [           # 2 medium beeps  ⚠
        (880, 250),
        (880, 250),
    ],
    "info": [              # 1 short low beep  ℹ
        (600, 180),
    ],
}

# Per-violation custom overrides (played instead of severity default)
VIOLATION_SOUNDS = {
    "PHONE_DETECTED":    [(1400, 150), (1000, 150), (1400, 150), (1000, 300)],
    "MULTIPLE_PERSONS":  [(1200, 300), (1200, 300), (1200, 300)],
    "NO_FACE":           [(800,  400), (800,  400)],
    "LOOKING_AWAY":      [(900,  250), (700,  250)],
    "HEAD_POSE_LEFT":    [(750,  300)],
    "HEAD_POSE_RIGHT":   [(750,  300)],
    "BOOK_DETECTED":     [(1000, 200), (800,  200)],
}

# Cooldown per violation type (seconds) — prevents sound spam
SOUND_COOLDOWN = {
    "PHONE_DETECTED":   5.0,
    "MULTIPLE_PERSONS": 5.0,
    "NO_FACE":          4.0,
    "LOOKING_AWAY":     3.0,
    "critical":         3.0,
    "warning":          2.5,
    "info":             2.0,
}


# ── Platform detection ──────────────────────────────────────────────────────

def _detect_platform():
    if sys.platform == "win32":
        return "windows"
    elif sys.platform == "darwin":
        return "macos"
    else:
        return "linux"


PLATFORM = _detect_platform()


# ── Low-level beep functions ────────────────────────────────────────────────

def _beep_windows(freq: int, duration_ms: int):
    """Windows: use winsound.Beep (built-in, no dependencies)."""
    try:
        import winsound
        winsound.Beep(max(37, min(32767, freq)), duration_ms)
    except Exception:
        pass


def _beep_macos(freq: int, duration_ms: int):
    """macOS: use afplay with a generated sine wave via Python."""
    try:
        # Try using the system beep first (simplest)
        os.system("afplay /System/Library/Sounds/Ping.aiff &")
    except Exception:
        pass


def _beep_linux(freq: int, duration_ms: int):
    """Linux: try paplay system sound, then beep command, then /dev/audio."""
    # Try PulseAudio system sound
    for cmd in [
        "paplay /usr/share/sounds/freedesktop/stereo/bell.oga 2>/dev/null",
        "aplay /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null",
        f"beep -f {freq} -l {duration_ms} 2>/dev/null",
        f"python3 -c \""
        f"import os; os.system('echo -e \\\\a')\"",
    ]:
        ret = os.system(cmd)
        if ret == 0:
            break


def _beep_numpy(freq: int, duration_ms: int):
    """
    Pure Python/NumPy sine-wave beep — works everywhere if sounddevice
    or simpleaudio is installed.
    """
    duration_s = duration_ms / 1000.0
    try:
        import numpy as np
        import sounddevice as sd
        sr = 44100
        t = np.linspace(0, duration_s, int(sr * duration_s), False)
        wave = 0.4 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        sd.play(wave, sr)
        sd.wait()
        return True
    except Exception:
        pass

    try:
        import numpy as np
        import simpleaudio as sa
        sr = 44100
        t = np.linspace(0, duration_s, int(sr * duration_s), False)
        wave = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
        play_obj = sa.play_buffer(wave, 1, 2, sr)
        play_obj.wait_done()
        return True
    except Exception:
        pass

    return False


def _beep(freq: int, duration_ms: int):
    """Platform-aware single beep. Falls back gracefully."""
    played = _beep_numpy(freq, duration_ms)  # try best quality first
    if not played:
        if PLATFORM == "windows":
            _beep_windows(freq, duration_ms)
        elif PLATFORM == "macos":
            _beep_macos(freq, duration_ms)
        else:
            _beep_linux(freq, duration_ms)


def _play_sequence(sequence, gap_ms: int = 80):
    """Play a list of (freq, duration_ms) beeps."""
    for i, (freq, dur) in enumerate(sequence):
        _beep(freq, dur)
        if i < len(sequence) - 1:
            time.sleep(gap_ms / 1000.0)


# ── Public SoundManager class ───────────────────────────────────────────────

class SoundManager:
    """
    Thread-safe sound alert manager.
    Plays distinct beep patterns for each violation type.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._last_played: dict = {}
        self._lock = threading.Lock()
        self._sound_thread: threading.Thread = None

        if enabled:
            print(f"[SOUND] Audio alerts enabled (platform: {PLATFORM})")
            # Quick startup test beep (non-blocking)
            self._play_async([(600, 100)])
        else:
            print("[SOUND] Audio alerts disabled.")

    # ── Public API ──────────────────────────────────────────────────────────

    def play_violation(self, violation_type: str):
        """
        Play the sound for a specific violation type.
        Respects per-violation cooldown to prevent spam.
        """
        if not self.enabled:
            return

        cooldown = SOUND_COOLDOWN.get(violation_type, 2.0)
        now = time.time()

        with self._lock:
            if now - self._last_played.get(violation_type, 0) < cooldown:
                return
            self._last_played[violation_type] = now

        sequence = VIOLATION_SOUNDS.get(violation_type)
        if sequence:
            self._play_async(sequence)

    def play_severity(self, severity: str):
        """
        Play the generic sound for a severity level (critical/warning/info).
        """
        if not self.enabled:
            return

        cooldown = SOUND_COOLDOWN.get(severity, 2.0)
        now = time.time()

        with self._lock:
            key = f"sev:{severity}"
            if now - self._last_played.get(key, 0) < cooldown:
                return
            self._last_played[key] = now

        sequence = SOUND_PROFILES.get(severity, SOUND_PROFILES["info"])
        self._play_async(sequence)

    def play_alert(self, message: str, severity: str, violation_type: str = None):
        """
        Unified entry point called by AlertManager.
        Prefers violation-specific sound; falls back to severity sound.
        """
        if not self.enabled:
            return
        if violation_type and violation_type in VIOLATION_SOUNDS:
            self.play_violation(violation_type)
        else:
            self.play_severity(severity)

    # ── Internal ────────────────────────────────────────────────────────────

    def _play_async(self, sequence):
        """Play sound sequence in a background thread (non-blocking)."""
        # Don't stack threads — skip if one is already running
        if self._sound_thread and self._sound_thread.is_alive():
            return
        self._sound_thread = threading.Thread(
            target=_play_sequence, args=(sequence,), daemon=True
        )
        self._sound_thread.start()
