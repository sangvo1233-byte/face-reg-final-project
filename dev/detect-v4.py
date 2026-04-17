"""
dev/detect-v4.py - Local research scanner for stronger anti-spoof testing.

V4 keeps dev/detect-v3.py intact and adds:
  1. Multi-band FFT moire analysis at 128x128.
  2. Per-face rolling moire decision.
  3. Challenge-first policy for suspicious frames.
  4. Calibration logging for real-world threshold tuning.

This file is intentionally local/OpenCV only. It does not change the web
dashboard, API contracts, or database schema.

Controls:
    Q  - Quit
    S  - Screenshot
    R  - Reload embeddings cache
    L  - Toggle landmarks
    I  - Toggle info panel
    M  - Toggle moire spectrum/debug overlay
    D  - Toggle detailed diagnostics panel
    C  - Toggle calibration JSONL logging
    B  - Cycle security view label
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Make the project root importable when running `python dev/detect-v4.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.anti_spoof import get_anti_spoof
from core.database import get_db
from core.face_engine import get_engine
from core.liveness import StreamingLivenessTracker


# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------

CAM_INDEX = 0
FRAME_W = 1280
FRAME_H = 720
DETECT_FPS = 10
MOIRE_EVERY_N_DETECT = 3
V4_COSINE_THRESHOLD = 0.52

SCREENSHOT_DIR = Path(__file__).resolve().parent / "screenshots"
LOG_DIR = Path(__file__).resolve().parent / "logs"
METRICS_LOG_PATH = LOG_DIR / "detect_v4_metrics.jsonl"

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX

CHALLENGE_TYPES = ["BLINK", "TURN_LEFT", "TURN_RIGHT"]
CHALLENGE_TEXT = {
    "BLINK": "Blink once",
    "TURN_LEFT": "Turn LEFT",
    "TURN_RIGHT": "Turn RIGHT",
}

CHALLENGE_TIMEOUT = 7.0
CHALLENGE_COOLDOWN = 12.0
CHALLENGE_BASELINE_FRAMES = 5
CHALLENGE_POSE_FRAMES = 2
TURN_THRESHOLD = 0.06


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

C_PRESENT = (80, 220, 80)
C_UNKNOWN = (40, 160, 255)
C_SPOOF = (60, 60, 220)
C_WHITE = (245, 245, 245)
C_BLACK = (10, 10, 10)
C_DIM = (170, 170, 175)
C_PANEL_BG = (28, 28, 32)
C_CYAN = (220, 200, 40)
C_PURPLE = (180, 60, 200)
C_ORANGE = (40, 140, 255)
C_BLUE = (220, 140, 40)
C_GREEN_DARK = (20, 90, 45)


def _get_screen_info() -> tuple[int, int, float]:
    screen_w, screen_h = 1920, 1080
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        screen_w = ctypes.windll.user32.GetSystemMetrics(0)
        screen_h = ctypes.windll.user32.GetSystemMetrics(1)
    except Exception:
        pass
    scale = max(0.6, min(2.5, screen_h / 1080.0))
    return screen_w, screen_h, scale


SCREEN_W, SCREEN_H, UI_SCALE = _get_screen_info()
INFO_PANEL_W = int(410 * UI_SCALE)


def S(value: int | float) -> int:
    return int(value * UI_SCALE)


def FS(value: float) -> float:
    return value * UI_SCALE


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def bbox_list(bbox: Any) -> list[int]:
    return [int(v) for v in bbox[:4]]


def safe_roi(frame: np.ndarray, bbox: list[int]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Moire V4
# ---------------------------------------------------------------------------


class MoireDetectorV4:
    """Multi-band FFT moire detector for local research.

    The output keeps the old fields used by V2/V3, but adds band-level
    diagnostics and a decision hint.
    """

    ANALYZE_SIZE = 128
    BANDS = {
        "low_mid": (0.15, 0.35),
        "mid": (0.35, 0.55),
        "mid_high": (0.55, 0.75),
        "high": (0.75, 0.92),
    }
    BAND_WEIGHTS = {
        "low_mid": 0.75,
        "mid": 1.00,
        "mid_high": 1.25,
        "high": 0.90,
    }

    PEAK_WEAK = 2.25
    PEAK_STRONG = 3.25
    PERIODIC_WEAK = 3.0
    PERIODIC_STRONG = 5.0

    def __init__(self):
        self.history: list[float] = []
        self.max_history = 20
        self.last_result: dict[str, Any] = {}

        sz = self.ANALYZE_SIZE
        cy, cx = sz // 2, sz // 2
        y, x = np.ogrid[:sz, :sz]
        self._dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.float32)
        self._window = np.outer(np.hanning(sz), np.hanning(sz)).astype(np.float32)
        self._dc_mask = self._dist > 0

        max_r = min(cx, cy)
        self._band_defs: dict[str, dict[str, Any]] = {}
        for name, (low, high) in self.BANDS.items():
            r_low = max(1, int(max_r * low))
            r_high = min(max_r - 1, int(max_r * high))
            self._band_defs[name] = {
                "r_low": r_low,
                "r_high": r_high,
                "mask": (self._dist >= r_low) & (self._dist <= r_high),
            }

        self._wide_mask = (
            (self._dist >= self._band_defs["low_mid"]["r_low"])
            & (self._dist <= self._band_defs["high"]["r_high"])
        )

    def analyze(self, face_roi: np.ndarray) -> dict[str, Any]:
        if face_roi is None or face_roi.size == 0:
            return self._empty_result()

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.ANALYZE_SIZE, self.ANALYZE_SIZE))
        gray_f = gray.astype(np.float32) / 255.0
        log_mag = self._fft_log_magnitude(gray_f)

        total_energy = float(np.sum(log_mag[self._dc_mask]))
        if total_energy < 1e-6:
            total_energy = 1e-6

        band_scores = {}
        evidence = 0.0
        strong_signals: list[str] = []

        for band, definition in self._band_defs.items():
            metrics = self._band_metrics(
                band=band,
                log_mag=log_mag,
                total_energy=total_energy,
                r_low=definition["r_low"],
                r_high=definition["r_high"],
                mask=definition["mask"],
            )
            band_scores[band] = metrics
            band_evidence, band_signals = self._band_evidence(band, metrics)
            evidence += band_evidence
            strong_signals.extend(band_signals)

        h_line = self._check_grid_lines(gray, "horizontal")
        v_line = self._check_grid_lines(gray, "vertical")
        grid_score = float((h_line + v_line) / 2.0)
        if grid_score >= 0.55:
            evidence += 0.22
            strong_signals.append("strong_grid")
        elif grid_score >= 0.34:
            evidence += 0.10
            strong_signals.append("weak_grid")

        high_energy = (
            band_scores["mid_high"]["band_energy"]
            + band_scores["high"]["band_energy"]
        )
        if high_energy > 0.52:
            evidence += 0.12
            strong_signals.append("high_frequency_energy")
        elif high_energy > 0.43:
            evidence += 0.06

        screen_evidence = clamp(evidence)
        raw_score = round(1.0 - screen_evidence, 3)

        self.history.append(raw_score)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        avg_score = round(float(np.mean(self.history)), 3)

        strong_count = len([s for s in strong_signals if not s.startswith("weak")])
        if screen_evidence >= 0.68 and strong_count >= 2:
            decision = "block"
        elif screen_evidence >= 0.38 or strong_signals:
            decision = "suspicious"
        else:
            decision = "clean"

        wide_values = log_mag[self._wide_mask]
        peak_ratio = float(np.max(wide_values) / max(np.mean(wide_values), 1e-6))
        energy_ratio = float(high_energy)
        periodicity = float(max(m["periodicity"] for m in band_scores.values()))

        result = {
            "moire_score": avg_score,
            "raw_score": raw_score,
            "is_screen": decision == "block",
            "decision_hint": decision,
            "screen_evidence": round(screen_evidence, 3),
            "strong_signals": strong_signals[:8],
            "band_scores": band_scores,
            "peak_ratio": round(peak_ratio, 2),
            "energy_ratio": round(energy_ratio, 3),
            "periodicity": round(periodicity, 2),
            "grid_score": round(grid_score, 3),
        }
        self.last_result = result
        return result

    def _empty_result(self) -> dict[str, Any]:
        return {
            "moire_score": 0.5,
            "raw_score": 0.5,
            "is_screen": None,
            "decision_hint": "clean",
            "screen_evidence": 0.0,
            "strong_signals": [],
            "band_scores": {},
            "peak_ratio": 0.0,
            "energy_ratio": 0.0,
            "periodicity": 0.0,
            "grid_score": 0.0,
        }

    def _fft_log_magnitude(self, gray_f: np.ndarray) -> np.ndarray:
        windowed = gray_f * self._window
        fft = np.fft.fft2(windowed)
        return np.log1p(np.abs(np.fft.fftshift(fft)))

    def _band_metrics(
        self,
        *,
        band: str,
        log_mag: np.ndarray,
        total_energy: float,
        r_low: int,
        r_high: int,
        mask: np.ndarray,
    ) -> dict[str, float]:
        values = log_mag[mask]
        if values.size == 0 or float(np.mean(values)) < 1e-6:
            return {
                "peak_ratio": 0.0,
                "energy_ratio": 0.0,
                "periodicity": 0.0,
                "band_energy": 0.0,
            }

        peak_ratio = float(np.max(values) / max(np.mean(values), 1e-6))
        band_energy = float(np.sum(values) / total_energy)
        periodicity = self._check_periodicity(log_mag, r_low, r_high)
        return {
            "peak_ratio": round(peak_ratio, 2),
            "energy_ratio": round(band_energy, 3),
            "periodicity": round(periodicity, 2),
            "band_energy": round(band_energy, 3),
        }

    def _band_evidence(self, band: str, metrics: dict[str, float]) -> tuple[float, list[str]]:
        weight = self.BAND_WEIGHTS.get(band, 1.0)
        evidence = 0.0
        signals: list[str] = []

        peak = metrics["peak_ratio"]
        periodicity = metrics["periodicity"]

        if peak >= self.PEAK_STRONG:
            evidence += 0.11 * weight
            signals.append(f"{band}_peak")
        elif peak >= self.PEAK_WEAK:
            evidence += 0.05 * weight
            signals.append(f"weak_{band}_peak")

        if periodicity >= self.PERIODIC_STRONG:
            evidence += 0.12 * weight
            signals.append(f"{band}_periodic")
        elif periodicity >= self.PERIODIC_WEAK:
            evidence += 0.055 * weight
            signals.append(f"weak_{band}_periodic")

        if band in {"mid_high", "high"} and metrics["band_energy"] >= 0.28:
            evidence += 0.05 * weight
            signals.append(f"{band}_energy_spike")

        return evidence, signals

    def _check_periodicity(self, log_mag: np.ndarray, r_low: int, r_high: int) -> float:
        sz = self.ANALYZE_SIZE
        cx = cy = sz // 2
        peak_counts = []
        n_angles = 48
        for i in range(n_angles):
            angle = i * np.pi / n_angles
            samples = []
            for r in range(r_low, r_high, 2):
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))
                if 0 <= x < sz and 0 <= y < sz:
                    samples.append(log_mag[y, x])

            if len(samples) < 5:
                continue
            arr = np.asarray(samples, dtype=np.float32)
            mean_val = float(np.mean(arr))
            if mean_val < 1e-6:
                continue
            peak_counts.append(int(np.sum(arr > mean_val * 2.0)))

        if not peak_counts:
            return 0.0
        return float(np.max(peak_counts))

    def _check_grid_lines(self, gray: np.ndarray, direction: str) -> float:
        if direction == "horizontal":
            projection = np.mean(gray.astype(np.float32), axis=1)
        else:
            projection = np.mean(gray.astype(np.float32), axis=0)

        if len(projection) < 16:
            return 0.0

        centered = projection - np.mean(projection)
        fft_1d = np.abs(np.fft.rfft(centered))
        if len(fft_1d) < 4:
            return 0.0

        mid_start = len(fft_1d) // 4
        mid_end = 3 * len(fft_1d) // 4
        mid_band = fft_1d[mid_start:mid_end]
        if mid_band.size == 0 or float(np.mean(mid_band)) < 1e-6:
            return 0.0

        ratio = float(np.max(mid_band) / max(np.mean(mid_band), 1e-6))
        return clamp((ratio - 1.5) / 3.0)


@dataclass
class RollingMoireDecision:
    maxlen: int = 18
    samples: deque = field(default_factory=lambda: deque(maxlen=18))

    def update(self, result: dict[str, Any]) -> dict[str, Any]:
        if self.samples.maxlen != self.maxlen:
            self.samples = deque(self.samples, maxlen=self.maxlen)
        self.samples.append(result)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {
                "decision": "checking",
                "samples": 0,
                "mean_score": 1.0,
                "min_score": 1.0,
                "p10_score": 1.0,
                "suspicious_count": 0,
                "block_count": 0,
                "clean_count": 0,
            }

        scores = np.asarray([float(s.get("raw_score", 0.5)) for s in self.samples], dtype=np.float32)
        hints = [s.get("decision_hint", "clean") for s in self.samples]
        suspicious_count = sum(1 for h in hints if h in {"suspicious", "block"})
        block_count = sum(1 for h in hints if h == "block")
        clean_count = sum(1 for h in hints if h == "clean")
        samples = len(self.samples)

        mean_score = float(np.mean(scores))
        min_score = float(np.min(scores))
        p10_score = float(np.percentile(scores, 10))

        decision = "checking"
        if samples >= 4 and (block_count >= 4 or (p10_score < 0.22 and suspicious_count >= 4)):
            decision = "block"
        elif samples >= 3 and (
            suspicious_count >= 3
            or min_score < 0.30
            or p10_score < 0.38
        ):
            decision = "suspicious"
        elif samples >= 4 and clean_count >= 3 and mean_score > 0.55:
            decision = "clean"

        return {
            "decision": decision,
            "samples": samples,
            "mean_score": round(mean_score, 3),
            "min_score": round(min_score, 3),
            "p10_score": round(p10_score, 3),
            "suspicious_count": suspicious_count,
            "block_count": block_count,
            "clean_count": clean_count,
        }


# ---------------------------------------------------------------------------
# Challenge V4
# ---------------------------------------------------------------------------


class ChallengeControllerV4:
    def __init__(self):
        self.active = False
        self.current_type = ""
        self.text = ""
        self.started_at = 0.0
        self.last_pass_at = 0.0
        self.last_student_id = ""
        self.fail_count = 0
        self.blink_baseline = 0
        self.pose_baseline = None
        self.pose_baseline_frames = 0
        self.pose_frames_passed = 0
        self.message = ""

    def can_skip_for(self, student_id: str) -> bool:
        return (
            bool(student_id)
            and self.last_student_id == student_id
            and time.time() - self.last_pass_at < CHALLENGE_COOLDOWN
        )

    def start(self, reason: str, candidate: dict[str, Any], live: dict[str, Any]):
        self.active = True
        self.current_type = random.choice(CHALLENGE_TYPES)
        self.text = CHALLENGE_TEXT[self.current_type]
        self.started_at = time.time()
        self.last_student_id = candidate.get("student_id", "")
        self.blink_baseline = int(live.get("blinks", 0))
        self.pose_baseline = None
        self.pose_baseline_frames = 0
        self.pose_frames_passed = 0
        self.message = f"{self.text} - {reason}"
        print(f"[V4 CHALLENGE] {self.text} for {self.last_student_id}: {reason}")

    def update(
        self,
        *,
        frame: np.ndarray,
        face: Any,
        match: Any,
        live: dict[str, Any],
        rolling: dict[str, Any],
        metrics: dict[str, Any],
    ) -> str:
        if not self.active:
            return "inactive"

        if time.time() - self.started_at > CHALLENGE_TIMEOUT:
            return self.fail("timeout")

        if not match.matched or match.student_id != self.last_student_id:
            return self.fail("identity mismatch")

        if rolling.get("decision") == "block":
            return self.fail("screen detected during challenge")

        if self.current_type == "BLINK":
            if int(live.get("blinks", 0)) > self.blink_baseline:
                return self.pass_challenge()
            self.message = f"{self.text} ({self.remaining():.1f}s)"
            return "active"

        if self._check_turn(float(metrics.get("nose_x_disp", 0.0))):
            return self.pass_challenge()

        self.message = f"{self.text} ({self.remaining():.1f}s)"
        return "active"

    def fail(self, reason: str) -> str:
        self.active = False
        self.fail_count += 1
        self.message = f"Challenge failed: {reason}"
        print(f"[V4 CHALLENGE] FAILED: {reason}")
        return "failed"

    def pass_challenge(self) -> str:
        self.active = False
        self.last_pass_at = time.time()
        self.fail_count = 0
        self.message = "Challenge passed"
        print("[V4 CHALLENGE] PASSED")
        return "passed"

    def remaining(self) -> float:
        return max(0.0, CHALLENGE_TIMEOUT - (time.time() - self.started_at))

    def _check_turn(self, nose_x_disp: float) -> bool:
        if self.pose_baseline_frames < CHALLENGE_BASELINE_FRAMES:
            self.pose_baseline = (
                nose_x_disp
                if self.pose_baseline is None
                else self.pose_baseline * 0.7 + nose_x_disp * 0.3
            )
            self.pose_baseline_frames += 1
            return False

        baseline = self.pose_baseline or 0.0
        dx = nose_x_disp - baseline
        if self.current_type == "TURN_LEFT" and dx >= TURN_THRESHOLD:
            self.pose_frames_passed += 1
        elif self.current_type == "TURN_RIGHT" and dx <= -TURN_THRESHOLD:
            self.pose_frames_passed += 1
        else:
            self.pose_frames_passed = max(0, self.pose_frames_passed - 1)

        return self.pose_frames_passed >= CHALLENGE_POSE_FRAMES


# ---------------------------------------------------------------------------
# Calibration logging
# ---------------------------------------------------------------------------


class CalibrationLogger:
    def __init__(self, path: Path):
        self.path = path
        self.enabled = False

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[V4 LOG] Calibration logging ON: {self.path}")
        else:
            print("[V4 LOG] Calibration logging OFF")
        return self.enabled

    def write(self, payload: dict[str, Any]):
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        safe = {
            "ts": round(time.time(), 3),
            "final_status": payload.get("status", ""),
            "match_score": round(float(payload.get("match_score", 0.0)), 4),
            "matched": bool(payload.get("matched", False)),
            "moire": payload.get("moire", {}),
            "rolling": payload.get("rolling", {}),
            "liveness": payload.get("liveness", {}),
            "passive": payload.get("passive", {}),
            "challenge": payload.get("challenge", {}),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe, ensure_ascii=True) + "\n")


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_rounded_rect(img, pt1, pt2, color, thickness=2, radius=12):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_label_box(img, title, line_1, line_2, x1, y1, color):
    x1 = max(S(10), x1)
    y_top = max(S(10), y1 - S(78))
    w = S(360)
    h = S(70)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y_top), (x1 + w, y_top + h), (22, 22, 26), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
    cv2.rectangle(img, (x1, y_top), (x1 + w, y_top + h), color, 1)
    cv2.putText(img, title[:28], (x1 + S(10), y_top + S(24)), FONT, FS(0.52), C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, line_1[:38], (x1 + S(10), y_top + S(45)), FONT_SMALL, FS(0.44), color, 1, cv2.LINE_AA)
    if line_2:
        cv2.putText(img, line_2[:46], (x1 + S(10), y_top + S(62)), FONT_SMALL, FS(0.38), C_DIM, 1, cv2.LINE_AA)


def draw_landmarks(img, landmarks, color=C_CYAN):
    if landmarks is None:
        return
    for x, y in landmarks[:5]:
        cv2.circle(img, (int(x), int(y)), S(3), color, -1, cv2.LINE_AA)


def conf_bar(img, x1, y2, x2, confidence, color):
    y = min(img.shape[0] - S(12), y2 + S(8))
    width = max(1, x2 - x1)
    fill = int(width * clamp(confidence))
    cv2.rectangle(img, (x1, y), (x2, y + S(5)), (50, 50, 55), -1)
    cv2.rectangle(img, (x1, y), (x1 + fill, y + S(5)), color, -1)


def draw_moire_badge(img, x2, y2, moire, rolling):
    if not moire:
        return
    decision = rolling.get("decision") or moire.get("decision_hint", "clean")
    score = float(moire.get("raw_score", moire.get("moire_score", 1.0)))
    if decision == "block":
        label, color = "SCREEN", C_SPOOF
    elif decision == "suspicious":
        label, color = "SUSPECT", C_ORANGE
    elif decision == "checking":
        label, color = "CHECK", C_UNKNOWN
    else:
        label, color = "CLEAN", C_PRESENT
    text = f"{label} {score:.0%}"
    bx = max(S(8), x2 - S(120))
    by = min(img.shape[0] - S(34), y2 + S(18))
    cv2.rectangle(img, (bx, by), (bx + S(116), by + S(28)), (25, 25, 28), -1)
    cv2.rectangle(img, (bx, by), (bx + S(116), by + S(28)), color, 1)
    cv2.putText(img, text, (bx + S(8), by + S(19)), FONT_SMALL, FS(0.42), color, 1, cv2.LINE_AA)


def draw_moire_spectrum(img, face_roi, x1, y1):
    if face_roi is None or face_roi.size == 0:
        return
    try:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        window = np.outer(np.hanning(64), np.hanning(64))
        mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray * window))))
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mag = cv2.applyColorMap(mag, cv2.COLORMAP_VIRIDIS)
        sx = max(0, x1)
        sy = max(S(10), y1 - S(150))
        if sy + 64 < img.shape[0] and sx + 64 < img.shape[1]:
            img[sy:sy + 64, sx:sx + 64] = mag
            cv2.putText(img, "FFT", (sx + 2, sy + 12), FONT_SMALL, 0.35, C_WHITE, 1, cv2.LINE_AA)
    except Exception:
        return


def draw_challenge_overlay(img, challenge: ChallengeControllerV4):
    if not challenge.active and not challenge.message:
        return

    h, w = img.shape[:2]
    overlay = img.copy()
    banner_h = S(90)
    cv2.rectangle(overlay, (S(40), S(32)), (w - S(40), S(32) + banner_h), (28, 28, 32), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)

    color = C_ORANGE if challenge.active else (C_PRESENT if "passed" in challenge.message.lower() else C_SPOOF)
    cv2.rectangle(img, (S(40), S(32)), (w - S(40), S(32) + banner_h), color, 2)
    label = "ACTIVE CHALLENGE" if challenge.active else "CHALLENGE STATUS"
    cv2.putText(img, label, (S(62), S(62)), FONT_SMALL, FS(0.50), C_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, challenge.message[:70], (S(62), S(100)), FONT, FS(0.78), color, 2, cv2.LINE_AA)


def draw_hud(img, fps, face_count, total_students, logging_on, policy_view):
    cv2.putText(img, "Detect V4 Research", (S(14), S(26)), FONT, FS(0.56), C_WHITE, 1, cv2.LINE_AA)
    meta = (
        f"FPS:{fps:.1f}  Faces:{face_count}  Students:{total_students}  "
        f"Log:{'ON' if logging_on else 'OFF'}  View:{policy_view.upper()}"
    )
    cv2.putText(img, meta, (S(14), S(52)), FONT_SMALL, FS(0.43), C_DIM, 1, cv2.LINE_AA)
    controls = "Q quit | S shot | R reload | L landmarks | I panel | M moire | D debug | C log | B view"
    cv2.putText(img, controls, (S(14), img.shape[0] - S(14)), FONT_SMALL, FS(0.38), C_DIM, 1, cv2.LINE_AA)


def draw_info_panel(frame, results, challenge, debug_enabled, panel_w=INFO_PANEL_W):
    h, w = frame.shape[:2]
    panel = np.full((h, panel_w, 3), C_PANEL_BG, dtype=np.uint8)
    y = S(26)
    cv2.putText(panel, "DETECT V4", (S(14), y), FONT, FS(0.55), C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Threshold: {V4_COSINE_THRESHOLD:.2f}", (S(14), y + S(26)), FONT_SMALL, FS(0.42), C_DIM, 1, cv2.LINE_AA)
    y += S(70)

    if challenge.active or challenge.message:
        cv2.putText(panel, "Challenge", (S(14), y), FONT_SMALL, FS(0.46), C_ORANGE, 1, cv2.LINE_AA)
        y += S(22)
        cv2.putText(panel, challenge.message[:34], (S(14), y), FONT_SMALL, FS(0.40), C_WHITE, 1, cv2.LINE_AA)
        y += S(34)

    for idx, r in enumerate(results[:4]):
        color = C_PRESENT if r["status"] == "present" else C_SPOOF if r["status"] == "spoof" else C_UNKNOWN
        cv2.putText(panel, f"Face {idx + 1}: {r['status'].upper()}", (S(14), y), FONT_SMALL, FS(0.45), color, 1, cv2.LINE_AA)
        y += S(20)
        cv2.putText(panel, r.get("name", "Unknown")[:32], (S(14), y), FONT_SMALL, FS(0.42), C_WHITE, 1, cv2.LINE_AA)
        y += S(20)
        cv2.putText(panel, r.get("label", "")[:38], (S(14), y), FONT_SMALL, FS(0.38), C_DIM, 1, cv2.LINE_AA)
        y += S(18)

        moire = r.get("moire", {})
        rolling = r.get("rolling", {})
        m_text = (
            f"Moire {rolling.get('decision', 'n/a')}: "
            f"raw={moire.get('raw_score', 0):.2f} mean={rolling.get('mean_score', 0):.2f}"
        )
        cv2.putText(panel, m_text[:42], (S(14), y), FONT_SMALL, FS(0.36), C_PURPLE, 1, cv2.LINE_AA)
        y += S(18)

        live = r.get("liveness", {})
        l_text = f"Live {live.get('state', 'n/a')} blink={live.get('blinks', 0)} ear={live.get('ear', 0)}"
        cv2.putText(panel, l_text[:42], (S(14), y), FONT_SMALL, FS(0.36), C_CYAN, 1, cv2.LINE_AA)
        y += S(18)

        passive = r.get("passive", {})
        p_text = f"Passive {passive.get('score', 0):.2f} {passive.get('reason', '')[:20]}"
        cv2.putText(panel, p_text[:42], (S(14), y), FONT_SMALL, FS(0.35), C_DIM, 1, cv2.LINE_AA)
        y += S(18)

        if debug_enabled:
            signals = ", ".join(moire.get("strong_signals", [])[:3])
            cv2.putText(panel, signals[:42], (S(14), y), FONT_SMALL, FS(0.32), C_ORANGE, 1, cv2.LINE_AA)
            y += S(18)

        y += S(14)
        if y > h - S(80):
            break

    return np.hstack([frame, panel])


# ---------------------------------------------------------------------------
# Detection decision
# ---------------------------------------------------------------------------


def passive_decision(score: float) -> str:
    if score < config.DETECT_V3_LIVENESS_BLOCK_THRESHOLD:
        return "block"
    if score < config.DETECT_V3_LIVENESS_CHALLENGE_THRESHOLD:
        return "suspicious"
    return "pass"


def build_candidate(match, student: dict, emb_count: int) -> dict[str, Any]:
    return {
        "name": match.name,
        "student_id": match.student_id,
        "class_name": student.get("class_name", ""),
        "confidence": float(match.score),
        "embedding_count": emb_count,
    }


def process_face(
    *,
    frame: np.ndarray,
    face: Any,
    face_index: int,
    engine: Any,
    db: Any,
    anti_spoof: Any,
    liveness: StreamingLivenessTracker,
    moire_detector: MoireDetectorV4,
    rolling_by_face: dict[int, RollingMoireDecision],
    last_moire: dict[int, dict[str, Any]],
    last_rolling: dict[int, dict[str, Any]],
    run_moire: bool,
    challenge: ChallengeControllerV4,
    logger: CalibrationLogger,
) -> dict[str, Any]:
    bbox = bbox_list(face.bbox)
    x1, y1, x2, y2 = bbox
    roi = safe_roi(frame, bbox)

    if run_moire or face_index not in last_moire:
        moire = moire_detector.analyze(roi)
        rolling = rolling_by_face[face_index].update(moire)
        last_moire[face_index] = moire
        last_rolling[face_index] = rolling
    else:
        moire = last_moire.get(face_index, moire_detector.last_result or {})
        rolling = last_rolling.get(face_index, rolling_by_face[face_index].summary())

    live = liveness.get_liveness(bbox)
    passive = anti_spoof.check(frame, face.bbox)
    passive_status = passive_decision(float(passive.score))
    metrics = engine.get_face_metrics(frame, face)
    match = engine.match_with_threshold(face.embedding, V4_COSINE_THRESHOLD)

    name = "Unknown"
    label = f"No match ({match.score:.0%})"
    extra = ""
    status = "unknown"
    color = C_UNKNOWN
    student = {}
    emb_count = 0

    if rolling.get("decision") == "block":
        status = "spoof"
        name = "SPOOF"
        label = f"SCREEN detected ({moire.get('raw_score', 0):.0%})"
        extra = "Rolling moire block"
        color = C_SPOOF
    elif live.get("state") == "spoof":
        status = "spoof"
        name = "SPOOF"
        label = live.get("message", "Liveness failed")
        extra = f"Blinks:{live.get('blinks', 0)} Track:{live.get('track_time', 0):.1f}s"
        color = C_SPOOF
    elif passive_status == "block":
        status = "spoof"
        name = "SPOOF"
        label = f"Passive failed ({passive.score:.2f})"
        extra = passive.reason
        color = C_SPOOF
    elif live.get("state") != "live":
        status = "checking"
        name = "Checking..."
        label = live.get("message", "Tracking liveness")
        extra = f"Blinks:{live.get('blinks', 0)} EAR:{live.get('ear', 0):.2f}"
        color = C_UNKNOWN
    elif not match.matched:
        status = "unknown"
        name = "Unknown"
        label = f"Score {match.score:.0%} < {V4_COSINE_THRESHOLD:.0%}"
        extra = f"Moire:{rolling.get('decision', 'n/a')} Live:OK"
        color = C_UNKNOWN
    else:
        student = db.get_student(match.student_id) or {}
        emb_count = db.get_embedding_count(match.student_id)
        candidate = build_candidate(match, student, emb_count)
        suspicious_reasons = []
        if rolling.get("decision") == "suspicious":
            suspicious_reasons.append("moire suspicious")
        if passive_status == "suspicious":
            suspicious_reasons.append(f"passive {passive.score:.2f}")
        if challenge.fail_count >= 2:
            suspicious_reasons.append("recent challenge failures")

        if suspicious_reasons and not challenge.can_skip_for(match.student_id):
            if not challenge.active:
                challenge.start("; ".join(suspicious_reasons), candidate, live)
            result = challenge.update(
                frame=frame,
                face=face,
                match=match,
                live=live,
                rolling=rolling,
                metrics=metrics,
            )
            if result == "passed":
                status = "present"
            elif result == "failed":
                status = "spoof"
            else:
                status = "challenge"
        elif challenge.active:
            result = challenge.update(
                frame=frame,
                face=face,
                match=match,
                live=live,
                rolling=rolling,
                metrics=metrics,
            )
            if result == "passed":
                status = "present"
            elif result == "failed":
                status = "spoof"
            else:
                status = "challenge"
        else:
            status = "present"

        if status == "present":
            name = match.name
            label = f"ID:{match.student_id} Conf:{match.score:.0%}"
            extra = f"Emb:{emb_count} Moire:{rolling.get('decision')} Live:OK"
            color = C_PRESENT
        elif status == "challenge":
            name = "VERIFY"
            label = challenge.text
            extra = challenge.message
            color = C_ORANGE
        else:
            name = "SPOOF"
            label = challenge.message or "Challenge failed"
            extra = "No attendance"
            color = C_SPOOF

    entry = {
        "bbox": bbox,
        "landmarks": face.landmarks,
        "name": name,
        "label": label,
        "extra": extra,
        "status": status,
        "color": color,
        "confidence": float(match.score),
        "matched": bool(match.matched),
        "match_score": float(match.score),
        "student_id": match.student_id if match.matched else "",
        "class_name": student.get("class_name", ""),
        "embedding_count": emb_count,
        "moire": moire,
        "rolling": rolling,
        "liveness": live,
        "passive": {
            "score": float(passive.score),
            "reason": passive.reason,
            "decision": passive_status,
        },
        "metrics": metrics,
        "challenge": {
            "active": challenge.active,
            "type": challenge.current_type,
            "message": challenge.message,
            "fail_count": challenge.fail_count,
        },
    }
    logger.write(entry)
    return entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("  Detect V4 Research - Multi-band moire + rolling + challenge-first")
    print("=" * 72)
    print("This script is local/OpenCV only and does not change production web.")

    db = get_db()
    engine = get_engine()
    print("Loading InsightFace and embeddings...")
    engine._ensure_model()
    engine.reload_cache()

    anti_spoof = get_anti_spoof()
    liveness = StreamingLivenessTracker()
    moire_detector = MoireDetectorV4()
    challenge = ChallengeControllerV4()
    cal_logger = CalibrationLogger(METRICS_LOG_PATH)

    emb_counts = {}
    try:
        for student in db.get_all_students():
            emb_counts[student["id"]] = db.get_embedding_count(student["id"])
    except Exception:
        pass
    print(f"DB: {len(emb_counts)} students, {sum(emb_counts.values())} embeddings")
    print(f"Cosine threshold: {V4_COSINE_THRESHOLD}")
    print("Opening camera...")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {CAM_INDEX}")
        return

    win_name = "Live Face Detection V4"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    total_w = FRAME_W + INFO_PANEL_W
    total_h = FRAME_H
    fit_scale = min((SCREEN_W * 0.92) / total_w, (SCREEN_H * 0.90) / total_h, 1.0)
    cv2.resizeWindow(win_name, int(total_w * fit_scale), int(total_h * fit_scale))

    rolling_by_face: dict[int, RollingMoireDecision] = defaultdict(RollingMoireDecision)
    last_moire: dict[int, dict[str, Any]] = {}
    last_rolling: dict[int, dict[str, Any]] = {}
    last_results: list[dict[str, Any]] = []

    last_detect_at = 0.0
    detect_interval = 1.0 / DETECT_FPS
    detect_cycle = 0
    show_landmarks = True
    show_info_panel = True
    show_moire_overlay = True
    debug_enabled = False
    policy_views = ["clean", "suspicious", "blocked"]
    policy_view_index = 0

    fps_counter = 0
    fps_start = time.time()
    display_fps = 0.0
    perf_detect_ms = 0.0
    perf_moire_ms = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed, retrying...")
                time.sleep(0.05)
                continue

            now = time.time()
            try:
                liveness.process_frame(frame)
            except Exception as exc:
                print(f"[WARN] liveness frame failed: {exc}")

            if now - last_detect_at >= detect_interval:
                last_detect_at = now
                detect_cycle += 1
                run_moire = detect_cycle % MOIRE_EVERY_N_DETECT == 0
                t_detect = time.time()
                if run_moire:
                    t_moire = time.time()
                try:
                    faces = engine.detect(frame)
                    if run_moire:
                        perf_moire_ms = (time.time() - t_moire) * 1000.0
                    last_results = []
                    if not faces:
                        # Clear stale per-index rolling when no face remains visible.
                        if detect_cycle % (MOIRE_EVERY_N_DETECT * 8) == 0:
                            last_moire.clear()
                            last_rolling.clear()
                            rolling_by_face.clear()

                    for idx, face in enumerate(faces):
                        if face.embedding is None or len(face.embedding) == 0:
                            continue
                        entry = process_face(
                            frame=frame,
                            face=face,
                            face_index=idx,
                            engine=engine,
                            db=db,
                            anti_spoof=anti_spoof,
                            liveness=liveness,
                            moire_detector=moire_detector,
                            rolling_by_face=rolling_by_face,
                            last_moire=last_moire,
                            last_rolling=last_rolling,
                            run_moire=run_moire,
                            challenge=challenge,
                            logger=cal_logger,
                        )
                        last_results.append(entry)
                except Exception as exc:
                    print(f"Detect error: {exc}")
                    import traceback

                    traceback.print_exc()
                perf_detect_ms = (time.time() - t_detect) * 1000.0

            # Draw results on the current frame.
            for result in last_results:
                x1, y1, x2, y2 = result["bbox"]
                color = result["color"]
                draw_rounded_rect(frame, (x1, y1), (x2, y2), color, thickness=2)
                if show_landmarks:
                    draw_landmarks(frame, result.get("landmarks"), color=C_CYAN)
                draw_label_box(
                    frame,
                    result["name"],
                    result["label"],
                    result["extra"],
                    x1,
                    y1,
                    color,
                )
                conf_bar(frame, x1, y2, x2, result.get("confidence", 0.0), color)
                draw_moire_badge(frame, x2, y2, result.get("moire"), result.get("rolling", {}))
                if show_moire_overlay:
                    draw_moire_spectrum(frame, safe_roi(frame, result["bbox"]), x1, y1)

            draw_challenge_overlay(frame, challenge)
            draw_hud(
                frame,
                display_fps,
                len(last_results),
                len(emb_counts),
                cal_logger.enabled,
                policy_views[policy_view_index],
            )
            perf_text = f"Detect:{perf_detect_ms:.0f}ms  Moire:{perf_moire_ms:.0f}ms  FFT:128  Skip:{MOIRE_EVERY_N_DETECT - 1}/{MOIRE_EVERY_N_DETECT}"
            cv2.putText(frame, perf_text, (S(14), S(78)), FONT_SMALL, FS(0.40), C_DIM, 1, cv2.LINE_AA)

            display_frame = (
                draw_info_panel(frame, last_results, challenge, debug_enabled)
                if show_info_panel
                else frame
            )

            fps_counter += 1
            if now - fps_start >= 1.0:
                display_fps = fps_counter / max(now - fps_start, 1e-6)
                fps_counter = 0
                fps_start = now

            cv2.imshow(win_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit.")
                break
            if key == ord("s"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = SCREENSHOT_DIR / f"detect_v4_{ts}.jpg"
                cv2.imwrite(str(path), display_frame)
                print(f"Screenshot saved: {path}")
            elif key == ord("r"):
                print("Reloading embeddings cache...")
                engine.reload_cache()
                print("Cache reloaded.")
            elif key == ord("l"):
                show_landmarks = not show_landmarks
                print(f"Landmarks: {'ON' if show_landmarks else 'OFF'}")
            elif key == ord("i"):
                show_info_panel = not show_info_panel
                print(f"Info panel: {'ON' if show_info_panel else 'OFF'}")
            elif key == ord("m"):
                show_moire_overlay = not show_moire_overlay
                print(f"Moire overlay: {'ON' if show_moire_overlay else 'OFF'}")
            elif key == ord("d"):
                debug_enabled = not debug_enabled
                print(f"Diagnostics: {'ON' if debug_enabled else 'OFF'}")
            elif key == ord("c"):
                cal_logger.toggle()
            elif key == ord("b"):
                policy_view_index = (policy_view_index + 1) % len(policy_views)
                print(f"Security view: {policy_views[policy_view_index]}")
            elif key == ord("e"):
                print("\n" + "=" * 40)
                print("  EMBEDDING STATS")
                print("=" * 40)
                for student in db.get_all_students():
                    sid = student["id"]
                    count = db.get_embedding_count(sid)
                    tag = "multi-angle" if count >= 3 else "single"
                    print(f"  {student['name']:>22} ({sid}): {count} emb [{tag}]")
                print("=" * 40 + "\n")
    finally:
        liveness.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
