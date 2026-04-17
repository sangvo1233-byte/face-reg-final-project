"""
dev/detect-v4.py - Local research scanner for stronger anti-spoof testing.

V4 keeps dev/detect-v3.py intact and adds:
  1. Multi-band FFT moire analysis at 128x128.
  2. Per-face rolling moire decision.
  3. Challenge-first policy for suspicious frames.
  4. Calibration logging for real-world threshold tuning.
  5. V4.1 screen-context evidence: flat background + glass glare.
  6. V4.2 phone rectangle evidence: device border/shape around face.
  7. V4.3 enhanced moire pipeline: Y/log/high-pass/Sobel/z-score + track median.

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


# =========================
# V4 TUNING ZONE - CHINH NHUNG THAM SO CHINH
# =========================

V4_COSINE_THRESHOLD = 0.52
# Nguong nhan dien khuon mat.
# Tang: kho nhan nham hon nhung de Unknown hon.
# Giam: de nhan dung hon nhung tang rui ro nhan nham.

MOIRE_CONTEXT_SCALE = 1.65
# Do no vung phan tich quanh mat.
# Tang: lay them vien dien thoai/glare/nen man hinh, tot hon cho bat replay.
# Giam: tap trung vao mat hon, it nhieu nen hon.

MOIRE_EVERY_N_DETECT = 3
# Bao nhieu lan detect mat thi chay moire mot lan.
# Giam ve 1: nhay hon, phan ung nhanh hon, nhung ton CPU hon.
# Tang: nhe may hon nhung de bo lo artifact ngan.

MOIRE_PIPELINE_MODE = "enhanced"
# Che do tien xu ly FFT.
# "legacy": grayscale + FFT nhu V4 cu.
# "enhanced": Y/log/high-pass/Sobel/z-score, nhay hon voi OLED/Retina replay.

MOIRE_SCREEN_THRESHOLD = 0.60
# Neu moire_score < nguong nay thi xem la nghi man hinh va dua vao challenge.
# Tang: nhay hon voi video dien thoai.
# Giam: it challenge nham hon voi nguoi that.

MOIRE_BLOCK_THRESHOLD = 0.45
# Neu moire_score < nguong nay thi block manh o lop Moire.
# Tang: block replay manh hon nhung de false positive hon.
# Giam: bao thu hon, can them context/phone evidence.

MOIRE_SMOOTH_WINDOW = 7
# So frame dung median smoothing tren moi face track.
# Tang: on dinh hon nhung phan ung cham hon.
# Giam: phan ung nhanh hon nhung de jitter.

MOIRE_TRACK_TTL = 12
# So detect cycle giu track khi mat tam thoi.
# Tang: history on dinh hon khi bbox rung/mat mat ngan.
# Giam: it rui ro dung history cu.

MOIRE_IOU_MATCH_THRESHOLD = 0.25
# IoU toi thieu de gan bbox vao track cu.
# Tang: tracking chat hon.
# Giam: de giu track hon khi bbox rung.

MOIRE_HIGH_BAND_WEIGHT = 0.90
# Do nhay voi vung tan so cao, quan trong voi OLED/Retina.
# Tang: de nghi video dien thoai hon.
# Giam: it false positive hon voi webcam/noise.

MOIRE_FRAME_SUSPICIOUS_EVIDENCE = 0.38
# Nguong de mot frame bi xem la SUSPECT.
# Giam: video OLED de bi challenge hon.
# Tang: nguoi that it bi challenge hon.

MOIRE_FRAME_BLOCK_EVIDENCE = 0.68
# Nguong de mot frame bi xem la SCREEN/BLOCK.
# Giam: chan man hinh manh hon nhung de chan nham.
# Tang: it chan nham hon nhung replay de lot hon.

MOIRE_ROLLING_SUSPICIOUS_P10_MAX = 0.38
# Nguong p10 tren nhieu frame de chuyen sang SUSPECT.
# Tang: chi can vai frame hoi xau cung bi challenge.
# Giam: can tin hieu man hinh ro hon moi challenge.

MOIRE_ROLLING_CLEAN_MEAN_MIN = 0.55
# Mean score toi thieu de tin la sach.
# Tang: kho duoc xem la clean, bao mat hon.
# Giam: de pass hon, it phien nguoi that hon.

CHALLENGE_COOLDOWN = 12.0
# Sau khi pass challenge, bo qua challenge trong N giay.
# Giam xuong 0-2 khi test replay de khong bi lan pass truoc che loi moi.
# Tang giup user that do bi hoi challenge lien tuc.

SCREEN_CONTEXT_ENABLED = True
# Bat/tat bo phat hien nen phang va glare quanh mat.
# True: them diem nghi ngo voi video dien thoai/man hinh.
# False: quay lai gan nhu V4 cu chi dung moire/liveness.

SCREEN_CONTEXT_WEIGHT = 0.35
# Muc anh huong cua context vao diem nghi ngo tong.
# Tang: video dien thoai de bi SUSPECT hon.
# Giam: nguoi that it bi challenge nham hon.

FLATNESS_SUSPICIOUS_THRESHOLD = 0.65
# Nguong nen phang de cong nghi ngo.
# Giam: nhay hon voi man hinh zoom full mat.
# Tang: it nghi nham nguoi dung truoc tuong phang.

GLARE_SUSPICIOUS_THRESHOLD = 0.45
# Nguong phan chieu kinh/man hinh de cong nghi ngo.
# Giam: bat glare dien thoai nhay hon.
# Tang: it nghi nham nguoi deo kinh/den manh.

SCREEN_CONTEXT_STRONG_THRESHOLD = 0.78
# Nguong context rat nghi man hinh.
# Chi block khi co them moire/passive suspicious; dung mot minh chi challenge.

PHONE_RECT_ENABLED = True
# Bat/tat lop tim khung dien thoai/man hinh quanh mat.
# True: cong diem nghi ngo khi thay mat nam trong khung chu nhat.
# False: quay lai V4.1, khong dung geometry cua thiet bi.

PHONE_RECT_CONTEXT_SCALE = 2.80
# Do no ROI de tim vien/mep/goc dien thoai quanh mat.
# Tang: de thay vien dien thoai hon.
# Giam: it dinh nham khung cua/poster/nen hon.

PHONE_RECT_VERTICAL_RATIO = 1.6
# Ty le chieu doc / chieu ngang cua ROI phat hien dien thoai.
# Dien thoai dung thuc te khoang 16:9 ~ 1.78, nhung de 1.6 de du margin.
# Tang: ROI cao hon, phu hop dien thoai doc.
# Giam ve 1.0: ROI hinh vuong nhu cu.

PHONE_RECT_SUSPICIOUS_THRESHOLD = 0.38
# Diem bat dau xem khung chu nhat la nghi ngo.
# Giam: nhay hon voi dien thoai OLED.
# Tang: it false positive hon voi background co khung.

PHONE_RECT_STRONG_THRESHOLD = 0.58
# Diem khung chu nhat rat giong dien thoai/man hinh.
# Giam: challenge/block manh hon.
# Tang: bao thu hon, can tin hieu ro hon.

PHONE_RECT_ROLLING_STRONG_COUNT = 2
# Can bao nhieu frame gan day thay phone-rect STRONG de nang thanh BLOCK candidate.
# Giam: bat nhanh hon.
# Tang: on dinh hon, it block nham hon.

# GOI Y CHINH NHANH:
# 1. Video OLED van pass:
#    - Tang MOIRE_SCREEN_THRESHOLD tu 0.60 len 0.65.
#    - Neu can block manh hon, tang MOIRE_BLOCK_THRESHOLD tu 0.45 len 0.50.
#
# 2. Nguoi that bi challenge qua nhieu:
#    - Giam MOIRE_SCREEN_THRESHOLD xuong 0.55.
#    - Giam MOIRE_BLOCK_THRESHOLD xuong 0.38-0.40.
#
# 3. Muon bat vien/glare dien thoai ro hon:
#    - Tang MOIRE_CONTEXT_SCALE tu 1.65 len 2.0 hoac 2.2.
#
# 4. Muon bat vien den/khung dien thoai ro hon:
#    - Tang PHONE_RECT_CONTEXT_SCALE tu 2.45 len 2.8.
#    - Giam PHONE_RECT_SUSPICIOUS_THRESHOLD tu 0.45 xuong 0.38.
#
# 5. Nguoi that bi bat nham do cua/poster/khung nen:
#    - Tang PHONE_RECT_SUSPICIOUS_THRESHOLD len 0.52.
#    - Tang PHONE_RECT_ROLLING_STRONG_COUNT len 4.
#
# 6. Khi test replay lien tuc:
#    - Giam CHALLENGE_COOLDOWN xuong 0 hoac 2 de moi lan deu danh gia lai.
#
# 7. Moi lan chi chinh 1-2 tham so roi test lai.

# ---------------------------------------------------------------------------
# Runtime config noi bo - it khi can chinh
# ---------------------------------------------------------------------------

CAM_INDEX = 0
FRAME_W = 1280
FRAME_H = 720
DETECT_FPS = 10

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
C_GOLD = (30, 200, 255)
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
INFO_PANEL_W = int(380 * UI_SCALE)
LANDMARK_LABELS = ["L-Eye", "R-Eye", "Nose", "L-Mouth", "R-Mouth"]


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


def bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def expanded_roi(frame: np.ndarray, bbox: list[int], scale: float = MOIRE_CONTEXT_SCALE) -> tuple[np.ndarray, list[int]]:
    """Return a context crop around the face for screen/glare cues.

    The original V3 moire crop used the tight face bbox. V4 keeps that UI bbox
    but analyzes a larger crop so phone bezel, screen glare, and surrounding
    flat display regions can influence the FFT/grid features.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(bw, bh) * scale
    ex1 = int(round(cx - side / 2.0))
    ey1 = int(round(cy - side / 2.0))
    ex2 = int(round(cx + side / 2.0))
    ey2 = int(round(cy + side / 2.0))
    ex1, ey1 = max(0, ex1), max(0, ey1)
    ex2, ey2 = min(w, ex2), min(h, ey2)
    if ex2 <= ex1 or ey2 <= ey1:
        return safe_roi(frame, bbox), bbox
    return frame[ey1:ey2, ex1:ex2], [ex1, ey1, ex2, ey2]


@dataclass
class FaceMoireTrack:
    track_id: int
    bbox: list[int]
    ttl: int = MOIRE_TRACK_TTL
    seen: bool = False
    scores: deque = field(default_factory=lambda: deque(maxlen=MOIRE_SMOOTH_WINDOW))
    last_result: dict[str, Any] = field(default_factory=dict)

    def update_score(self, raw_score: float) -> float:
        if self.scores.maxlen != MOIRE_SMOOTH_WINDOW:
            self.scores = deque(self.scores, maxlen=MOIRE_SMOOTH_WINDOW)
        self.scores.append(float(raw_score))
        return float(np.median(np.asarray(self.scores, dtype=np.float32)))


class FaceMoireTrackStore:
    def __init__(self):
        self.tracks: dict[int, FaceMoireTrack] = {}
        self.next_id = 1

    def begin_cycle(self):
        for track in self.tracks.values():
            track.seen = False

    def match_and_update(self, bbox: list[int]) -> FaceMoireTrack:
        best_track = None
        best_iou = 0.0
        for track in self.tracks.values():
            if track.seen:
                continue
            iou = bbox_iou(track.bbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best_track = track

        if best_track is None or best_iou < MOIRE_IOU_MATCH_THRESHOLD:
            best_track = FaceMoireTrack(track_id=self.next_id, bbox=bbox[:])
            self.tracks[best_track.track_id] = best_track
            self.next_id += 1

        best_track.bbox = bbox[:]
        best_track.ttl = MOIRE_TRACK_TTL
        best_track.seen = True
        return best_track

    def finish_cycle(self) -> set[int]:
        expired = []
        for track_id, track in self.tracks.items():
            if not track.seen:
                track.ttl -= 1
            if track.ttl <= 0:
                expired.append(track_id)
        for track_id in expired:
            self.tracks.pop(track_id, None)
        return set(self.tracks.keys())


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
        "high": MOIRE_HIGH_BAND_WEIGHT,
    }

    PEAK_WEAK = 2.25
    PEAK_STRONG = 3.25
    PERIODIC_WEAK = 3.0
    PERIODIC_STRONG = 5.0

    def __init__(self):
        self.last_result: dict[str, Any] = {}

        sz = self.ANALYZE_SIZE
        cy, cx = sz // 2, sz // 2
        y, x = np.ogrid[:sz, :sz]
        self._dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.float32)
        self._angle = np.mod(np.arctan2(y - cy, x - cx), np.pi).astype(np.float32)
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

    def analyze(self, face_roi: np.ndarray, track: FaceMoireTrack | None = None) -> dict[str, Any]:
        if face_roi is None or face_roi.size == 0:
            return self._empty_result(track)

        signal = self._preprocess_for_fft(face_roi)
        log_mag = self._fft_log_magnitude(signal)

        total_energy = float(np.sum(log_mag[self._dc_mask]))
        if total_energy < 1e-6:
            total_energy = 1e-6

        band_scores = {}

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

        h_line = self._check_grid_lines(signal, "horizontal")
        v_line = self._check_grid_lines(signal, "vertical")
        grid_score = float((h_line + v_line) / 2.0)

        high_energy = (
            band_scores["mid_high"]["band_energy"]
            + band_scores["high"]["band_energy"]
        )

        wide_values = log_mag[self._wide_mask]
        peak_ratio = float(np.max(wide_values) / max(np.mean(wide_values), 1e-6))
        energy_ratio = float(high_energy)
        periodicity = float(max(m["periodicity"] for m in band_scores.values()))
        anisotropy = self._check_anisotropy(log_mag)

        f_peak = clamp((peak_ratio - 1.8) / 2.2)
        f_period = clamp((periodicity - 2.2) / 4.0)
        f_aniso = clamp((anisotropy - 1.3) / 1.8)
        f_grid = clamp((grid_score - 0.20) / 0.55)

        screen_evidence = clamp(
            0.50 * f_peak
            + 0.34 * f_period
            + 0.10 * f_aniso
            + 0.06 * f_grid
        )
        raw_score = round(1.0 - screen_evidence, 3)

        smooth_score = float(raw_score)
        samples = 1
        track_id = None
        if track is not None:
            smooth_score = track.update_score(raw_score)
            samples = len(track.scores)
            track_id = track.track_id

        smooth_score = round(float(smooth_score), 3)

        if smooth_score < MOIRE_BLOCK_THRESHOLD:
            decision = "block"
        elif smooth_score < MOIRE_SCREEN_THRESHOLD:
            decision = "suspicious"
        else:
            decision = "clean"

        strong_signals: list[str] = []
        if f_peak >= 0.55:
            strong_signals.append("peak_prominence")
        if f_period >= 0.55:
            strong_signals.append("periodicity")
        if f_aniso >= 0.55:
            strong_signals.append("anisotropy")
        if f_grid >= 0.45:
            strong_signals.append("grid_projection")
        if high_energy > 0.52:
            strong_signals.append("high_frequency_energy")

        result = {
            "pipeline_mode": MOIRE_PIPELINE_MODE,
            "track_id": track_id,
            "samples": samples,
            "moire_score": smooth_score,
            "smooth_score": smooth_score,
            "raw_score": raw_score,
            "is_screen": decision == "block",
            "decision_hint": decision,
            "screen_evidence": round(screen_evidence, 3),
            "strong_signals": strong_signals[:8],
            "band_scores": band_scores,
            "feature_scores": {
                "peak": round(f_peak, 3),
                "periodicity": round(f_period, 3),
                "anisotropy": round(f_aniso, 3),
                "grid": round(f_grid, 3),
            },
            "peak_ratio": round(peak_ratio, 2),
            "energy_ratio": round(energy_ratio, 3),
            "periodicity": round(periodicity, 2),
            "anisotropy": round(anisotropy, 3),
            "grid_score": round(grid_score, 3),
        }
        if track is not None:
            track.last_result = result
        self.last_result = result
        return result

    def _empty_result(self, track: FaceMoireTrack | None = None) -> dict[str, Any]:
        return {
            "pipeline_mode": MOIRE_PIPELINE_MODE,
            "track_id": track.track_id if track is not None else None,
            "samples": len(track.scores) if track is not None else 0,
            "moire_score": 0.5,
            "smooth_score": 0.5,
            "raw_score": 0.5,
            "is_screen": None,
            "decision_hint": "clean",
            "screen_evidence": 0.0,
            "strong_signals": [],
            "band_scores": {},
            "feature_scores": {},
            "peak_ratio": 0.0,
            "energy_ratio": 0.0,
            "periodicity": 0.0,
            "anisotropy": 0.0,
            "grid_score": 0.0,
        }

    def _preprocess_for_fft(self, face_roi: np.ndarray) -> np.ndarray:
        if MOIRE_PIPELINE_MODE == "legacy":
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (self.ANALYZE_SIZE, self.ANALYZE_SIZE))
            return gray.astype(np.float32) / 255.0

        y = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        y = cv2.resize(y, (self.ANALYZE_SIZE, self.ANALYZE_SIZE)).astype(np.float32)
        y = np.log1p(y)
        blur = cv2.GaussianBlur(y, (0, 0), sigmaX=3.0, sigmaY=3.0)
        high = y - blur
        sx = cv2.Sobel(high, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(high, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(sx, sy)
        return ((grad - float(np.mean(grad))) / (float(np.std(grad)) + 1e-6)).astype(np.float32)

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

    def _check_anisotropy(self, log_mag: np.ndarray) -> float:
        mask = self._wide_mask
        angles = self._angle[mask]
        values = log_mag[mask]
        if values.size < 16:
            return 1.0

        bins = np.linspace(0.0, np.pi, 25, dtype=np.float32)
        energies = []
        for start, end in zip(bins[:-1], bins[1:]):
            sector = values[(angles >= start) & (angles < end)]
            if sector.size:
                energies.append(float(np.sum(sector)))
        if not energies:
            return 1.0

        arr = np.asarray(energies, dtype=np.float32)
        return float(np.max(arr) / max(float(np.mean(arr)), 1e-6))

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


class ScreenContextDetectorV41:
    """Detect flat screen-like context and glass glare around a face.

    This detector is intentionally conservative: it produces suspicion
    evidence, not an independent spoof classifier.
    """

    def analyze(self, frame: np.ndarray, face_bbox: list[int], roi_bbox: list[int]) -> dict[str, Any]:
        if not SCREEN_CONTEXT_ENABLED:
            return self._empty("disabled")

        roi = safe_roi(frame, roi_bbox)
        if roi is None or roi.size == 0:
            return self._empty("empty_roi")

        local_face_bbox = self._local_bbox(face_bbox, roi_bbox)
        context_mask = self._context_mask(roi, local_face_bbox)
        boundary_mask = self._boundary_mask(roi, local_face_bbox)

        flatness = self._flatness_metrics(roi, context_mask, boundary_mask)
        glare = self._glare_metrics(roi, context_mask)

        flat_score = float(flatness["score"])
        glare_score = float(glare["score"])
        score = clamp((flat_score * 0.60 + glare_score * 0.40) * SCREEN_CONTEXT_WEIGHT / 0.35)

        signals = []
        strong_signals = []
        if flat_score >= FLATNESS_SUSPICIOUS_THRESHOLD:
            signals.append("flat_background")
        if flat_score >= 0.80:
            strong_signals.append("strong_flat_background")
        if glare_score >= GLARE_SUSPICIOUS_THRESHOLD:
            signals.append("specular_glare")
        if glare_score >= 0.70:
            strong_signals.append("strong_specular_glare")

        if score >= SCREEN_CONTEXT_STRONG_THRESHOLD and strong_signals:
            decision = "strong"
        elif score >= 0.45 or signals:
            decision = "suspicious"
        else:
            decision = "clean"

        return {
            "score": round(score, 3),
            "decision": decision,
            "signals": signals + strong_signals,
            "flatness": flatness,
            "glare": glare,
        }

    def _empty(self, reason: str) -> dict[str, Any]:
        return {
            "score": 0.0,
            "decision": "clean",
            "signals": [reason] if reason != "disabled" else [],
            "flatness": {},
            "glare": {},
        }

    def _local_bbox(self, face_bbox: list[int], roi_bbox: list[int]) -> list[int]:
        fx1, fy1, fx2, fy2 = face_bbox
        rx1, ry1, _, _ = roi_bbox
        return [fx1 - rx1, fy1 - ry1, fx2 - rx1, fy2 - ry1]

    def _context_mask(self, roi: np.ndarray, local_face_bbox: list[int]) -> np.ndarray:
        h, w = roi.shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)
        x1, y1, x2, y2 = self._clip_bbox(local_face_bbox, w, h)
        pad_x = int((x2 - x1) * 0.08)
        pad_y = int((y2 - y1) * 0.08)
        cv2.rectangle(
            mask,
            (max(0, x1 - pad_x), max(0, y1 - pad_y)),
            (min(w - 1, x2 + pad_x), min(h - 1, y2 + pad_y)),
            0,
            -1,
        )
        return mask

    def _boundary_mask(self, roi: np.ndarray, local_face_bbox: list[int]) -> np.ndarray:
        h, w = roi.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = self._clip_bbox(local_face_bbox, w, h)
        pad = max(4, int(min(x2 - x1, y2 - y1) * 0.08))
        cv2.rectangle(
            mask,
            (max(0, x1 - pad), max(0, y1 - pad)),
            (min(w - 1, x2 + pad), min(h - 1, y2 + pad)),
            255,
            -1,
        )
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
        return mask

    def _flatness_metrics(self, roi: np.ndarray, context_mask: np.ndarray, boundary_mask: np.ndarray) -> dict[str, Any]:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        context_values = gray[context_mask > 0]
        if context_values.size < 64:
            return {
                "score": 0.0,
                "laplacian_var": 0.0,
                "gradient_entropy": 0.0,
                "color_std": 0.0,
                "edge_density": 0.0,
                "boundary_edge_strength": 0.0,
            }

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_context = lap[context_mask > 0]
        lap_var = float(np.var(lap_context)) if lap_context.size else 0.0

        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(sobel_x, sobel_y)
        grad_context = grad[context_mask > 0]
        grad_entropy = self._entropy(grad_context, bins=32, value_range=(0, 255))

        color_pixels = roi[context_mask > 0]
        color_std = float(np.mean(np.std(color_pixels.astype(np.float32), axis=0))) if color_pixels.size else 0.0

        edges = cv2.Canny(gray, 60, 160)
        edge_density = float(np.mean(edges[context_mask > 0] > 0)) if context_values.size else 0.0
        boundary_values = grad[boundary_mask > 0]
        boundary_edge_strength = float(np.mean(boundary_values)) if boundary_values.size else 0.0

        flat_lap = 1.0 - clamp(lap_var / 220.0)
        flat_entropy = 1.0 - clamp(grad_entropy / 3.2)
        flat_color = 1.0 - clamp(color_std / 38.0)
        low_edge = 1.0 - clamp(edge_density / 0.12)
        sharp_boundary = clamp((boundary_edge_strength - 18.0) / 45.0)

        score = clamp(
            flat_lap * 0.30
            + flat_entropy * 0.22
            + flat_color * 0.20
            + low_edge * 0.13
            + sharp_boundary * 0.15
        )

        return {
            "score": round(score, 3),
            "laplacian_var": round(lap_var, 2),
            "gradient_entropy": round(float(grad_entropy), 3),
            "color_std": round(color_std, 2),
            "edge_density": round(edge_density, 3),
            "boundary_edge_strength": round(boundary_edge_strength, 2),
        }

    def _glare_metrics(self, roi: np.ndarray, context_mask: np.ndarray) -> dict[str, Any]:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        bright = (v > 235).astype(np.uint8) * 255
        white_specular = ((v > 220) & (s < 70)).astype(np.uint8) * 255
        mask = cv2.bitwise_or(bright, white_specular)
        mask = cv2.bitwise_and(mask, context_mask)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        context_area = max(1, int(np.sum(context_mask > 0)))
        bright_ratio = float(np.sum(mask > 0) / context_area)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blob_count = 0
        max_area_ratio = 0.0
        linear_score = 0.0
        edge_scores = []
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        grad = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 8:
                continue
            blob_count += 1
            max_area_ratio = max(max_area_ratio, area / context_area)
            x, y, w, h = cv2.boundingRect(contour)
            aspect = max(w, h) / max(1, min(w, h))
            linear_score = max(linear_score, clamp((aspect - 2.0) / 5.0))
            blob_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(blob_mask, [contour], -1, 255, 1)
            values = grad[blob_mask > 0]
            if values.size:
                edge_scores.append(float(np.mean(values)))

        highlight_edge_sharpness = float(np.mean(edge_scores)) if edge_scores else 0.0
        glare_score = clamp(
            clamp(bright_ratio / 0.025) * 0.35
            + clamp(max_area_ratio / 0.018) * 0.25
            + clamp(blob_count / 4.0) * 0.15
            + linear_score * 0.15
            + clamp((highlight_edge_sharpness - 20.0) / 60.0) * 0.10
        )

        return {
            "score": round(glare_score, 3),
            "bright_pixel_ratio": round(bright_ratio, 4),
            "highlight_blob_count": blob_count,
            "max_highlight_area_ratio": round(max_area_ratio, 4),
            "highlight_edge_sharpness": round(highlight_edge_sharpness, 2),
            "linear_glare_score": round(linear_score, 3),
        }

    def _clip_bbox(self, bbox: list[int], w: int, h: int) -> list[int]:
        x1, y1, x2, y2 = bbox
        return [
            max(0, min(w - 1, int(x1))),
            max(0, min(h - 1, int(y1))),
            max(0, min(w - 1, int(x2))),
            max(0, min(h - 1, int(y2))),
        ]

    def _entropy(self, values: np.ndarray, *, bins: int, value_range: tuple[int, int]) -> float:
        if values.size == 0:
            return 0.0
        hist, _ = np.histogram(values, bins=bins, range=value_range)
        total = float(np.sum(hist))
        if total <= 0:
            return 0.0
        p = hist.astype(np.float32) / total
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))


class PhoneRectangleDetectorV42:
    """Detect phone/screen rectangle evidence around a detected face.

    This is a geometry/device evidence layer. It should normally trigger
    challenge or combine with other suspicious signals, not classify spoof
    alone from a single frame.
    """

    def analyze(self, frame: np.ndarray, face_bbox: list[int]) -> dict[str, Any]:
        if not PHONE_RECT_ENABLED:
            return self._empty("disabled")

        roi, roi_bbox = self._portrait_roi(frame, face_bbox)
        if roi is None or roi.size == 0:
            return self._empty("empty_roi", roi_bbox)

        local_face = self._local_bbox(face_bbox, roi_bbox)
        edges = self._edges(roi)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        candidate_count = 0
        for contour in contours:
            candidate = self._score_candidate(roi, edges, contour, local_face, roi_bbox)
            if not candidate:
                continue
            candidate_count += 1
            if best is None or candidate["score"] > best["score"]:
                best = candidate

        if best is None:
            result = self._empty("no_rectangle", roi_bbox)
            result["candidate_count"] = 0
            return result

        best["candidate_count"] = candidate_count
        score = float(best["score"])
        signals = []
        if best["face_inside_score"] >= 0.70:
            signals.append("face_inside_rect")
        if best["black_border_score"] >= 0.45:
            signals.append("dark_border")
        if best["border_edge_score"] >= 0.45:
            signals.append("sharp_rect_edge")
        if best["corner_score"] >= 0.70:
            signals.append("rect_corners")

        if score >= PHONE_RECT_STRONG_THRESHOLD:
            decision = "strong"
        elif score >= PHONE_RECT_SUSPICIOUS_THRESHOLD:
            decision = "suspicious"
        else:
            decision = "clean"

        return {
            "score": round(score, 3),
            "decision": decision,
            "signals": signals,
            "candidate_count": int(best["candidate_count"]),
            "roi_bbox": roi_bbox,
            "best_rect": best["best_rect"],
            "best_rect_points": best["best_rect_points"],
            "rectangularity": round(best["rectangularity_score"], 3),
            "aspect_ratio": round(best["aspect_ratio"], 3),
            "aspect_score": round(best["aspect_score"], 3),
            "face_inside_score": round(best["face_inside_score"], 3),
            "rect_area_ratio": round(best["rect_area_ratio"], 3),
            "rect_area_score": round(best["rect_area_score"], 3),
            "border_edge_score": round(best["border_edge_score"], 3),
            "black_border_score": round(best["black_border_score"], 3),
            "corner_score": round(best["corner_score"], 3),
            "margin_score": round(best["margin_score"], 3),
        }

    def _empty(self, reason: str, roi_bbox: list[int] | None = None) -> dict[str, Any]:
        return {
            "score": 0.0,
            "decision": "clean",
            "signals": [reason] if reason != "disabled" else [],
            "candidate_count": 0,
            "roi_bbox": roi_bbox or [],
            "best_rect": [],
            "best_rect_points": [],
            "rectangularity": 0.0,
            "aspect_ratio": 0.0,
            "aspect_score": 0.0,
            "face_inside_score": 0.0,
            "rect_area_ratio": 0.0,
            "rect_area_score": 0.0,
            "border_edge_score": 0.0,
            "black_border_score": 0.0,
            "corner_score": 0.0,
            "margin_score": 0.0,
        }

    def _edges(self, roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 45, 135)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    def _score_candidate(
        self,
        roi: np.ndarray,
        edges: np.ndarray,
        contour: np.ndarray,
        local_face: list[int],
        roi_bbox: list[int],
    ) -> dict[str, Any] | None:
        roi_h, roi_w = roi.shape[:2]
        roi_area = max(1.0, float(roi_h * roi_w))
        contour_area = float(cv2.contourArea(contour))
        if contour_area < roi_area * 0.025:
            return None

        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), _ = rect
        rw, rh = float(rw), float(rh)
        if rw < 12 or rh < 12:
            return None

        rect_area = max(1.0, rw * rh)
        rect_area_ratio = rect_area / roi_area
        if rect_area_ratio < 0.08 or rect_area_ratio > 0.98:
            return None

        box = cv2.boxPoints(rect).astype(np.int32)
        box_float = box.astype(np.float32)
        x, y, w, h = cv2.boundingRect(box)
        if w < 12 or h < 12:
            return None

        rectangularity = clamp(contour_area / rect_area)
        rectangularity_score = clamp((rectangularity - 0.45) / 0.35)

        long_side = max(rw, rh)
        short_side = max(1.0, min(rw, rh))
        aspect_ratio = long_side / short_side
        aspect_score = clamp(1.0 - abs(aspect_ratio - 1.75) / 1.05)

        face_inside_score = self._face_inside_score(box_float, local_face)
        if face_inside_score < 0.35:
            return None

        rect_area_score = clamp(1.0 - abs(rect_area_ratio - 0.42) / 0.38)
        border_edge_score, black_border_score = self._border_scores(roi, edges, box)
        corner_score = self._corner_score(contour)
        margin_score = self._margin_score([x, y, x + w, y + h], local_face)

        score = clamp(
            rectangularity_score * 0.22
            + face_inside_score * 0.20
            + aspect_score * 0.12
            + rect_area_score * 0.10
            + border_edge_score * 0.14
            + black_border_score * 0.16
            + corner_score * 0.08
            + margin_score * 0.08
        )

        rx1, ry1, _, _ = roi_bbox
        abs_box = [[int(px + rx1), int(py + ry1)] for px, py in box.tolist()]
        abs_rect = [int(x + rx1), int(y + ry1), int(x + w + rx1), int(y + h + ry1)]

        return {
            "score": score,
            "candidate_count": 1,
            "best_rect": abs_rect,
            "best_rect_points": abs_box,
            "rectangularity_score": rectangularity_score,
            "aspect_ratio": aspect_ratio,
            "aspect_score": aspect_score,
            "face_inside_score": face_inside_score,
            "rect_area_ratio": rect_area_ratio,
            "rect_area_score": rect_area_score,
            "border_edge_score": border_edge_score,
            "black_border_score": black_border_score,
            "corner_score": corner_score,
            "margin_score": margin_score,
        }

    def _border_scores(self, roi: np.ndarray, edges: np.ndarray, box: np.ndarray) -> tuple[float, float]:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        fill_mask = np.zeros(gray.shape, dtype=np.uint8)
        border_mask = np.zeros(gray.shape, dtype=np.uint8)
        min_side = max(1, int(min(cv2.boundingRect(box)[2], cv2.boundingRect(box)[3])))
        thickness = max(3, int(min_side * 0.035))
        cv2.drawContours(fill_mask, [box], -1, 255, -1)
        cv2.drawContours(border_mask, [box], -1, 255, thickness)

        border_pixels = int(np.sum(border_mask > 0))
        if border_pixels <= 0:
            return 0.0, 0.0

        border_edge_density = float(np.mean(edges[border_mask > 0] > 0))
        border_edge_score = clamp((border_edge_density - 0.06) / 0.24)

        border_values = gray[border_mask > 0]
        dark_ratio = float(np.mean(border_values < 55)) if border_values.size else 0.0

        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness * 2 + 1, thickness * 2 + 1))
        inner_mask = cv2.erode(fill_mask, erode_kernel, iterations=1)
        inner_values = gray[inner_mask > 0]
        border_mean = float(np.mean(border_values)) if border_values.size else 0.0
        inner_mean = float(np.mean(inner_values)) if inner_values.size else border_mean
        dark_contrast = clamp((inner_mean - border_mean) / 80.0)
        black_border_score = clamp(clamp((dark_ratio - 0.10) / 0.40) * 0.65 + dark_contrast * 0.35)

        return border_edge_score, black_border_score

    def _corner_score(self, contour: np.ndarray) -> float:
        peri = float(cv2.arcLength(contour, True))
        if peri <= 0:
            return 0.0
        approx = cv2.approxPolyDP(contour, 0.035 * peri, True)
        n = len(approx)
        if n <= 0:
            return 0.0
        return clamp(1.0 - abs(n - 4) / 5.0)

    def _face_inside_score(self, box: np.ndarray, local_face: list[int]) -> float:
        fx1, fy1, fx2, fy2 = local_face
        points = [
            ((fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0),
            (fx1, fy1),
            (fx2, fy1),
            (fx2, fy2),
            (fx1, fy2),
        ]
        inside = 0
        for point in points:
            if cv2.pointPolygonTest(box, point, False) >= 0:
                inside += 1
        return inside / len(points)

    def _margin_score(self, rect_bbox: list[int], local_face: list[int]) -> float:
        rx1, ry1, rx2, ry2 = rect_bbox
        fx1, fy1, fx2, fy2 = local_face
        rw = max(1, rx2 - rx1)
        rh = max(1, ry2 - ry1)
        margins = [fx1 - rx1, fy1 - ry1, rx2 - fx2, ry2 - fy2]
        if min(margins) < 0:
            return 0.0
        min_margin = min(margins)
        balance = min(margins[0], margins[2]) / max(1, max(margins[0], margins[2]))
        balance = min(balance, min(margins[1], margins[3]) / max(1, max(margins[1], margins[3])))
        margin_room = clamp(min_margin / (0.07 * min(rw, rh)))
        return clamp(margin_room * 0.65 + balance * 0.35)

    def _local_bbox(self, face_bbox: list[int], roi_bbox: list[int]) -> list[int]:
        fx1, fy1, fx2, fy2 = face_bbox
        rx1, ry1, _, _ = roi_bbox
        return [fx1 - rx1, fy1 - ry1, fx2 - rx1, fy2 - ry1]

    @staticmethod
    def _portrait_roi(frame: np.ndarray, bbox: list[int]) -> tuple[np.ndarray, list[int]]:
        """ROI hinh chu nhat dung (portrait) phu hop dien thoai doc."""
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        base = max(bw, bh) * PHONE_RECT_CONTEXT_SCALE
        half_w = base / 2.0
        half_h = (base * PHONE_RECT_VERTICAL_RATIO) / 2.0
        ex1 = int(round(cx - half_w))
        ey1 = int(round(cy - half_h))
        ex2 = int(round(cx + half_w))
        ey2 = int(round(cy + half_h))
        ex1, ey1 = max(0, ex1), max(0, ey1)
        ex2, ey2 = min(fw, ex2), min(fh, ey2)
        if ex2 <= ex1 or ey2 <= ey1:
            return safe_roi(frame, bbox), bbox
        return frame[ey1:ey2, ex1:ex2], [ex1, ey1, ex2, ey2]


@dataclass
class RollingPhoneRectDecision:
    maxlen: int = 8
    samples: deque = field(default_factory=lambda: deque(maxlen=8))

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
                "mean_score": 0.0,
                "max_score": 0.0,
                "suspicious_count": 0,
                "strong_count": 0,
                "clean_count": 0,
            }

        scores = np.asarray([float(s.get("score", 0.0)) for s in self.samples], dtype=np.float32)
        hints = [s.get("decision", "clean") for s in self.samples]
        suspicious_count = sum(1 for h in hints if h in {"suspicious", "strong"})
        strong_count = sum(1 for h in hints if h == "strong")
        clean_count = sum(1 for h in hints if h == "clean")
        samples = len(self.samples)
        mean_score = float(np.mean(scores))
        max_score = float(np.max(scores))

        if samples >= PHONE_RECT_ROLLING_STRONG_COUNT and strong_count >= PHONE_RECT_ROLLING_STRONG_COUNT:
            decision = "block"
        elif samples >= 2 and (strong_count >= 1 or suspicious_count >= 2 or mean_score >= PHONE_RECT_SUSPICIOUS_THRESHOLD):
            decision = "suspicious"
        elif samples >= 3 and clean_count >= 2:
            decision = "clean"
        else:
            decision = "checking"

        return {
            "decision": decision,
            "samples": samples,
            "mean_score": round(mean_score, 3),
            "max_score": round(max_score, 3),
            "suspicious_count": suspicious_count,
            "strong_count": strong_count,
            "clean_count": clean_count,
        }


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

        scores = np.asarray([float(s.get("moire_score", s.get("raw_score", 0.5))) for s in self.samples], dtype=np.float32)
        hints = [s.get("decision_hint", "clean") for s in self.samples]
        suspicious_count = sum(1 for h in hints if h in {"suspicious", "block"})
        block_count = sum(1 for h in hints if h == "block")
        clean_count = sum(1 for h in hints if h == "clean")
        samples = len(self.samples)

        mean_score = float(np.mean(scores))
        min_score = float(np.min(scores))
        p10_score = float(np.percentile(scores, 10))

        decision = "checking"
        if block_count >= 1 or min_score < MOIRE_BLOCK_THRESHOLD:
            decision = "block"
        elif suspicious_count >= 1 or mean_score < MOIRE_SCREEN_THRESHOLD or p10_score < MOIRE_SCREEN_THRESHOLD:
            decision = "suspicious"
        elif samples >= 4 and clean_count >= 3 and mean_score > MOIRE_ROLLING_CLEAN_MEAN_MIN:
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
            "moire_context_scale": MOIRE_CONTEXT_SCALE,
            "moire_roi_bbox": payload.get("moire_roi_bbox", []),
            "moire": payload.get("moire", {}),
            "rolling": payload.get("rolling", {}),
            "screen_context": payload.get("screen_context", {}),
            "phone_rect": payload.get("phone_rect", {}),
            "phone_rect_rolling": payload.get("phone_rect_rolling", {}),
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
    """V3-style rounded rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    radius = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)
    if radius < 1:
        cv2.rectangle(img, pt1, pt2, color, thickness, cv2.LINE_AA)
        return
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)


def draw_label_box(img, title, line_1, line_2, x1, y1, color):
    """V3-style filled label above the bounding box."""
    line_count = 1 + (1 if line_1 else 0) + (1 if line_2 else 0)
    label_h = S(22) * line_count + S(8)
    lx1, ly1 = x1, max(0, y1 - label_h)
    lx2, ly2 = x1 + max(S(240), len(title) * S(13) + S(24)), y1
    overlay = img.copy()
    cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), color, -1)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
    y_off = ly1 + S(18)
    cv2.putText(img, title[:28], (lx1 + S(8), y_off), FONT, FS(0.52), C_WHITE, 1, cv2.LINE_AA)
    if line_1:
        y_off += S(20)
        cv2.putText(img, line_1[:38], (lx1 + S(8), y_off), FONT_SMALL, FS(0.48), C_WHITE, 1, cv2.LINE_AA)
    if line_2:
        y_off += S(20)
        cv2.putText(img, line_2[:46], (lx1 + S(8), y_off), FONT_SMALL, FS(0.48), C_GOLD, 1, cv2.LINE_AA)


def draw_landmarks(img, landmarks, color=C_CYAN):
    """V3-style 5-point landmark overlay."""
    if landmarks is None or len(landmarks) < 5:
        return
    for i, (x, y) in enumerate(landmarks[:5]):
        px, py = int(x), int(y)
        cv2.circle(img, (px, py), S(5), color, 2, cv2.LINE_AA)
        cv2.circle(img, (px, py), S(2), C_WHITE, -1, cv2.LINE_AA)
        label = LANDMARK_LABELS[i] if i < len(LANDMARK_LABELS) else str(i)
        cv2.putText(img, label, (px + S(7), py - S(4)), FONT_SMALL, FS(0.42), color, 1, cv2.LINE_AA)


def conf_bar(img, x1, y2, x2, confidence, color):
    """V3-style confidence bar below bbox."""
    bar_w = x2 - x1
    bar_h = S(6)
    filled = int(bar_w * clamp(confidence))
    cv2.rectangle(img, (x1, y2 + S(4)), (x2, y2 + S(4) + bar_h), C_BLACK, -1)
    cv2.rectangle(img, (x1, y2 + S(4)), (x1 + filled, y2 + S(4) + bar_h), color, -1)


def draw_moire_badge(img, x2, y2, moire, rolling):
    """V3-style compact moire badge, extended with V4 SUSPECT state."""
    if not moire:
        return
    decision = rolling.get("decision") or moire.get("decision_hint", "clean")
    score = float(moire.get("moire_score", moire.get("raw_score", 1.0)))
    if decision == "block":
        txt, color = "SCREEN", C_SPOOF
    elif decision == "suspicious":
        txt, color = "SUSPECT", C_ORANGE
    elif decision == "checking":
        txt, color = "?", C_GOLD
    else:
        txt, color = "REAL", C_PRESENT

    bx, by = x2 - S(92), y2 + S(14)
    cv2.rectangle(img, (bx, by), (bx + S(90), by + S(18)), color, -1)
    cv2.putText(img, f"{txt} {score:.0%}", (bx + S(3), by + S(13)), FONT, FS(0.35), C_WHITE, 1, cv2.LINE_AA)


def draw_moire_spectrum(img, face_roi, x1, y1):
    """V3-style FFT thumbnail near bbox."""
    if face_roi is None or face_roi.size == 0:
        return
    try:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0
        window = np.outer(np.hanning(64), np.hanning(64))
        fft = np.fft.fft2(gray * window)
        mag = np.log1p(np.abs(np.fft.fftshift(fft)))
        mag = (mag / mag.max() * 255).astype(np.uint8) if mag.max() > 0 else np.zeros((64, 64), np.uint8)
        mag = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
        sx, sy = max(0, x1 - S(70)), max(0, y1)
        if sy + 64 < img.shape[0] and sx + 64 < img.shape[1]:
            img[sy:sy + 64, sx:sx + 64] = mag
            cv2.rectangle(img, (sx, sy), (sx + 64, sy + 64), C_PURPLE, 1)
            cv2.putText(img, "FFT", (sx + 2, sy + 12), FONT, 0.35, C_WHITE, 1, cv2.LINE_AA)
    except Exception:
        return


def draw_phone_rectangle_overlay(img, phone_rect, phone_rect_rolling):
    if not phone_rect or not phone_rect.get("best_rect_points"):
        return
    decision = phone_rect_rolling.get("decision", phone_rect.get("decision", "clean"))
    if decision == "clean" and phone_rect.get("score", 0) < PHONE_RECT_SUSPICIOUS_THRESHOLD:
        return

    color = C_SPOOF if decision == "block" else (C_ORANGE if decision in {"suspicious", "strong"} else C_GOLD)
    points = np.asarray(phone_rect["best_rect_points"], dtype=np.int32)
    cv2.polylines(img, [points], True, color, S(2), cv2.LINE_AA)
    x, y, _, _ = phone_rect.get("best_rect", [0, 0, 0, 0])
    label = f"PHONE {phone_rect.get('score', 0):.0%}"
    cv2.rectangle(img, (x, max(0, y - S(20))), (x + S(104), y), color, -1)
    cv2.putText(img, label, (x + S(4), max(S(14), y - S(5))), FONT, FS(0.35), C_WHITE, 1, cv2.LINE_AA)


def draw_challenge_overlay(img, challenge: ChallengeControllerV4):
    """V3-style top challenge banner, backed by V4 challenge state."""
    if not challenge.active and not challenge.message:
        return

    h, w = img.shape[:2]
    if challenge.active:
        overlay = img.copy()
        alpha = float(0.55 + 0.15 * np.sin(time.time() * 6))
        cv2.rectangle(overlay, (0, S(44)), (w, S(110)), (20, 20, 60), -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        text = f">> {challenge.text} <<"
        text_size = cv2.getTextSize(text, FONT, FS(0.85), 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(img, text, (text_x, S(72)), FONT, FS(0.85), C_ORANGE, 2, cv2.LINE_AA)
        cv2.putText(img, f"[{challenge.remaining():.1f}s]", (text_x, S(100)), FONT, FS(0.55), C_GOLD, 1, cv2.LINE_AA)

        progress = min(1.0, (time.time() - challenge.started_at) / CHALLENGE_TIMEOUT)
        bar_w_px = w - S(40)
        cv2.rectangle(img, (S(20), S(108)), (S(20) + bar_w_px, S(112)), C_DIM, -1)
        cv2.rectangle(img, (S(20), S(108)), (S(20) + int(bar_w_px * progress), S(112)), C_ORANGE, -1)
    elif "passed" in challenge.message.lower():
        cv2.putText(img, "VERIFIED (challenge passed)", (S(20), S(66)), FONT, FS(0.50), C_PRESENT, 1, cv2.LINE_AA)
    else:
        cv2.putText(img, challenge.message[:64], (S(20), S(66)), FONT, FS(0.50), C_SPOOF, 1, cv2.LINE_AA)


def draw_hud(
    img,
    fps,
    face_count,
    total_students,
    show_landmarks,
    show_info,
    show_moire,
    logging_on=False,
    debug_enabled=False,
):
    """V3-style top HUD with small V4 additions."""
    h, w = img.shape[:2]
    hud_h = S(44)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), C_BLACK, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    items = [
        (f"FPS {fps:5.1f}", C_GOLD),
        (f"Faces: {face_count}", C_PRESENT if face_count > 0 else C_DIM),
        (f"DB: {total_students}", C_WHITE),
        (f"[L]mk:{'ON' if show_landmarks else 'OFF'}", C_CYAN if show_landmarks else C_DIM),
        (f"[I]nfo:{'ON' if show_info else 'OFF'}", C_BLUE if show_info else C_DIM),
        (f"[M]oire:{'ON' if show_moire else 'OFF'}", C_PURPLE if show_moire else C_DIM),
        (f"[D]bg:{'ON' if debug_enabled else 'OFF'}", C_ORANGE if debug_enabled else C_DIM),
        (f"[C]log:{'ON' if logging_on else 'OFF'}", C_PRESENT if logging_on else C_DIM),
        ("Q=Quit  S=Shot  R=Reload", C_DIM),
    ]
    x = S(14)
    for txt, color in items:
        cv2.putText(img, txt, (x, S(29)), FONT_SMALL, FS(0.48), color, 1, cv2.LINE_AA)
        x += len(txt) * S(8) + S(16)


def draw_info_panel(frame, results, challenge, debug_enabled, panel_w=INFO_PANEL_W):
    """V3-style right panel; V4 only adds moire rolling/debug fields."""
    h, w = frame.shape[:2]
    panel = np.full((h, panel_w, 3), C_PANEL_BG, dtype=np.uint8)

    cv2.putText(panel, "FACE RECOGNITION V4.3", (S(14), S(28)), FONT, FS(0.55), C_GOLD, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Thr: {V4_COSINE_THRESHOLD:.0%}", (panel_w - S(90), S(28)), FONT_SMALL, FS(0.45), C_DIM, 1, cv2.LINE_AA)
    cv2.line(panel, (S(14), S(38)), (panel_w - S(14), S(38)), C_DIM, 1, cv2.LINE_AA)

    y = S(58)
    if challenge.active or challenge.message:
        cv2.putText(panel, "ANTI-SPOOF V4.3", (S(14), y), FONT, FS(0.45), C_ORANGE, 1, cv2.LINE_AA)
        y += S(22)
        status_txt = challenge.message or challenge.text
        s_color = C_ORANGE if challenge.active else (C_PRESENT if "passed" in status_txt.lower() else C_SPOOF)
        max_chars = max(20, panel_w // S(9))
        while status_txt:
            chunk = status_txt[:max_chars]
            status_txt = status_txt[max_chars:]
            cv2.putText(panel, chunk, (S(14), y), FONT_SMALL, FS(0.45), s_color, 1, cv2.LINE_AA)
            y += S(18)
        y += S(6)
        cv2.line(panel, (S(14), y), (panel_w - S(14), y), (60, 60, 65), 1)
        y += S(14)

    if not results:
        cv2.putText(panel, "No faces detected", (S(14), y), FONT, FS(0.45), C_DIM, 1, cv2.LINE_AA)
        return np.hstack([frame, panel])

    for idx, r in enumerate(results[:4]):
        if y > h - S(40):
            break

        color = C_PRESENT if r["status"] == "present" else (C_SPOOF if r["status"] == "spoof" else C_UNKNOWN)
        cv2.putText(panel, f"--- Face #{idx + 1} ---", (S(14), y), FONT, FS(0.45), color, 1, cv2.LINE_AA)
        y += S(24)

        if r.get("aligned") is not None:
            thumb = cv2.resize(r["aligned"], (64, 64))
            cv2.rectangle(panel, (S(14), y), (S(80), y + S(66)), color, 2)
            panel[y + 1:y + 65, S(15):S(79)] = thumb
            tx = S(90)
            cv2.putText(panel, r.get("name", "Unknown")[:24], (tx, y + S(18)), FONT, FS(0.48), C_WHITE, 1, cv2.LINE_AA)
            cv2.putText(panel, r["status"].upper(), (tx, y + S(38)), FONT_SMALL, FS(0.45), color, 1, cv2.LINE_AA)
            if r.get("sid"):
                cv2.putText(panel, f"ID: {r['sid']}", (tx, y + S(56)), FONT_SMALL, FS(0.45), C_DIM, 1, cv2.LINE_AA)
            y += S(72)
        else:
            cv2.putText(panel, f"Name: {r.get('name', 'Unknown')}", (S(14), y), FONT_SMALL, FS(0.45), C_WHITE, 1, cv2.LINE_AA)
            y += S(20)

        conf_value = r.get("conf", r.get("confidence", 0.0))
        conf_pct = conf_value * 100
        conf_color = C_PRESENT if conf_pct >= 50 else C_UNKNOWN
        cv2.putText(panel, f"Confidence: {conf_pct:.1f}%", (S(14), y), FONT_SMALL, FS(0.45), conf_color, 1, cv2.LINE_AA)
        bar_x = S(180)
        bar_w = panel_w - bar_x - S(16)
        filled = int(bar_w * clamp(conf_value))
        cv2.rectangle(panel, (bar_x, y - S(10)), (bar_x + bar_w, y), C_BLACK, -1)
        cv2.rectangle(panel, (bar_x, y - S(10)), (bar_x + filled, y), conf_color, -1)
        y += S(22)

        if r.get("det_conf") is not None:
            cv2.putText(panel, f"Det Score: {r['det_conf']:.2f}", (S(14), y), FONT_SMALL, FS(0.45), C_DIM, 1, cv2.LINE_AA)
            y += S(20)

        moire = r.get("moire", {})
        rolling = r.get("rolling", {})
        if moire:
            decision = rolling.get("decision", moire.get("decision_hint", "clean"))
            m_color = C_SPOOF if decision == "block" else (C_ORANGE if decision == "suspicious" else C_PRESENT)
            cv2.putText(panel, f"Moire V4: {moire.get('moire_score', 0):.0%}", (S(14), y), FONT_SMALL, FS(0.45), m_color, 1, cv2.LINE_AA)
            bar_x2 = S(140)
            bar_w2 = panel_w - bar_x2 - S(16)
            filled2 = int(bar_w2 * clamp(moire.get("moire_score", 0.0)))
            cv2.rectangle(panel, (bar_x2, y - S(10)), (bar_x2 + bar_w2, y), C_BLACK, -1)
            cv2.rectangle(panel, (bar_x2, y - S(10)), (bar_x2 + filled2, y), m_color, -1)
            y += S(18)
            cv2.putText(panel, f"  {decision.upper()} raw:{moire.get('raw_score', 0):.2f} med:{moire.get('smooth_score', 0):.2f}", (S(14), y), FONT_SMALL, FS(0.42), m_color, 1, cv2.LINE_AA)
            y += S(18)
            cv2.putText(panel, f"  Track:{moire.get('track_id', '-')} N:{moire.get('samples', 0)} Mode:{moire.get('pipeline_mode', '')[:3]}", (S(14), y), FONT_SMALL, FS(0.40), C_DIM, 1, cv2.LINE_AA)
            y += S(18)
            cv2.putText(panel, f"  Pk:{moire.get('peak_ratio', 0):.1f} Pd:{moire.get('periodicity', 0):.1f} An:{moire.get('anisotropy', 0):.2f} Gr:{moire.get('grid_score', 0):.2f}", (S(14), y), FONT_SMALL, FS(0.38), C_DIM, 1, cv2.LINE_AA)
            y += S(18)

        screen_context = r.get("screen_context", {})
        if screen_context:
            ctx_decision = screen_context.get("decision", "clean")
            ctx_color = C_SPOOF if ctx_decision == "strong" else (C_ORANGE if ctx_decision == "suspicious" else C_PRESENT)
            cv2.putText(
                panel,
                f"Context V4.1: {ctx_decision.upper()} {screen_context.get('score', 0):.2f}",
                (S(14), y),
                FONT_SMALL,
                FS(0.45),
                ctx_color,
                1,
                cv2.LINE_AA,
            )
            y += S(18)
            flat = screen_context.get("flatness", {})
            glare = screen_context.get("glare", {})
            cv2.putText(
                panel,
                f"  Flat:{flat.get('score', 0):.2f} Glare:{glare.get('score', 0):.2f}",
                (S(14), y),
                FONT_SMALL,
                FS(0.40),
                C_DIM,
                1,
                cv2.LINE_AA,
            )
            y += S(18)

        phone_rect = r.get("phone_rect", {})
        phone_rect_rolling = r.get("phone_rect_rolling", {})
        if phone_rect:
            phone_decision = phone_rect_rolling.get("decision", phone_rect.get("decision", "clean"))
            phone_color = C_SPOOF if phone_decision == "block" else (C_ORANGE if phone_decision in {"suspicious", "strong"} else C_PRESENT)
            cv2.putText(
                panel,
                f"Phone Rect V4.2: {phone_decision.upper()} {phone_rect.get('score', 0):.2f}",
                (S(14), y),
                FONT_SMALL,
                FS(0.45),
                phone_color,
                1,
                cv2.LINE_AA,
            )
            y += S(18)
            cv2.putText(
                panel,
                f"  In:{phone_rect.get('face_inside_score', 0):.2f} Edge:{phone_rect.get('border_edge_score', 0):.2f} Dark:{phone_rect.get('black_border_score', 0):.2f}",
                (S(14), y),
                FONT_SMALL,
                FS(0.38),
                C_DIM,
                1,
                cv2.LINE_AA,
            )
            y += S(18)
            cv2.putText(
                panel,
                f"  Rect:{phone_rect.get('rectangularity', 0):.2f} Asp:{phone_rect.get('aspect_ratio', 0):.2f} Strong:{phone_rect_rolling.get('strong_count', 0)}",
                (S(14), y),
                FONT_SMALL,
                FS(0.38),
                C_DIM,
                1,
                cv2.LINE_AA,
            )
            y += S(18)

        quality = r.get("quality")
        if quality:
            cv2.putText(panel, f"Blur: {quality['blur']:.0f}  Bright: {quality['bright']:.0f}", (S(14), y), FONT_SMALL, FS(0.42), C_DIM, 1, cv2.LINE_AA)
            y += S(18)
            cv2.putText(panel, f"Yaw: {quality['yaw']:.1f}deg  Size: {quality['size']}px", (S(14), y), FONT_SMALL, FS(0.42), C_DIM, 1, cv2.LINE_AA)
            y += S(18)
            q_status = "PASS" if quality["passed"] else "FAIL"
            q_color = C_PRESENT if quality["passed"] else C_SPOOF
            cv2.putText(panel, f"Quality: {q_status}", (S(14), y), FONT_SMALL, FS(0.45), q_color, 1, cv2.LINE_AA)
            y += S(22)

        live = r.get("liveness", {})
        if live:
            state = live.get("state")
            if state in {"checking", "blink_required"}:
                lv_color, lv_status = C_GOLD, "CHECKING"
            elif state == "live":
                lv_color, lv_status = C_PRESENT, "LIVE"
            elif state == "spoof":
                lv_color, lv_status = C_SPOOF, "SPOOF"
            else:
                lv_color, lv_status = C_DIM, str(state or "N/A").upper()
            cv2.putText(panel, f"Liveness: {lv_status}", (S(14), y), FONT_SMALL, FS(0.45), lv_color, 1, cv2.LINE_AA)
            y += S(20)
            cv2.putText(panel, f"  Blinks: {live.get('blinks', 0)}  EAR: {live.get('ear', 0):.2f}", (S(14), y), FONT_SMALL, FS(0.42), C_DIM, 1, cv2.LINE_AA)
            y += S(18)
            cv2.putText(panel, f"  Movement: {live.get('movement', 0):.1f}px", (S(14), y), FONT_SMALL, FS(0.42), C_DIM, 1, cv2.LINE_AA)
            y += S(18)
            cv2.putText(panel, f"  Track: {live.get('track_time', 0):.1f}s", (S(14), y), FONT_SMALL, FS(0.42), C_DIM, 1, cv2.LINE_AA)
            y += S(18)

        if r.get("emb_dim"):
            emb_count_txt = f"Embedding: {r['emb_dim']}D"
            if r.get("emb_count"):
                ec = r["emb_count"]
                ec_color = C_PRESENT if ec >= 3 else (C_GOLD if ec >= 2 else C_ORANGE)
                emb_count_txt += f"  Angles: {ec}"
                cv2.putText(panel, emb_count_txt, (S(14), y), FONT_SMALL, FS(0.45), ec_color, 1, cv2.LINE_AA)
                y += S(18)
                enroll_txt = "Multi-angle V2" if ec >= 3 else "Single (re-enroll)"
                cv2.putText(panel, f"  Enroll: {enroll_txt}", (S(14), y), FONT_SMALL, FS(0.42), ec_color, 1, cv2.LINE_AA)
            else:
                cv2.putText(panel, emb_count_txt, (S(14), y), FONT_SMALL, FS(0.45), C_DIM, 1, cv2.LINE_AA)
            y += S(20)

        if debug_enabled:
            passive = r.get("passive", {})
            cv2.putText(panel, f"Passive: {passive.get('score', 0):.2f} {passive.get('decision', '')}", (S(14), y), FONT_SMALL, FS(0.35), C_DIM, 1, cv2.LINE_AA)
            y += S(18)
            signals = ", ".join(moire.get("strong_signals", [])[:3])
            cv2.putText(panel, signals[:42], (S(14), y), FONT_SMALL, FS(0.32), C_ORANGE, 1, cv2.LINE_AA)
            y += S(18)
            ctx_signals = ", ".join(screen_context.get("signals", [])[:3])
            if ctx_signals:
                cv2.putText(panel, ctx_signals[:42], (S(14), y), FONT_SMALL, FS(0.32), C_ORANGE, 1, cv2.LINE_AA)
                y += S(18)
            phone_signals = ", ".join(phone_rect.get("signals", [])[:3])
            if phone_signals:
                cv2.putText(panel, phone_signals[:42], (S(14), y), FONT_SMALL, FS(0.32), C_ORANGE, 1, cv2.LINE_AA)
                y += S(18)

        cv2.line(panel, (S(14), y), (panel_w - S(14), y), (60, 60, 65), 1)
        y += S(14)

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
    moire_tracks: FaceMoireTrackStore,
    context_detector: ScreenContextDetectorV41,
    phone_rect_detector: PhoneRectangleDetectorV42,
    rolling_by_face: dict[int, RollingMoireDecision],
    phone_rolling_by_face: dict[int, RollingPhoneRectDecision],
    last_moire: dict[int, dict[str, Any]],
    last_rolling: dict[int, dict[str, Any]],
    run_moire: bool,
    challenge: ChallengeControllerV4,
    logger: CalibrationLogger,
) -> dict[str, Any]:
    bbox = bbox_list(face.bbox)
    x1, y1, x2, y2 = bbox
    roi, moire_roi_bbox = expanded_roi(frame, bbox)
    moire_track = moire_tracks.match_and_update(bbox)
    track_id = moire_track.track_id

    if run_moire or not moire_track.last_result:
        moire = moire_detector.analyze(roi, moire_track)
        rolling = rolling_by_face[track_id].update(moire)
        last_moire[track_id] = moire
        last_rolling[track_id] = rolling
    else:
        moire = moire_track.last_result or last_moire.get(track_id, moire_detector.last_result or {})
        rolling = last_rolling.get(track_id, rolling_by_face[track_id].summary())

    live = liveness.get_liveness(bbox)
    passive = anti_spoof.check(frame, face.bbox)
    passive_status = passive_decision(float(passive.score))
    screen_context = context_detector.analyze(frame, bbox, moire_roi_bbox)
    phone_rect = phone_rect_detector.analyze(frame, bbox)
    phone_rect_rolling = phone_rolling_by_face[track_id].update(phone_rect)
    metrics = engine.get_face_metrics(frame, face)
    match = engine.match_with_threshold(face.embedding, V4_COSINE_THRESHOLD)

    name = "Unknown"
    label = f"No match ({match.score:.0%})"
    extra = ""
    status = "unknown"
    color = C_UNKNOWN
    student = {}
    emb_count = 0

    moire_decision = moire.get("decision_hint", rolling.get("decision", "clean"))
    if rolling.get("decision") == "block" or moire_decision == "block":
        status = "spoof"
        name = "SPOOF"
        label = f"SCREEN detected ({moire.get('moire_score', 0):.0%})"
        extra = f"Moire track:{moire.get('track_id')} raw:{moire.get('raw_score', 0):.2f}"
        color = C_SPOOF
    elif live.get("state") == "spoof":
        status = "spoof"
        name = "SPOOF"
        label = live.get("message", "Liveness failed")
        extra = f"Blinks:{live.get('blinks', 0)} Track:{live.get('track_time', 0):.1f}s"
        color = C_SPOOF
    elif (
        screen_context.get("decision") == "strong"
        and (rolling.get("decision") == "suspicious" or passive_status == "suspicious")
    ):
        status = "spoof"
        name = "SPOOF"
        label = "Screen context strong"
        extra = ", ".join(screen_context.get("signals", [])[:3])
        color = C_SPOOF
    elif phone_rect_rolling.get("decision") == "block":
        status = "spoof"
        name = "SPOOF"
        label = "Phone rectangle detected"
        extra = f"Strong:{phone_rect_rolling.get('strong_count', 0)} Score:{phone_rect.get('score', 0):.2f}"
        color = C_SPOOF
    elif (
        phone_rect.get("decision") == "strong"
        and (
            moire_decision == "suspicious"
            or screen_context.get("decision") in {"suspicious", "strong"}
            or passive_status == "suspicious"
        )
    ):
        status = "spoof"
        name = "SPOOF"
        label = "Phone rectangle strong"
        extra = ", ".join(phone_rect.get("signals", [])[:3])
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
        if moire_decision == "suspicious":
            suspicious_reasons.append("moire suspicious")
        if screen_context.get("decision") in {"suspicious", "strong"}:
            suspicious_reasons.append(f"context {screen_context.get('decision')}")
        if phone_rect.get("decision") in {"suspicious", "strong"}:
            suspicious_reasons.append(f"phone rectangle {phone_rect.get('decision')}")
        if phone_rect_rolling.get("decision") == "suspicious":
            suspicious_reasons.append("phone rectangle rolling")
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
            extra = f"Emb:{emb_count} Moire:{moire_decision} Ctx:{screen_context.get('decision')} Phone:{phone_rect.get('decision')} Live:OK"
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
        "aligned": face.aligned_face,
        "name": name,
        "label": label,
        "extra": extra,
        "status": status,
        "color": color,
        "confidence": float(match.score),
        "conf": float(match.score),
        "det_conf": float(face.confidence),
        "matched": bool(match.matched),
        "match_score": float(match.score),
        "student_id": match.student_id if match.matched else "",
        "sid": match.student_id if match.matched else "",
        "class_name": student.get("class_name", ""),
        "embedding_count": emb_count,
        "emb_count": emb_count,
        "emb_dim": int(len(face.embedding)) if face.embedding is not None else 0,
        "quality": {
            "passed": bool(metrics.get("passed", False)),
            "blur": float(metrics.get("blur_score", 0.0)),
            "bright": float(metrics.get("brightness", 0.0)),
            "yaw": float(metrics.get("yaw_angle", 0.0)),
            "size": int(metrics.get("face_size", 0)),
            "reasons": metrics.get("reasons", []),
        },
        "moire": moire,
        "rolling": rolling,
        "screen_context": screen_context,
        "phone_rect": phone_rect,
        "phone_rect_rolling": phone_rect_rolling,
        "liveness": live,
        "passive": {
            "score": float(passive.score),
            "reason": passive.reason,
            "decision": passive_status,
        },
        "metrics": metrics,
        "moire_roi_bbox": moire_roi_bbox,
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
    print("  Detect V4.3 Research - Enhanced moire + context + phone rectangle")
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
    moire_tracks = FaceMoireTrackStore()
    context_detector = ScreenContextDetectorV41()
    phone_rect_detector = PhoneRectangleDetectorV42()
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

    win_name = "Live Face Detection V4.3"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    total_w = FRAME_W + INFO_PANEL_W
    total_h = FRAME_H
    fit_scale = min((SCREEN_W * 0.92) / total_w, (SCREEN_H * 0.90) / total_h, 1.0)
    cv2.resizeWindow(win_name, int(total_w * fit_scale), int(total_h * fit_scale))

    rolling_by_face: dict[int, RollingMoireDecision] = defaultdict(RollingMoireDecision)
    phone_rolling_by_face: dict[int, RollingPhoneRectDecision] = defaultdict(RollingPhoneRectDecision)
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
                    moire_tracks.begin_cycle()
                    if not faces:
                        # Clear stale per-index rolling when no face remains visible.
                        if detect_cycle % (MOIRE_EVERY_N_DETECT * 8) == 0:
                            last_moire.clear()
                            last_rolling.clear()
                            rolling_by_face.clear()
                            phone_rolling_by_face.clear()

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
                            moire_tracks=moire_tracks,
                            context_detector=context_detector,
                            phone_rect_detector=phone_rect_detector,
                            rolling_by_face=rolling_by_face,
                            phone_rolling_by_face=phone_rolling_by_face,
                            last_moire=last_moire,
                            last_rolling=last_rolling,
                            run_moire=run_moire,
                            challenge=challenge,
                            logger=cal_logger,
                        )
                        last_results.append(entry)
                    active_track_ids = moire_tracks.finish_cycle()
                    for track_id in list(rolling_by_face.keys()):
                        if track_id not in active_track_ids:
                            rolling_by_face.pop(track_id, None)
                            phone_rolling_by_face.pop(track_id, None)
                            last_moire.pop(track_id, None)
                            last_rolling.pop(track_id, None)
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
                    mx1, my1, mx2, my2 = result.get("moire_roi_bbox", result["bbox"])
                    cv2.rectangle(frame, (mx1, my1), (mx2, my2), C_PURPLE, 1, cv2.LINE_AA)
                    draw_moire_spectrum(frame, safe_roi(frame, result.get("moire_roi_bbox", result["bbox"])), x1, y1)
                    phone_roi = result.get("phone_rect", {}).get("roi_bbox", [])
                    if len(phone_roi) == 4:
                        px1, py1, px2, py2 = phone_roi
                        cv2.rectangle(frame, (px1, py1), (px2, py2), C_GOLD, 1, cv2.LINE_AA)
                    draw_phone_rectangle_overlay(frame, result.get("phone_rect"), result.get("phone_rect_rolling", {}))

            draw_challenge_overlay(frame, challenge)
            draw_hud(
                frame,
                display_fps,
                len(last_results),
                len(emb_counts),
                show_landmarks,
                show_info_panel,
                show_moire_overlay,
                cal_logger.enabled,
                debug_enabled,
            )
            perf_text = f"Det:{perf_detect_ms:.0f}ms  Moire:{perf_moire_ms:.0f}ms  FFT:{MOIRE_PIPELINE_MODE}  Screen<{MOIRE_SCREEN_THRESHOLD:.2f} Block<{MOIRE_BLOCK_THRESHOLD:.2f}  PhoneRect:ON  Skip:{MOIRE_EVERY_N_DETECT - 1}/{MOIRE_EVERY_N_DETECT}"
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
