"""
Detect V4.4 backend service and anti-spoof detectors.

This module ports the non-UI logic from dev/detect-v4.4.py into backend code:
enhanced moire analysis, screen-context evidence, portrait phone-rectangle
evidence, rolling decisions, and attendance result helpers.
"""
from __future__ import annotations

import cv2
import numpy as np

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from core.database import get_db


DETECT_VERSION = "v4.4"

V4_COSINE_THRESHOLD = 0.52

MOIRE_CONTEXT_SCALE = 1.65
MOIRE_EVERY_N_DETECT = 3
MOIRE_PIPELINE_MODE = "enhanced"
MOIRE_SCREEN_THRESHOLD = 0.60
MOIRE_BLOCK_THRESHOLD = 0.45
MOIRE_SMOOTH_WINDOW = 7
MOIRE_TRACK_TTL = 12
MOIRE_IOU_MATCH_THRESHOLD = 0.25
MOIRE_ROLLING_CLEAN_MEAN_MIN = 0.55

CHALLENGE_COOLDOWN = 12.0
CHALLENGE_TYPES = [
    "TURN_LEFT",
    "TURN_RIGHT",
    "LOOK_UP",
    "LOOK_DOWN",
    "OPEN_MOUTH",
    "CENTER_HOLD",
]
CHALLENGE_ACTION_GROUPS = {
    "TURN_LEFT": "yaw",
    "TURN_RIGHT": "yaw",
    "LOOK_UP": "pitch",
    "LOOK_DOWN": "pitch",
    "OPEN_MOUTH": "facial",
    "CENTER_HOLD": "stability",
}
CHALLENGE_TIMEOUT = 7.0
CHALLENGE_BASELINE_FRAMES = 3
CHALLENGE_POSE_FRAMES = 3
CHALLENGE_MOUTH_FRAMES = 3
CHALLENGE_CENTER_HOLD_FRAMES = 3
CHALLENGE_MIN_YAW_ANGLE = 12.0
CHALLENGE_CENTER_MAX_YAW_ANGLE = 8.0
TURN_THRESHOLD = 0.10
LOOK_UP_THRESHOLD = 0.06
LOOK_DOWN_THRESHOLD = 0.08
LOOK_PITCH_CENTER_YAW_THRESHOLD = 0.08
OPEN_MOUTH_DELTA = 0.12
OPEN_MOUTH_MIN_RATIO = 0.22
CENTER_YAW_THRESHOLD = 0.05
CENTER_PITCH_THRESHOLD = 0.05
CENTER_SCALE_DELTA = 0.14
CHALLENGE_RECENTER_PITCH_THRESHOLD = 0.08

SCREEN_CONTEXT_ENABLED = True
SCREEN_CONTEXT_WEIGHT = 0.35
FLATNESS_SUSPICIOUS_THRESHOLD = 0.65
GLARE_SUSPICIOUS_THRESHOLD = 0.45
SCREEN_CONTEXT_STRONG_THRESHOLD = 0.78

PHONE_RECT_ENABLED = True
PHONE_RECT_CONTEXT_SCALE = 2.80
PHONE_RECT_VERTICAL_RATIO = 1.6
PHONE_RECT_SUSPICIOUS_THRESHOLD = 0.38
PHONE_RECT_STRONG_THRESHOLD = 0.58
PHONE_RECT_ROLLING_STRONG_COUNT = 2


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def bbox_list(bbox: Any) -> list[int]:
    return [int(v) for v in bbox[:4]]


def safe_roi(frame: np.ndarray, bbox: list[int]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=frame.dtype)
    return frame[y1:y2, x1:x2]


def bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def expanded_roi(
    frame: np.ndarray,
    bbox: list[int],
    scale: float = MOIRE_CONTEXT_SCALE,
) -> tuple[np.ndarray, list[int]]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(bw, bh) * scale / 2.0
    ex1 = max(0, int(round(cx - half)))
    ey1 = max(0, int(round(cy - half)))
    ex2 = min(w, int(round(cx + half)))
    ey2 = min(h, int(round(cy + half)))
    if ex2 <= ex1 or ey2 <= ey1:
        return safe_roi(frame, bbox), bbox
    return frame[ey1:ey2, ex1:ex2], [ex1, ey1, ex2, ey2]


@dataclass
class FaceMoireTrack:
    track_id: int
    bbox: list[int]
    ttl: int = MOIRE_TRACK_TTL
    scores: deque = field(default_factory=lambda: deque(maxlen=MOIRE_SMOOTH_WINDOW))
    last_result: dict[str, Any] = field(default_factory=dict)

    def update_score(self, score: float) -> float:
        if self.scores.maxlen != MOIRE_SMOOTH_WINDOW:
            self.scores = deque(self.scores, maxlen=MOIRE_SMOOTH_WINDOW)
        self.scores.append(float(score))
        return float(np.median(np.asarray(self.scores, dtype=np.float32)))


class FaceMoireTrackStore:
    def __init__(self):
        self.tracks: dict[int, FaceMoireTrack] = {}
        self.next_id = 1
        self._seen_this_cycle: set[int] = set()

    def begin_cycle(self):
        self._seen_this_cycle = set()

    def match_and_update(self, bbox: list[int]) -> FaceMoireTrack:
        best_track = None
        best_iou = 0.0
        for track in self.tracks.values():
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
        self._seen_this_cycle.add(best_track.track_id)
        return best_track

    def finish_cycle(self) -> set[int]:
        expired = []
        for track_id, track in self.tracks.items():
            if track_id not in self._seen_this_cycle:
                track.ttl -= 1
            if track.ttl <= 0:
                expired.append(track_id)
        for track_id in expired:
            self.tracks.pop(track_id, None)
        return set(self.tracks.keys())


class MoireDetectorV4:
    """Enhanced multi-band FFT moire detector from Detect V4.4."""

    ANALYZE_SIZE = 128
    BANDS = {
        "low_mid": (0.15, 0.35),
        "mid": (0.35, 0.55),
        "mid_high": (0.55, 0.75),
        "high": (0.75, 0.92),
    }

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

    def analyze(
        self,
        face_roi: np.ndarray,
        track: FaceMoireTrack | None = None,
    ) -> dict[str, Any]:
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
        return clamp((ratio - 1.4) / 3.0)


class ScreenContextDetectorV41:
    def analyze(
        self,
        frame: np.ndarray,
        face_bbox: list[int],
        roi_bbox: list[int] | None = None,
    ) -> dict[str, Any]:
        if not SCREEN_CONTEXT_ENABLED:
            return self._empty("disabled")

        roi_bbox = roi_bbox or expanded_roi(frame, face_bbox)[1]
        roi = safe_roi(frame, roi_bbox)
        if roi.size == 0:
            return self._empty("empty_roi")

        local_face = self._local_bbox(face_bbox, roi_bbox)
        context_mask = self._context_mask(roi.shape[:2], local_face)
        boundary_mask = self._boundary_mask(roi.shape[:2], local_face)
        if int(np.sum(context_mask > 0)) < 80:
            return self._empty("small_context")

        flatness = self._flatness_metrics(roi, context_mask, boundary_mask)
        glare = self._glare_metrics(roi, context_mask)

        flat_score = float(flatness["score"])
        glare_score = float(glare["score"])
        score = clamp((flat_score * 0.60 + glare_score * 0.40) * SCREEN_CONTEXT_WEIGHT / 0.35)

        signals = []
        if flat_score >= FLATNESS_SUSPICIOUS_THRESHOLD:
            signals.append("flat_background")
        if glare_score >= GLARE_SUSPICIOUS_THRESHOLD:
            signals.append("glass_glare")

        strong_signals = [
            signal for signal in signals
            if signal == "flat_background" and flat_score >= 0.78
            or signal == "glass_glare" and glare_score >= 0.60
        ]

        if score >= SCREEN_CONTEXT_STRONG_THRESHOLD and strong_signals:
            decision = "strong"
        elif signals:
            decision = "suspicious"
        else:
            decision = "clean"

        return {
            "score": round(score, 3),
            "decision": decision,
            "signals": signals,
            "flatness": flatness,
            "glare": glare,
            "roi_bbox": roi_bbox,
        }

    def _empty(self, reason: str) -> dict[str, Any]:
        return {
            "score": 0.0,
            "decision": "clean",
            "signals": [reason] if reason != "disabled" else [],
            "flatness": {},
            "glare": {},
            "roi_bbox": [],
        }

    def _local_bbox(self, face_bbox: list[int], roi_bbox: list[int]) -> list[int]:
        fx1, fy1, fx2, fy2 = face_bbox
        rx1, ry1, _, _ = roi_bbox
        return [fx1 - rx1, fy1 - ry1, fx2 - rx1, fy2 - ry1]

    def _context_mask(self, shape: tuple[int, int], local_face: list[int]) -> np.ndarray:
        h, w = shape
        mask = np.full((h, w), 255, dtype=np.uint8)
        x1, y1, x2, y2 = self._clip_bbox(local_face, w, h)
        pad_x = max(4, int((x2 - x1) * 0.08))
        pad_y = max(4, int((y2 - y1) * 0.08))
        cv2.rectangle(
            mask,
            (max(0, x1 - pad_x), max(0, y1 - pad_y)),
            (min(w - 1, x2 + pad_x), min(h - 1, y2 + pad_y)),
            0,
            -1,
        )
        return mask

    def _boundary_mask(self, shape: tuple[int, int], local_face: list[int]) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = self._clip_bbox(local_face, w, h)
        pad_x = max(6, int((x2 - x1) * 0.18))
        pad_y = max(6, int((y2 - y1) * 0.18))
        cv2.rectangle(
            mask,
            (max(0, x1 - pad_x), max(0, y1 - pad_y)),
            (min(w - 1, x2 + pad_x), min(h - 1, y2 + pad_y)),
            255,
            max(2, int(min(w, h) * 0.015)),
        )
        return mask

    def _flatness_metrics(
        self,
        roi: np.ndarray,
        context_mask: np.ndarray,
        boundary_mask: np.ndarray,
    ) -> dict[str, Any]:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        masked_values = gray[context_mask > 0]
        if masked_values.size < 32:
            return {
                "score": 0.0,
                "laplacian_var": 0.0,
                "gradient_entropy": 0.0,
                "color_std": 0.0,
                "edge_density": 0.0,
                "boundary_edge_strength": 0.0,
            }

        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap_values = lap[context_mask > 0]
        lap_var = float(np.var(lap_values)) if lap_values.size else 0.0

        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(sobel_x, sobel_y)
        grad_values = grad[context_mask > 0]
        grad_entropy = self._entropy(
            np.clip(grad_values, 0, 255).astype(np.uint8),
            bins=16,
            value_range=(0, 255),
        )

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color_std = float(np.mean(np.std(hsv[context_mask > 0], axis=0))) if masked_values.size else 0.0

        edges = cv2.Canny(gray, 60, 140)
        edge_density = float(np.mean(edges[context_mask > 0] > 0))
        boundary_values = grad[boundary_mask > 0]
        boundary_edge_strength = float(np.mean(boundary_values)) if boundary_values.size else 0.0

        flat_score = clamp(
            clamp((42.0 - lap_var) / 42.0) * 0.33
            + clamp((2.9 - grad_entropy) / 2.9) * 0.24
            + clamp((28.0 - color_std) / 28.0) * 0.18
            + clamp((0.08 - edge_density) / 0.08) * 0.15
            + clamp((boundary_edge_strength - 8.0) / 35.0) * 0.10
        )

        return {
            "score": round(flat_score, 3),
            "laplacian_var": round(lap_var, 2),
            "gradient_entropy": round(float(grad_entropy), 3),
            "color_std": round(color_std, 2),
            "edge_density": round(edge_density, 3),
            "boundary_edge_strength": round(boundary_edge_strength, 2),
        }

    def _glare_metrics(self, roi: np.ndarray, context_mask: np.ndarray) -> dict[str, Any]:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
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
            _, _, w, h = cv2.boundingRect(contour)
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

    def _entropy(
        self,
        values: np.ndarray,
        *,
        bins: int,
        value_range: tuple[int, int],
    ) -> float:
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
    """Detect phone/screen rectangle evidence around a detected face."""

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
        (_, _), (rw, rh), _ = rect
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

        aspect_ratio = max(rw, rh) / max(1.0, min(rw, rh))
        aspect_score = clamp(1.0 - abs(aspect_ratio - 1.78) / 1.4)
        if 1.20 <= aspect_ratio <= 2.80:
            aspect_score = max(aspect_score, 0.40)

        face_inside_score = self._face_inside_score(box_float, local_face)
        rect_area_score = clamp((rect_area_ratio - 0.10) / 0.45)
        border_edge_score, black_border_score = self._border_scores(roi, edges, box)
        corner_score = self._corner_score(contour)
        margin_score = self._margin_score([x, y, x + w, y + h], local_face)

        score = clamp(
            rectangularity_score * 0.18
            + aspect_score * 0.14
            + face_inside_score * 0.24
            + rect_area_score * 0.10
            + border_edge_score * 0.12
            + black_border_score * 0.12
            + corner_score * 0.05
            + margin_score * 0.05
        )

        abs_points = [
            [int(p[0] + roi_bbox[0]), int(p[1] + roi_bbox[1])]
            for p in box.tolist()
        ]

        return {
            "score": score,
            "best_rect": [int(x + roi_bbox[0]), int(y + roi_bbox[1]), int(w), int(h)],
            "best_rect_points": abs_points,
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

    def _border_scores(
        self,
        roi: np.ndarray,
        edges: np.ndarray,
        box: np.ndarray,
    ) -> tuple[float, float]:
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

        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (thickness * 2 + 1, thickness * 2 + 1),
        )
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

        scores = np.asarray(
            [float(s.get("moire_score", s.get("raw_score", 0.5))) for s in self.samples],
            dtype=np.float32,
        )
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


def passive_decision(score: float) -> str:
    if score < config.DETECT_V3_LIVENESS_BLOCK_THRESHOLD:
        return "block"
    if score < config.DETECT_V3_LIVENESS_CHALLENGE_THRESHOLD:
        return "suspicious"
    return "pass"


def collect_suspicious_reasons(
    *,
    moire_decision: str,
    screen_context: dict[str, Any],
    phone_rect: dict[str, Any],
    phone_rect_rolling: dict[str, Any],
    passive_status: str,
    passive_score: float,
    challenge_fail_count: int,
) -> list[str]:
    reasons = []
    if moire_decision == "suspicious":
        reasons.append("moire suspicious")
    if screen_context.get("decision") in {"suspicious", "strong"}:
        reasons.append(f"context {screen_context.get('decision')}")
    if phone_rect.get("decision") in {"suspicious", "strong"}:
        reasons.append(f"phone rectangle {phone_rect.get('decision')}")
    if phone_rect_rolling.get("decision") == "suspicious":
        reasons.append("phone rectangle rolling")
    if passive_status == "suspicious":
        reasons.append(f"passive {passive_score:.2f}")
    if challenge_fail_count >= 2:
        reasons.append("recent challenge failures")
    return reasons


def assess_challenge_need(
    *,
    moire_decision: str,
    screen_context: dict[str, Any],
    phone_rect: dict[str, Any],
    phone_rect_rolling: dict[str, Any],
    passive_status: str,
    passive_score: float,
    challenge_fail_count: int,
) -> dict[str, Any]:
    suspicious_reasons = []
    strong_reasons = []

    if moire_decision == "suspicious":
        suspicious_reasons.append("moire suspicious")

    screen_decision = screen_context.get("decision")
    if screen_decision == "strong":
        strong_reasons.append("context strong")
    elif screen_decision == "suspicious":
        suspicious_reasons.append("context suspicious")

    phone_decision = phone_rect.get("decision")
    if phone_decision == "strong":
        strong_reasons.append("phone rectangle strong")
    elif phone_decision == "suspicious":
        suspicious_reasons.append("phone rectangle suspicious")

    if phone_rect_rolling.get("decision") == "suspicious":
        suspicious_reasons.append("phone rectangle rolling")

    if passive_status == "suspicious":
        suspicious_reasons.append(f"passive {passive_score:.2f}")

    if challenge_fail_count >= 2:
        strong_reasons.append("recent challenge failures")

    severity = "none"
    if strong_reasons:
        severity = "strong"
    elif len(suspicious_reasons) >= 2:
        severity = "medium"

    return {
        "should_challenge": severity != "none",
        "severity": severity,
        "reasons": strong_reasons + suspicious_reasons,
        "strong_reasons": strong_reasons,
        "suspicious_reasons": suspicious_reasons,
    }


class DetectV4Service:
    """Attendance result helper for HTTP and streaming Detect V4 runtimes."""

    def record_attendance_result(
        self,
        *,
        frame: np.ndarray,
        session_id: int,
        session: dict | None,
        match: Any,
        student: dict,
        emb_count: int,
        bbox: list[int],
        moire_score: float,
        liveness_score: float,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        evidence = str(config.EVIDENCE_DIR / f"{match.student_id}_{ts}.jpg")
        cv2.imwrite(evidence, frame)

        db = get_db()
        db_result = db.mark_attendance(
            session_id, match.student_id, match.score, evidence
        )
        evidence_filename = Path(evidence).name

        return {
            "name": match.name,
            "student_id": match.student_id,
            "class_name": student.get("class_name", ""),
            "session_id": session_id,
            "session_name": (
                session.get("name", f"Session #{session_id}")
                if session else f"Session #{session_id}"
            ),
            "confidence": float(match.score),
            "status": "present" if db_result["success"] else "already",
            "message": db_result["message"],
            "bbox": bbox,
            "evidence_url": f"/api/evidence/{evidence_filename}",
            "moire_score": moire_score,
            "moire_is_screen": False,
            "liveness_score": liveness_score,
            "embedding_count": emb_count,
            "enroll_type": "multi_angle_v2" if emb_count >= 3 else "single",
            "challenge_required": False,
            "scan_version": DETECT_VERSION,
            "diagnostics": diagnostics or {},
        }

    def spoof_result(
        self,
        *,
        bbox: list[int],
        message: str,
        moire_score: float,
        moire_is_screen: bool,
        liveness_score: float | None,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "name": "Unknown",
            "student_id": "",
            "confidence": 0,
            "status": "spoof",
            "message": message,
            "bbox": bbox,
            "moire_score": moire_score,
            "moire_is_screen": moire_is_screen,
            "liveness_score": liveness_score,
            "challenge_required": False,
            "scan_version": DETECT_VERSION,
            "diagnostics": diagnostics or {},
        }

    def unknown_result(
        self,
        *,
        bbox: list[int],
        match: Any,
        moire_score: float,
        liveness_score: float,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "name": "Unknown",
            "student_id": "",
            "confidence": float(match.score),
            "status": "unknown",
            "message": (
                f"No match (score={match.score:.3f}, "
                f"threshold={V4_COSINE_THRESHOLD})"
            ),
            "bbox": bbox,
            "moire_score": moire_score,
            "moire_is_screen": False,
            "liveness_score": liveness_score,
            "challenge_required": False,
            "scan_version": DETECT_VERSION,
            "diagnostics": diagnostics or {},
        }

    def candidate_from_match(
        self,
        *,
        match: Any,
        student: dict,
        emb_count: int,
        bbox: list[int],
    ) -> dict[str, Any]:
        return {
            "name": match.name,
            "student_id": match.student_id,
            "class_name": student.get("class_name", ""),
            "confidence": float(match.score),
            "bbox": bbox,
            "embedding_count": emb_count,
            "enroll_type": "multi_angle_v2" if emb_count >= 3 else "single",
        }


_detect_v4_service: DetectV4Service | None = None


def get_detect_v4_service() -> DetectV4Service:
    global _detect_v4_service
    if _detect_v4_service is None:
        _detect_v4_service = DetectV4Service()
    return _detect_v4_service
