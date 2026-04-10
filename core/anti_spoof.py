"""
Anti-Spoofing — Passive Liveness Detection.

Detects spoofed faces using three signals:
1. Texture analysis (LBP variance) — distinguishes real skin from printed/screen images
2. Reflection detection — identifies specular highlights characteristic of LCD screens
3. Color analysis — validates skin tone distribution in YCrCb space

No user action required (fully passive).
"""
import cv2
import numpy as np
from loguru import logger

import config
from core.schemas import LivenessResult


class AntiSpoof:
    """Passive liveness detection — rejects spoofed images and screen-displayed faces."""

    def __init__(self):
        self._enabled = config.LIVENESS_ENABLED
        self._threshold = config.LIVENESS_SCORE_THRESHOLD

    def check(self, frame: np.ndarray, bbox: list | np.ndarray) -> LivenessResult:
        """Run liveness check on a detected face region.

        Args:
            frame: Full BGR frame
            bbox: [x1, y1, x2, y2] bounding box

        Returns:
            LivenessResult: is_live, score, reason
        """
        if not self._enabled:
            return LivenessResult(is_live=True, score=1.0, reason="disabled")

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return LivenessResult(is_live=False, score=0.0, reason="empty_roi")

        # 3 signal scores
        texture_score = self._texture_analysis(face_roi)
        reflection_score = self._reflection_detection(face_roi)
        color_score = self._color_analysis(face_roi)

        # Weighted average
        score = (
            texture_score * 0.4 +
            reflection_score * 0.3 +
            color_score * 0.3
        )

        is_live = score >= self._threshold
        reasons = []
        if texture_score < 0.4:
            reasons.append("texture_flat")
        if reflection_score < 0.4:
            reasons.append("reflection_detected")
        if color_score < 0.4:
            reasons.append("color_unnatural")

        reason = "; ".join(reasons) if reasons else "pass"

        return LivenessResult(
            is_live=is_live,
            score=round(score, 3),
            reason=reason
        )

    def _texture_analysis(self, face_roi: np.ndarray) -> float:
        """Analyze texture using Local Binary Pattern histogram variance.

        Real faces have rich, varied texture (high LBP variance).
        Printed or screen images have uniform texture (low LBP variance).
        """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))

        # Simplified LBP: compare each pixel against its 8 neighbors
        h, w = gray.shape
        lbp = np.zeros_like(gray, dtype=np.uint8)
        for dy, dx, bit in [
            (-1, -1, 0), (-1, 0, 1), (-1, 1, 2),
            (0, 1, 3), (1, 1, 4), (1, 0, 5),
            (1, -1, 6), (0, -1, 7)
        ]:
            shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
            lbp |= ((gray >= shifted).astype(np.uint8) << bit)

        # LBP histogram variance — real faces have more uniform distribution
        hist = cv2.calcHist([lbp[1:-1, 1:-1]], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        variance = np.var(hist)

        # Normalize: low variance (~0.0001) = spoof, high variance (~0.001+) = real
        score = min(1.0, variance / 0.001)
        return score

    def _reflection_detection(self, face_roi: np.ndarray) -> float:
        """Detect screen reflections and specular highlights.

        LCD/phone screens exhibit characteristic specular highlights not present on real skin.
        """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Detect very bright regions (specular highlights)
        _, bright = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        bright_ratio = np.sum(bright > 0) / bright.size

        # Gradient analysis — screens produce smoother gradients than real skin
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        grad_std = np.std(grad_mag)

        # High bright_ratio or unnatural gradient pattern = spoof
        # Real faces: low bright_ratio, high gradient diversity
        reflection_penalty = bright_ratio * 10  # 10% bright pixels -> full penalty
        grad_score = min(1.0, grad_std / 40)    # gradient std >= 40 = natural

        score = max(0.0, min(1.0, grad_score - reflection_penalty))
        return score

    def _color_analysis(self, face_roi: np.ndarray) -> float:
        """Validate skin tone distribution in YCrCb color space.

        Real skin has a characteristic YCrCb range.
        Printed or screen images exhibit color distribution shift.
        """
        ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # Real skin tone: Cr in [133, 173], Cb in [77, 127]
        skin_mask = (
            (cr >= 133) & (cr <= 173) &
            (cb >= 77) & (cb <= 127)
        )
        skin_ratio = np.sum(skin_mask) / skin_mask.size

        # Saturation analysis
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        sat_std = np.std(saturation)

        # High skin coverage + diverse saturation = real face
        skin_score = min(1.0, skin_ratio / 0.3)    # 30%+ skin = good
        sat_score = min(1.0, sat_std / 30)          # sat std ~30+ = natural

        score = (skin_score * 0.6 + sat_score * 0.4)
        return score


# ── Singleton ───────────────────────────────────────────────

_anti_spoof = None

def get_anti_spoof() -> AntiSpoof:
    global _anti_spoof
    if _anti_spoof is None:
        _anti_spoof = AntiSpoof()
    return _anti_spoof
