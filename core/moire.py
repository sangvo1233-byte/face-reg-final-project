"""
Moiré Pattern Detector — Passive Anti-Spoof via FFT Analysis.

Detects moiré patterns that appear when a camera captures a screen.
When a 2D camera records an LCD/OLED screen, the camera's sensor grid
interferes with the screen's pixel grid, creating characteristic
moiré interference patterns visible in the frequency domain.

Real faces have smooth, organic frequency spectra.
Screen-displayed faces show sharp periodic peaks from moiré.

100% compatible with standard 2D webcams — no depth camera needed.
"""
import cv2
import numpy as np
from loguru import logger

import config


class MoireDetector:
    """Detect moiré patterns using FFT analysis on face ROI.

    Scoring: moire_score ∈ [0, 1]
        0 = definitely screen (strong moiré)
        1 = definitely real face (no moiré)

    is_screen = True when rolling-average score < threshold.
    """

    # Analysis size (square crop for consistent FFT)
    ANALYZE_SIZE = 64

    # Frequency band for moiré detection (normalized 0-1 of spectrum)
    FREQ_LOW  = 0.15   # ignore DC + very low freq (face structure)
    FREQ_HIGH = 0.85   # ignore very high freq (noise)

    # Thresholds
    PEAK_RATIO_THRESHOLD = 2.8
    ENERGY_RATIO_THRESHOLD = 0.35
    PERIODIC_THRESHOLD = 3.5

    def __init__(self, screen_threshold: float | None = None):
        self.screen_threshold = (
            screen_threshold
            if screen_threshold is not None
            else config.DETECT_V3_MOIRE_SCREEN_THRESHOLD
        )
        self.history: list[float] = []
        self.max_history = 20
        self.last_result: dict = {}

        # Pre-compute Hanning window and radial mask
        sz = self.ANALYZE_SIZE
        self._window = np.outer(
            np.hanning(sz), np.hanning(sz)
        ).astype(np.float32)

        cy, cx = sz // 2, sz // 2
        Y, X = np.ogrid[:sz, :sz]
        self._dist = np.sqrt(
            (X - cx) ** 2 + (Y - cy) ** 2
        ).astype(np.float32)

        max_r = min(cx, cy)
        r_low = int(max_r * self.FREQ_LOW)
        r_high = int(max_r * self.FREQ_HIGH)
        self._band_mask = (self._dist >= r_low) & (self._dist <= r_high)
        self._r_low = r_low
        self._r_high = r_high
        self._dc_mask = self._dist > 0
        self._hf_mask = self._dist >= r_low

    def analyze(self, face_roi: np.ndarray) -> dict:
        """Analyze a face ROI for moiré patterns using FFT.

        Args:
            face_roi: BGR face crop from the frame

        Returns:
            dict with moire_score (0=screen, 1=real), is_screen, details
        """
        if face_roi is None or face_roi.size == 0:
            return {
                "moire_score": 0.5, "is_screen": None,
                "peak_ratio": 0, "energy_ratio": 0, "periodicity": 0,
            }

        # Convert to grayscale and resize to fixed size
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.ANALYZE_SIZE, self.ANALYZE_SIZE))

        # Normalize to float and apply pre-computed window
        gray_f = gray.astype(np.float32) * (1.0 / 255.0)
        windowed = gray_f * self._window

        # 2D FFT
        fft = np.fft.fft2(windowed)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        log_mag = np.log1p(magnitude)

        # ── Signal 1: Peak-to-Mean ratio ──────────────────
        band_values = log_mag[self._band_mask]

        if band_values.size == 0 or np.mean(band_values) < 1e-6:
            return {
                "moire_score": 0.5, "is_screen": None,
                "peak_ratio": 0, "energy_ratio": 0, "periodicity": 0,
            }

        peak_ratio = float(np.max(band_values) / np.mean(band_values))

        # ── Signal 2: High-frequency energy ratio ─────────
        total_energy = np.sum(log_mag[self._dc_mask])
        if total_energy < 1e-6:
            total_energy = 1e-6
        high_freq_energy = np.sum(log_mag[self._hf_mask])
        energy_ratio = float(high_freq_energy / total_energy)

        # ── Signal 3: Periodicity detection ───────────────
        sz = self.ANALYZE_SIZE
        cx, cy = sz // 2, sz // 2
        periodicity = self._check_periodicity(
            log_mag, cx, cy, self._r_low, self._r_high
        )

        # ── Signal 4: Horizontal/Vertical line detection ──
        h_line_score = self._check_grid_lines(gray, direction="horizontal")
        v_line_score = self._check_grid_lines(gray, direction="vertical")
        grid_score = (h_line_score + v_line_score) / 2.0

        # ── Combined score ────────────────────────────────
        screen_evidence = 0.0

        if peak_ratio > self.PEAK_RATIO_THRESHOLD:
            screen_evidence += 0.30
        elif peak_ratio > self.PEAK_RATIO_THRESHOLD * 0.7:
            screen_evidence += 0.15

        if periodicity > self.PERIODIC_THRESHOLD:
            screen_evidence += 0.30
        elif periodicity > self.PERIODIC_THRESHOLD * 0.6:
            screen_evidence += 0.15

        if grid_score > 0.5:
            screen_evidence += 0.25
        elif grid_score > 0.3:
            screen_evidence += 0.10

        if energy_ratio > self.ENERGY_RATIO_THRESHOLD:
            screen_evidence += 0.15

        # Convert: 0 = definitely screen, 1 = definitely real
        moire_score = max(0.0, min(1.0, 1.0 - screen_evidence))

        # Rolling average for stability
        self.history.append(moire_score)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        avg_score = float(np.mean(self.history))

        is_screen = avg_score < self.screen_threshold

        self.last_result = {
            "moire_score": round(avg_score, 3),
            "raw_score": round(moire_score, 3),
            "is_screen": is_screen,
            "peak_ratio": round(peak_ratio, 2),
            "energy_ratio": round(energy_ratio, 3),
            "periodicity": round(periodicity, 2),
            "grid_score": round(grid_score, 3),
        }
        return self.last_result

    def analyze_single(self, face_roi: np.ndarray) -> dict:
        """Analyze a single image without rolling history.

        Useful for API calls where each request is independent.
        """
        saved_history = self.history.copy()
        self.history.clear()
        result = self.analyze(face_roi)
        self.history = saved_history
        return result

    def reset(self):
        """Clear rolling history."""
        self.history.clear()
        self.last_result = {}

    # ── Internal helpers ──────────────────────────────────

    def _check_periodicity(self, log_mag, cx, cy, r_low, r_high) -> float:
        """Check for periodic peaks in the frequency spectrum.

        Moiré creates evenly-spaced peaks along certain angles.
        """
        n_angles = 36  # sample every 10 degrees
        peak_counts = []

        for i in range(n_angles):
            angle = i * np.pi / n_angles
            samples = []
            for r in range(r_low, r_high, 2):
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))
                if 0 <= x < log_mag.shape[1] and 0 <= y < log_mag.shape[0]:
                    samples.append(log_mag[y, x])

            if len(samples) < 5:
                continue

            samples = np.array(samples)
            mean_val = np.mean(samples)
            if mean_val < 1e-6:
                continue

            # Count peaks above 2x mean (indicates periodic structure)
            peaks = np.sum(samples > mean_val * 2.0)
            peak_counts.append(peaks)

        if not peak_counts:
            return 0.0

        return float(np.max(peak_counts))

    def _check_grid_lines(self, gray, direction="horizontal") -> float:
        """Detect screen pixel grid structure.

        Screen images captured by camera show subtle horizontal/vertical
        line artifacts from the screen's pixel arrangement.
        """
        if direction == "horizontal":
            projection = np.mean(gray.astype(np.float32), axis=1)
        else:
            projection = np.mean(gray.astype(np.float32), axis=0)

        if len(projection) < 16:
            return 0.0

        from_mean = projection - np.mean(projection)

        fft_1d = np.abs(np.fft.rfft(from_mean))
        if len(fft_1d) < 4:
            return 0.0

        mid_start = len(fft_1d) // 4
        mid_end = 3 * len(fft_1d) // 4
        mid_band = fft_1d[mid_start:mid_end]

        if mid_band.size == 0 or np.mean(mid_band) < 1e-6:
            return 0.0

        ratio = float(np.max(mid_band) / np.mean(mid_band))
        return min(1.0, max(0.0, (ratio - 1.5) / 3.0))


# ── Singleton ───────────────────────────────────────────────

_moire_detector = None


def get_moire_detector() -> MoireDetector:
    """Get or create the singleton MoireDetector instance."""
    global _moire_detector
    if _moire_detector is None:
        _moire_detector = MoireDetector()
        logger.info("MoireDetector initialized")
    return _moire_detector
