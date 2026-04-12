"""
dev/detect-v2.py — Live Face Detection & Recognition (V2 Anti-Spoof)
=====================================================================
Giữ nguyên toàn bộ tính năng detect.py gốc, bổ sung 2 lớp chống giả mạo
hoạt động hoàn toàn với camera 2D thông thường:

  1. Moiré Pattern Detection (Passive) — phân tích FFT trên vùng mặt.
     Camera 2D quay màn hình LCD/OLED luôn tạo vân moiré (lưới pixel xung đột).
     Mặt thật không có vân moiré. Không cần depth camera.

  2. Challenge-Response (Active) — khi nghi ngờ, yêu cầu user thực hiện
     hành động ngẫu nhiên (nhắm mắt, quay đầu, mỉm cười...) mà video ghi sẵn
     không thể đáp ứng.

Cách dùng:
    cd face-reg-finnal-project
    python dev/detect-v2.py

Controls:
    Q  — Quit
    S  — Screenshot (saved to dev/screenshots/)
    R  — Reload embeddings cache
    L  — Toggle landmarks display
    I  — Toggle info panel
    M  — Toggle moiré overlay (NEW)
"""

import sys
import os
import time
import random
import cv2
import numpy as np

# ── Path setup ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.face_engine import get_engine
from core.liveness import get_liveness

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python as mp_python
import config

# ── Config ──────────────────────────────────────────────────
CAM_INDEX   = 0
FRAME_W     = 1280
FRAME_H     = 720
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL  = cv2.FONT_HERSHEY_PLAIN
DETECT_FPS  = 10          # max face-detect calls per second (throttle)
SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
INFO_PANEL_W = 360        # width of the right-side info panel

# ── Colors (BGR) ────────────────────────────────────────────
C_PRESENT  = ( 80, 220,  80)   # green  — known face
C_UNKNOWN  = ( 40, 160, 255)   # amber  — face detected, no match
C_SPOOF    = ( 60,  60, 220)   # red    — spoof
C_WHITE    = (240, 240, 240)
C_BLACK    = ( 10,  10,  10)
C_GOLD     = ( 30, 200, 255)
C_DIM      = (120, 120, 120)
C_CYAN     = (220, 200,  40)   # cyan — landmarks

C_BLUE     = (220, 140,  40)   # blue — info
C_PANEL_BG = ( 30,  30,  35)   # dark panel background
C_PURPLE   = (180,  60, 200)   # purple — moiré
C_ORANGE   = ( 40, 140, 255)   # orange — challenge active

# Landmark labels for 5-point
LANDMARK_LABELS = ["L-Eye", "R-Eye", "Nose", "L-Mouth", "R-Mouth"]


# ═══════════════════════════════════════════════════════════
#  MOIRÉ PATTERN DETECTION  (Passive Anti-Spoof — camera 2D)
# ═══════════════════════════════════════════════════════════

class MoireDetector:
    """Detect moiré patterns that appear when a camera captures a screen.

    When a 2D camera records an LCD/OLED screen, the camera's sensor grid
    interferes with the screen's pixel grid, creating characteristic
    moiré interference patterns. These patterns appear as periodic
    high-frequency peaks in the frequency domain (FFT).

    Real faces have smooth, organic frequency spectra.
    Screen-displayed faces show sharp periodic peaks from moiré.

    100% compatible with standard 2D webcams — no depth camera needed.
    """

    # Analysis size (square crop for consistent FFT)
    ANALYZE_SIZE = 128

    # Frequency band for moiré detection (normalized 0-1 of spectrum)
    # Moiré typically appears in mid-to-high frequency range
    FREQ_LOW  = 0.15   # ignore DC + very low freq (face structure)
    FREQ_HIGH = 0.85   # ignore very high freq (noise)

    # Thresholds
    PEAK_RATIO_THRESHOLD = 2.8    # ratio of peak to mean in target band
    ENERGY_RATIO_THRESHOLD = 0.35 # high-freq energy ratio threshold
    PERIODIC_THRESHOLD = 3.5      # periodicity score threshold

    def __init__(self):
        self.history = []
        self.max_history = 20
        self.last_result = {}

    def analyze(self, face_roi: np.ndarray) -> dict:
        """Analyze a face ROI for moiré patterns using FFT.

        Args:
            face_roi: BGR face crop from the frame

        Returns:
            dict with moire_score (0=screen, 1=real), is_screen, details
        """
        if face_roi is None or face_roi.size == 0:
            return {"moire_score": 0.5, "is_screen": None,
                    "peak_ratio": 0, "energy_ratio": 0, "periodicity": 0}

        # Convert to grayscale and resize to fixed size
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.ANALYZE_SIZE, self.ANALYZE_SIZE))

        # Normalize to float
        gray_f = gray.astype(np.float32) / 255.0

        # Apply window function to reduce spectral leakage
        window = np.outer(np.hanning(self.ANALYZE_SIZE), np.hanning(self.ANALYZE_SIZE))
        windowed = gray_f * window

        # 2D FFT
        fft = np.fft.fft2(windowed)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        # Log magnitude for better dynamic range
        log_mag = np.log1p(magnitude)

        # ── Signal 1: Peak-to-Mean ratio in target frequency band ──
        h, w = log_mag.shape
        cy, cx = h // 2, w // 2

        # Create radial mask for target frequency band
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_r = min(cx, cy)
        r_low = int(max_r * self.FREQ_LOW)
        r_high = int(max_r * self.FREQ_HIGH)

        band_mask = (dist >= r_low) & (dist <= r_high)
        band_values = log_mag[band_mask]

        if band_values.size == 0 or np.mean(band_values) < 1e-6:
            return {"moire_score": 0.5, "is_screen": None,
                    "peak_ratio": 0, "energy_ratio": 0, "periodicity": 0}

        peak_ratio = float(np.max(band_values) / np.mean(band_values))

        # ── Signal 2: High-frequency energy ratio ──────────────
        # Screen images have more structured high-freq energy (pixel grid)
        total_energy = np.sum(log_mag[dist > 0])
        if total_energy < 1e-6:
            total_energy = 1e-6
        high_freq_energy = np.sum(log_mag[dist >= r_low])
        energy_ratio = float(high_freq_energy / total_energy)

        # ── Signal 3: Periodicity detection ────────────────────
        # Moiré creates periodic peaks. Analyze angular spectrum for regularity.
        # Sample along radial lines and check for periodic peaks
        periodicity = self._check_periodicity(log_mag, cx, cy, r_low, r_high)

        # ── Signal 4: Horizontal/Vertical line detection ───────
        # Screen pixel grids create horizontal and vertical line artifacts
        h_line_score = self._check_grid_lines(gray, direction="horizontal")
        v_line_score = self._check_grid_lines(gray, direction="vertical")
        grid_score = (h_line_score + v_line_score) / 2.0

        # ── Combined score ─────────────────────────────────────
        # Higher peak_ratio, periodicity, grid = more likely screen
        screen_evidence = 0.0

        # Peak ratio: moiré creates sharp peaks
        if peak_ratio > self.PEAK_RATIO_THRESHOLD:
            screen_evidence += 0.30
        elif peak_ratio > self.PEAK_RATIO_THRESHOLD * 0.7:
            screen_evidence += 0.15

        # Periodicity: moiré is periodic
        if periodicity > self.PERIODIC_THRESHOLD:
            screen_evidence += 0.30
        elif periodicity > self.PERIODIC_THRESHOLD * 0.6:
            screen_evidence += 0.15

        # Grid lines: screen pixel structure
        if grid_score > 0.5:
            screen_evidence += 0.25
        elif grid_score > 0.3:
            screen_evidence += 0.10

        # Energy distribution anomaly
        if energy_ratio > self.ENERGY_RATIO_THRESHOLD:
            screen_evidence += 0.15

        # Convert: 0 = definitely screen, 1 = definitely real
        moire_score = max(0.0, min(1.0, 1.0 - screen_evidence))

        # Rolling average for stability
        self.history.append(moire_score)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        avg_score = float(np.mean(self.history))

        is_screen = avg_score < 0.45

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

    def _check_periodicity(self, log_mag, cx, cy, r_low, r_high) -> float:
        """Check for periodic peaks in the frequency spectrum.

        Moiré creates evenly-spaced peaks along certain angles.
        """
        n_angles = 36  # sample every 10 degrees
        peak_counts = []

        for i in range(n_angles):
            angle = i * np.pi / n_angles
            # Sample along this radial line
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

        # High max peak count along any angle indicates periodic moiré
        return float(np.max(peak_counts))

    def _check_grid_lines(self, gray, direction="horizontal") -> float:
        """Detect screen pixel grid structure.

        Screen images captured by camera show subtle horizontal/vertical
        line artifacts from the screen's pixel arrangement.
        """
        if direction == "horizontal":
            # Sum along rows, look for periodic pattern in column projection
            projection = np.mean(gray.astype(np.float32), axis=1)
        else:
            projection = np.mean(gray.astype(np.float32), axis=0)

        if len(projection) < 16:
            return 0.0

        # High-pass filter: remove DC and low-frequency components
        from_mean = projection - np.mean(projection)

        # FFT of 1D projection
        fft_1d = np.abs(np.fft.rfft(from_mean))
        if len(fft_1d) < 4:
            return 0.0

        # Skip DC component, look for peaks in mid-frequency
        mid_start = len(fft_1d) // 4
        mid_end = 3 * len(fft_1d) // 4
        mid_band = fft_1d[mid_start:mid_end]

        if mid_band.size == 0 or np.mean(mid_band) < 1e-6:
            return 0.0

        # Ratio of max peak to mean — high ratio = periodic grid
        ratio = float(np.max(mid_band) / np.mean(mid_band))

        # Normalize to 0-1 score (ratio > 4 = strong grid)
        return min(1.0, max(0.0, (ratio - 1.5) / 3.0))


# ═══════════════════════════════════════════════════════════
#  CHALLENGE-RESPONSE  (Active Anti-Spoof Layer)
# ═══════════════════════════════════════════════════════════

# MediaPipe eye landmark indices for EAR
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]


class ChallengeResponse:
    """Active liveness: ask user to perform random actions.

    Challenges:
    - BLINK:      Close both eyes briefly
    - BLINK_LEFT: Close only left eye (wink)
    - TURN_RIGHT: Turn head to the right
    - TURN_LEFT:  Turn head to the left
    - NOD_UP:     Look up
    - SMILE:      Smile (mouth width increases)
    """

    CHALLENGES = [
        ("BLINK",      "Hay CHOP MAT"),
        ("BLINK_LEFT", "Hay NHAM MAT TRAI"),
        ("TURN_RIGHT", "Hay QUAY DAU SANG PHAI"),
        ("TURN_LEFT",  "Hay QUAY DAU SANG TRAI"),
        ("NOD_UP",     "Hay NGUOC LEN"),
        ("SMILE",      "Hay MIM CUOI"),
    ]

    # Timing
    CHALLENGE_TIMEOUT = 5.0    # seconds to complete a challenge
    COOLDOWN_AFTER_PASS = 15.0 # seconds before re-challenging after pass

    # Thresholds
    EAR_BLINK = 0.20          # EAR below this = eye closed
    TURN_THRESHOLD = 0.06     # nose X displacement for turn
    NOD_THRESHOLD = 0.04      # nose Y displacement for nod up
    SMILE_THRESHOLD = 1.3     # mouth width increase ratio

    def __init__(self):
        self.active = False
        self.current_challenge = None
        self.challenge_text = ""
        self.challenge_start = 0.0
        self.last_pass_time = 0.0
        self.passed = False
        self.failed = False
        self.fail_count = 0

        # Baseline measurements
        self._baseline_nose_x = None
        self._baseline_nose_y = None
        self._baseline_mouth_w = None
        self._baseline_ear_left = None
        self._baseline_ear_right = None
        self._baseline_frames = 0

    def should_trigger(self, moire_result: dict) -> bool:
        """Decide if we should trigger a challenge based on moiré analysis."""
        now = time.time()

        # Already passed recently → no challenge
        if self.passed and (now - self.last_pass_time) < self.COOLDOWN_AFTER_PASS:
            return False

        # Already active
        if self.active:
            return False

        # Moiré suggests screen → trigger immediately
        if moire_result.get("is_screen") is True:
            return True

        # Moiré score borderline (0.45 - 0.6) → trigger with some probability
        score = moire_result.get("moire_score", 1.0)
        if score < 0.6 and random.random() < 0.02:  # ~2% per frame at borderline
            return True

        return False

    def start_challenge(self):
        """Pick a random challenge and start the timer."""
        self.current_challenge, self.challenge_text = random.choice(self.CHALLENGES)
        self.active = True
        self.challenge_start = time.time()
        self.passed = False
        self.failed = False
        self._baseline_nose_x = None
        self._baseline_nose_y = None
        self._baseline_mouth_w = None
        self._baseline_ear_left = None
        self._baseline_ear_right = None
        self._baseline_frames = 0
        print(f"[CHALLENGE] {self.challenge_text} (type: {self.current_challenge})")

    def update(self, face_landmarks, img_w, img_h) -> str:
        """Process a frame during an active challenge.

        Returns: "checking" | "passed" | "failed" | "timeout" | "inactive"
        """
        if not self.active:
            return "inactive"

        now = time.time()
        elapsed = now - self.challenge_start

        # Timeout
        if elapsed > self.CHALLENGE_TIMEOUT:
            self.active = False
            self.failed = True
            self.fail_count += 1
            print(f"[CHALLENGE] TIMEOUT — failed ({self.fail_count} fails)")
            return "timeout"

        if face_landmarks is None or len(face_landmarks) < 468:
            return "checking"

        # Extract measurements
        nose = face_landmarks[1]
        nose_x, nose_y = nose.x, nose.y

        left_ear = self._calc_ear_from_lm(face_landmarks, LEFT_EYE_IDX, img_w, img_h)
        right_ear = self._calc_ear_from_lm(face_landmarks, RIGHT_EYE_IDX, img_w, img_h)

        mouth_left = face_landmarks[61]
        mouth_right = face_landmarks[291]
        mouth_w = abs(mouth_right.x - mouth_left.x)

        # Capture baseline from first few frames
        if self._baseline_frames < 5:
            if self._baseline_nose_x is None:
                self._baseline_nose_x = nose_x
                self._baseline_nose_y = nose_y
                self._baseline_mouth_w = mouth_w
                self._baseline_ear_left = left_ear
                self._baseline_ear_right = right_ear
            else:
                a = 0.7
                self._baseline_nose_x = a * self._baseline_nose_x + (1-a) * nose_x
                self._baseline_nose_y = a * self._baseline_nose_y + (1-a) * nose_y
                self._baseline_mouth_w = a * self._baseline_mouth_w + (1-a) * mouth_w
                self._baseline_ear_left = a * self._baseline_ear_left + (1-a) * left_ear
                self._baseline_ear_right = a * self._baseline_ear_right + (1-a) * right_ear
            self._baseline_frames += 1
            return "checking"

        # Check completion based on challenge type
        completed = False

        if self.current_challenge == "BLINK":
            if left_ear < self.EAR_BLINK and right_ear < self.EAR_BLINK:
                completed = True

        elif self.current_challenge == "BLINK_LEFT":
            if left_ear < self.EAR_BLINK and right_ear > self.EAR_BLINK + 0.03:
                completed = True

        elif self.current_challenge == "TURN_RIGHT":
            dx = self._baseline_nose_x - nose_x
            if dx > self.TURN_THRESHOLD:
                completed = True

        elif self.current_challenge == "TURN_LEFT":
            dx = nose_x - self._baseline_nose_x
            if dx > self.TURN_THRESHOLD:
                completed = True

        elif self.current_challenge == "NOD_UP":
            dy = self._baseline_nose_y - nose_y
            if dy > self.NOD_THRESHOLD:
                completed = True

        elif self.current_challenge == "SMILE":
            if self._baseline_mouth_w > 0.001:
                ratio = mouth_w / self._baseline_mouth_w
                if ratio > self.SMILE_THRESHOLD:
                    completed = True

        if completed:
            self.active = False
            self.passed = True
            self.last_pass_time = time.time()
            self.fail_count = 0
            print(f"[CHALLENGE] PASSED ({self.current_challenge})")
            return "passed"

        return "checking"

    def _calc_ear_from_lm(self, landmarks, indices, w, h):
        """Calculate Eye Aspect Ratio from MediaPipe landmarks."""
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]

        def dist(a, b):
            return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

        vert1 = dist(pts[1], pts[5])
        vert2 = dist(pts[2], pts[4])
        horiz = dist(pts[0], pts[3])

        if horiz < 1e-6:
            return 0.3
        return (vert1 + vert2) / (2.0 * horiz)

    @property
    def status_text(self) -> str:
        if self.active:
            remaining = max(0, self.CHALLENGE_TIMEOUT - (time.time() - self.challenge_start))
            return f"{self.challenge_text} ({remaining:.1f}s)"
        if self.passed:
            cooldown = self.COOLDOWN_AFTER_PASS - (time.time() - self.last_pass_time)
            if cooldown > 0:
                return f"VERIFIED ({cooldown:.0f}s)"
            return ""
        if self.failed:
            return f"FAILED (fails: {self.fail_count})"
        return ""


# ═══════════════════════════════════════════════════════════
#  DRAWING HELPERS  (same as detect.py + new overlays)
# ═══════════════════════════════════════════════════════════

def draw_rounded_rect(img, pt1, pt2, color, thickness=2, r=12):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(r, abs(x2 - x1) // 2, abs(y2 - y1) // 2)
    if r < 1:
        cv2.rectangle(img, pt1, pt2, color, thickness, cv2.LINE_AA)
        return
    cv2.line(img,  (x1+r, y1),   (x2-r, y1),   color, thickness, cv2.LINE_AA)
    cv2.line(img,  (x1+r, y2),   (x2-r, y2),   color, thickness, cv2.LINE_AA)
    cv2.line(img,  (x1,   y1+r), (x1,   y2-r), color, thickness, cv2.LINE_AA)
    cv2.line(img,  (x2,   y1+r), (x2,   y2-r), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1+r, y1+r), (r, r), 180,  0, 90,  color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2-r, y1+r), (r, r), 270,  0, 90,  color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1+r, y2-r), (r, r),  90,  0, 90,  color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2-r, y2-r), (r, r),   0,  0, 90,  color, thickness, cv2.LINE_AA)


def draw_label_box(img, text, sub_text, extra_text, x1, y1, color):
    """Draw a filled label box above the bounding box with up to 3 lines."""
    line_count = 1 + (1 if sub_text else 0) + (1 if extra_text else 0)
    label_h = 20 * line_count + 8
    lx1, ly1 = x1, max(0, y1 - label_h)
    lx2, ly2 = x1 + max(220, len(text) * 13 + 20), y1

    overlay = img.copy()
    cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), color, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    y_off = ly1 + 16
    cv2.putText(img, text, (lx1 + 8, y_off), FONT, 0.55, C_WHITE, 2, cv2.LINE_AA)
    if sub_text:
        y_off += 18
        cv2.putText(img, sub_text, (lx1 + 8, y_off), FONT, 0.44, C_WHITE, 1, cv2.LINE_AA)
    if extra_text:
        y_off += 18
        cv2.putText(img, extra_text, (lx1 + 8, y_off), FONT, 0.44, C_GOLD, 1, cv2.LINE_AA)


def draw_landmarks(img, landmarks, color=C_CYAN):
    """Draw 5-point face landmarks with labels."""
    if landmarks is None or len(landmarks) < 5:
        return
    for i, (x, y) in enumerate(landmarks[:5]):
        px, py = int(x), int(y)
        cv2.circle(img, (px, py), 5, color, 2, cv2.LINE_AA)
        cv2.circle(img, (px, py), 2, C_WHITE, -1, cv2.LINE_AA)
        label = LANDMARK_LABELS[i] if i < len(LANDMARK_LABELS) else str(i)
        cv2.putText(img, label, (px + 7, py - 4), FONT, 0.33, color, 1, cv2.LINE_AA)


def draw_hud(img, fps, face_count, total_students, show_landmarks, show_info, show_moire):
    """Top-left HUD: FPS, face count, DB size, toggle states."""
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), C_BLACK, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    items = [
        (f"FPS {fps:5.1f}", C_GOLD),
        (f"Faces: {face_count}", C_PRESENT if face_count > 0 else C_DIM),
        (f"DB: {total_students}", C_WHITE),
        (f"[L]mk:{'ON' if show_landmarks else 'OFF'}", C_CYAN if show_landmarks else C_DIM),
        (f"[I]nfo:{'ON' if show_info else 'OFF'}", C_BLUE if show_info else C_DIM),
        (f"[M]oire:{'ON' if show_moire else 'OFF'}", C_PURPLE if show_moire else C_DIM),
        ("Q=Quit S=Shot R=Reload", C_DIM),
    ]
    x = 14
    for txt, color in items:
        cv2.putText(img, txt, (x, 28), FONT, 0.42, color, 1, cv2.LINE_AA)
        x += len(txt) * 8 + 14


def conf_bar(img, x1, y2, x2, confidence, color):
    """Draw a small confidence bar below the bounding box."""
    bar_w = x2 - x1
    bar_h = 6
    filled = int(bar_w * min(1.0, confidence))
    cv2.rectangle(img, (x1, y2 + 4), (x2, y2 + 4 + bar_h), C_BLACK, -1)
    cv2.rectangle(img, (x1, y2 + 4), (x1 + filled, y2 + 4 + bar_h), color, -1)


def draw_challenge_overlay(img, challenge: ChallengeResponse):
    """Draw a prominent challenge banner across the frame."""
    if not challenge.active and not challenge.passed and not challenge.failed:
        return

    h, w = img.shape[:2]
    text = challenge.status_text
    if not text:
        return

    if challenge.active:
        # Large pulsing banner
        overlay = img.copy()
        alpha = float(0.55 + 0.15 * np.sin(time.time() * 6))  # pulsing

        # Top banner
        cv2.rectangle(overlay, (0, 44), (w, 110), (20, 20, 60), -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Challenge text (centered)
        text_size = cv2.getTextSize(f">> {challenge.challenge_text} <<", FONT, 0.85, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(img, f">> {challenge.challenge_text} <<",
                    (text_x, 72), FONT, 0.85, C_ORANGE, 2, cv2.LINE_AA)

        remaining = max(0, challenge.CHALLENGE_TIMEOUT - (time.time() - challenge.challenge_start))
        cv2.putText(img, f"[{remaining:.1f}s]",
                    (text_x, 100), FONT, 0.55, C_GOLD, 1, cv2.LINE_AA)

        # Progress bar
        progress = min(1.0, (time.time() - challenge.challenge_start) / challenge.CHALLENGE_TIMEOUT)
        bar_w_px = w - 40
        cv2.rectangle(img, (20, 108), (20 + bar_w_px, 112), C_DIM, -1)
        cv2.rectangle(img, (20, 108), (20 + int(bar_w_px * progress), 112), C_ORANGE, -1)

    elif challenge.passed:
        cooldown = challenge.COOLDOWN_AFTER_PASS - (time.time() - challenge.last_pass_time)
        if cooldown > 0:
            cv2.putText(img, f"VERIFIED (challenge passed) [{cooldown:.0f}s]",
                        (20, 66), FONT, 0.5, C_PRESENT, 1, cv2.LINE_AA)

    elif challenge.failed:
        cv2.putText(img, f"CHALLENGE FAILED x{challenge.fail_count}",
                    (20, 66), FONT, 0.5, C_SPOOF, 1, cv2.LINE_AA)


def draw_moire_badge(img, x2, y2, moire_result):
    """Draw a small moiré indicator badge near the bounding box corner."""
    if moire_result is None:
        return

    score = moire_result.get("moire_score", 1.0)
    is_screen = moire_result.get("is_screen")

    if is_screen is True:
        badge_color = C_SPOOF
        txt = "SCREEN"
    elif is_screen is False:
        badge_color = C_PRESENT
        txt = "REAL"
    else:
        badge_color = C_GOLD
        txt = "?"

    # Badge background
    bx, by = x2 - 70, y2 + 14
    cv2.rectangle(img, (bx, by), (bx + 68, by + 18), badge_color, -1)
    cv2.putText(img, f"{txt} {score:.0%}", (bx + 3, by + 13), FONT, 0.35, C_WHITE, 1, cv2.LINE_AA)


def draw_moire_spectrum(img, face_roi, x1, y1):
    """Draw a small FFT magnitude spectrum thumbnail near the face (debug view)."""
    if face_roi is None or face_roi.size == 0:
        return

    try:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0
        window = np.outer(np.hanning(64), np.hanning(64))
        fft = np.fft.fft2(gray * window)
        mag = np.log1p(np.abs(np.fft.fftshift(fft)))
        # Normalize to 0-255 for display
        mag = (mag / mag.max() * 255).astype(np.uint8) if mag.max() > 0 else np.zeros((64,64), np.uint8)
        mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
        # Place at top-left of bounding box
        sx, sy = max(0, x1 - 70), max(0, y1)
        if sy + 64 <= img.shape[0] and sx + 64 <= img.shape[1]:
            img[sy:sy+64, sx:sx+64] = mag_color
            cv2.rectangle(img, (sx, sy), (sx+64, sy+64), C_PURPLE, 1)
            cv2.putText(img, "FFT", (sx+2, sy+12), FONT, 0.35, C_WHITE, 1, cv2.LINE_AA)
    except Exception:
        pass


def draw_info_panel(img, results, moire_results, challenge, panel_w=INFO_PANEL_W):
    """Draw right-side info panel — original + moiré/challenge info."""
    h, w = img.shape[:2]
    panel = np.full((h, panel_w, 3), C_PANEL_BG, dtype=np.uint8)

    # Title
    cv2.putText(panel, "FACE RECOGNITION V2", (12, 30), FONT, 0.6, C_GOLD, 2, cv2.LINE_AA)
    cv2.line(panel, (12, 40), (panel_w - 12, 40), C_DIM, 1, cv2.LINE_AA)

    y = 60

    # ── Challenge status section ────────────────────────
    if challenge.active or challenge.passed or challenge.failed:
        cv2.putText(panel, "ANTI-SPOOF V2", (12, y), FONT, 0.45, C_ORANGE, 1, cv2.LINE_AA)
        y += 20

        status_txt = challenge.status_text
        if challenge.active:
            s_color = C_ORANGE
        elif challenge.passed:
            s_color = C_PRESENT
        else:
            s_color = C_SPOOF

        max_chars = panel_w // 8
        while status_txt:
            chunk = status_txt[:max_chars]
            status_txt = status_txt[max_chars:]
            cv2.putText(panel, chunk, (12, y), FONT, 0.4, s_color, 1, cv2.LINE_AA)
            y += 16
        y += 6
        cv2.line(panel, (12, y), (panel_w - 12, y), (50, 50, 55), 1)
        y += 14

    if not results:
        cv2.putText(panel, "No faces detected", (12, y), FONT, 0.45, C_DIM, 1, cv2.LINE_AA)
    else:
        for i, r in enumerate(results):
            if y > h - 40:
                break

            color = C_PRESENT if r["status"] == "present" else (C_SPOOF if r["status"] == "spoof" else C_UNKNOWN)

            # Face index header
            cv2.putText(panel, f"--- Face #{i+1} ---", (12, y), FONT, 0.45, color, 1, cv2.LINE_AA)
            y += 22

            # Aligned face thumbnail
            if r.get("aligned") is not None:
                thumb = cv2.resize(r["aligned"], (64, 64))
                cv2.rectangle(panel, (12, y), (78, y + 66), color, 2)
                panel[y+1:y+65, 13:77] = thumb
                tx = 88
                cv2.putText(panel, r["name"], (tx, y + 18), FONT, 0.5, C_WHITE, 1, cv2.LINE_AA)
                status_txt = r["status"].upper()
                cv2.putText(panel, status_txt, (tx, y + 36), FONT, 0.42, color, 1, cv2.LINE_AA)
                if r["sid"]:
                    cv2.putText(panel, f"ID: {r['sid']}", (tx, y + 54), FONT, 0.4, C_DIM, 1, cv2.LINE_AA)
                y += 72
            else:
                cv2.putText(panel, f"Name: {r['name']}", (12, y), FONT, 0.42, C_WHITE, 1, cv2.LINE_AA)
                y += 18

            # Confidence
            conf_pct = r["conf"] * 100
            conf_color = C_PRESENT if conf_pct >= 50 else C_UNKNOWN
            cv2.putText(panel, f"Confidence: {conf_pct:.1f}%", (12, y), FONT, 0.42, conf_color, 1, cv2.LINE_AA)
            bar_x = 180
            bar_w = panel_w - bar_x - 16
            filled = int(bar_w * min(1.0, r["conf"]))
            cv2.rectangle(panel, (bar_x, y - 10), (bar_x + bar_w, y), C_BLACK, -1)
            cv2.rectangle(panel, (bar_x, y - 10), (bar_x + filled, y), conf_color, -1)
            y += 20

            # Detection confidence
            if r.get("det_conf") is not None:
                cv2.putText(panel, f"Det Score: {r['det_conf']:.2f}", (12, y), FONT, 0.42, C_DIM, 1, cv2.LINE_AA)
                y += 18

            # ── Moiré info (NEW) ────────────────────────
            mr = moire_results.get(i)
            if mr:
                m_color = C_PRESENT if not mr.get("is_screen") else C_SPOOF
                cv2.putText(panel, f"Moire Score: {mr['moire_score']:.0%}", (12, y), FONT, 0.42, m_color, 1, cv2.LINE_AA)
                bar_x2 = 180
                bar_w2 = panel_w - bar_x2 - 16
                filled2 = int(bar_w2 * min(1.0, mr["moire_score"]))
                cv2.rectangle(panel, (bar_x2, y - 10), (bar_x2 + bar_w2, y), C_BLACK, -1)
                cv2.rectangle(panel, (bar_x2, y - 10), (bar_x2 + filled2, y), m_color, -1)
                y += 18
                screen_txt = "SCREEN detected!" if mr.get("is_screen") else "REAL face"
                cv2.putText(panel, f"  {screen_txt}", (12, y), FONT, 0.38, m_color, 1, cv2.LINE_AA)
                y += 16
                cv2.putText(panel, f"  Peak:{mr.get('peak_ratio', 0):.1f} Period:{mr.get('periodicity', 0):.1f} Grid:{mr.get('grid_score', 0):.2f}",
                            (12, y), FONT, 0.32, C_DIM, 1, cv2.LINE_AA)
                y += 16

            # Quality
            if r.get("quality"):
                q = r["quality"]
                cv2.putText(panel, f"Blur: {q['blur']:.0f}  Bright: {q['bright']:.0f}", (12, y), FONT, 0.38, C_DIM, 1, cv2.LINE_AA)
                y += 16
                cv2.putText(panel, f"Yaw: {q['yaw']:.1f} deg  Size: {q['size']}px", (12, y), FONT, 0.38, C_DIM, 1, cv2.LINE_AA)
                y += 16
                q_status = "PASS" if q["passed"] else "FAIL"
                q_color = C_PRESENT if q["passed"] else C_SPOOF
                cv2.putText(panel, f"Quality: {q_status}", (12, y), FONT, 0.42, q_color, 1, cv2.LINE_AA)
                y += 20

            # Liveness info (multi-frame — original)
            if r.get("liveness"):
                lv = r["liveness"]
                if lv["is_live"] is None:
                    lv_color = C_GOLD
                    lv_status = "CHECKING"
                elif lv["is_live"]:
                    lv_color = C_PRESENT
                    lv_status = "LIVE"
                else:
                    lv_color = C_SPOOF
                    lv_status = "SPOOF"
                cv2.putText(panel, f"Liveness: {lv_status}", (12, y), FONT, 0.42, lv_color, 1, cv2.LINE_AA)
                y += 18
                cv2.putText(panel, f"  Blinks: {lv.get('blinks', 0)}  EAR: {lv.get('ear', 0):.2f}", (12, y), FONT, 0.36, C_DIM, 1, cv2.LINE_AA)
                y += 16
                cv2.putText(panel, f"  Movement: {lv.get('movement', 0):.1f}px", (12, y), FONT, 0.36, C_DIM, 1, cv2.LINE_AA)
                y += 16
                cv2.putText(panel, f"  Track: {lv.get('track_time', 0):.1f}s", (12, y), FONT, 0.36, C_DIM, 1, cv2.LINE_AA)
                y += 16
                if lv_status == "SPOOF" and lv.get("reason"):
                    cv2.putText(panel, f"  Reason: {lv['reason']}", (12, y), FONT, 0.36, C_SPOOF, 1, cv2.LINE_AA)
                    y += 16

            # Embedding info
            if r.get("emb_dim"):
                cv2.putText(panel, f"Embedding: {r['emb_dim']}D", (12, y), FONT, 0.38, C_DIM, 1, cv2.LINE_AA)
                y += 18

            # Separator
            cv2.line(panel, (12, y), (panel_w - 12, y), (50, 50, 55), 1)
            y += 14

    combined = np.hstack([img, panel])
    return combined


# ═══════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════

def main():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Face Detection V2 — Moire + Challenge Anti-Spoof")
    print("  Works with any standard 2D webcam")
    print("=" * 60)

    print("\nLoading face engine...")
    engine = get_engine()

    from core.database import get_db
    db = get_db()

    # ── MediaPipe FaceLandmarker (for challenge-response) ───
    face_landmarker_model = str(config.MODELS_DIR / "face_landmarker.task")
    mesh_landmarker = None

    if os.path.exists(face_landmarker_model):
        base_options = mp_python.BaseOptions(model_asset_path=face_landmarker_model)
        mesh_options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=5,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        mesh_landmarker = vision.FaceLandmarker.create_from_options(mesh_options)
        print("FaceLandmarker loaded (challenge-response ready)")
    else:
        print(f"[WARN] FaceLandmarker not found: {face_landmarker_model}")
        print("[WARN] Challenge-response anti-spoof DISABLED (moire still active).")
        print("[WARN] Download: https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

    # ── Original multi-frame liveness (EAR + movement) ──────
    liveness_tracker = None
    try:
        liveness_tracker = get_liveness()
        print("Multi-frame liveness ready (EAR blink + head movement)")
    except FileNotFoundError as e:
        print(f"[WARN] Multi-frame liveness disabled: {e}")

    # ── V2 modules ──────────────────────────────────────────
    moire_detector = MoireDetector()
    challenge = ChallengeResponse()

    print("\nOpening camera...")
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera (index {CAM_INDEX})")
        sys.exit(1)

    print("\nLive detection V2 running. Press Q to quit.")
    print("Controls: L=Landmarks | I=Info | M=Moire | S=Screenshot | R=Reload | Q=Quit\n")

    # State
    last_results = []
    last_raw_faces = []
    last_detect_time = 0.0
    detect_interval  = 1.0 / DETECT_FPS

    # V2 state
    moire_results = {}            # per face index
    current_mesh_landmarks = []   # latest FaceMesh results
    mesh_ts = 0                   # monotonic timestamp for MediaPipe

    # Toggle states
    show_landmarks = True
    show_info_panel = True
    show_moire_overlay = True

    # FPS tracking
    fps_counter = 0
    fps_start   = time.time()
    display_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed — retrying...")
            time.sleep(0.05)
            continue

        now = time.time()
        h_frame, w_frame = frame.shape[:2]

        # ── Multi-frame liveness tracking (EVERY frame) ─────
        if liveness_tracker is not None:
            liveness_tracker.process_frame(frame)

        # ── MediaPipe FaceMesh (EVERY frame for challenge) ──
        if mesh_landmarker is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mesh_ts += 33
            try:
                mesh_result = mesh_landmarker.detect_for_video(mp_image, mesh_ts)
                current_mesh_landmarks = mesh_result.face_landmarks if mesh_result.face_landmarks else []
            except Exception as e:
                print(f"[WARN] FaceMesh error: {e}")
                current_mesh_landmarks = []

            # ── Challenge-Response logic ────────────────
            if current_mesh_landmarks:
                primary_face_lm = current_mesh_landmarks[0]
                primary_moire = moire_results.get(0, {})

                # Update active challenge
                if challenge.active:
                    challenge.update(primary_face_lm, w_frame, h_frame)

                # Check if we should trigger a new challenge
                elif challenge.should_trigger(primary_moire):
                    challenge.start_challenge()

        # ── Face detection (throttled) ──────────────────────
        if now - last_detect_time >= detect_interval:
            last_detect_time = now
            try:
                engine._ensure_model()
                raw_faces = engine._app.get(frame)
                last_results = []
                last_raw_faces = raw_faces
                moire_results = {}

                for fi, raw_face in enumerate(raw_faces):
                    det_score = float(raw_face.det_score)
                    if det_score < 0.5:
                        continue

                    x1, y1, x2, y2 = [int(v) for v in raw_face.bbox]
                    w_face, h_face = x2 - x1, y2 - y1
                    if w_face < 30 or h_face < 30:
                        continue

                    kps = raw_face.kps if hasattr(raw_face, 'kps') else None
                    aligned = engine._align_face(frame, raw_face)

                    emb = (raw_face.normed_embedding
                           if hasattr(raw_face, 'normed_embedding')
                           and raw_face.normed_embedding is not None
                           else np.array([]))
                    emb_dim = len(emb) if len(emb) > 0 else 0

                    # ── Moiré analysis (NEW — passive) ──────
                    face_roi = frame[max(0,y1):y2, max(0,x1):x2]
                    mr = moire_detector.analyze(face_roi)
                    moire_results[fi] = mr

                    # Quality check
                    quality_info = None
                    try:
                        from core.schemas import DetectedFace
                        det_face = DetectedFace(
                            bbox=np.array([x1, y1, x2, y2]),
                            landmarks=kps,
                            confidence=det_score,
                            aligned_face=aligned,
                            embedding=emb
                        )
                        qr = engine.quality_check(frame, det_face)
                        quality_info = {
                            "passed": qr.passed,
                            "blur": qr.blur_score,
                            "bright": qr.brightness,
                            "yaw": qr.yaw_angle,
                            "size": qr.face_size,
                            "reasons": qr.reasons,
                        }
                    except Exception:
                        pass

                    # Multi-frame liveness
                    if liveness_tracker is not None:
                        lv = liveness_tracker.get_liveness((x1, y1, x2, y2))
                        liveness_info = {
                            "is_live": lv.is_live,
                            "score": lv.score,
                            "reason": lv.reason,
                            "blinks": lv.blinks,
                            "ear": lv.ear,
                            "movement": lv.movement,
                            "track_time": lv.track_time,
                        }
                    else:
                        lv = None
                        liveness_info = {
                            "is_live": True, "score": 1.0, "reason": "liveness_disabled",
                            "blinks": 0, "ear": 0.0, "movement": 0.0, "track_time": 0.0,
                        }

                    entry = {
                        "bbox":     (x1, y1, x2, y2),
                        "name":     "Unknown",
                        "sid":      "",
                        "conf":     0.0,
                        "det_conf": det_score,
                        "status":   "unknown",
                        "label":    "No match",
                        "kps":      kps,
                        "aligned":  aligned,
                        "emb_dim":  emb_dim,
                        "quality":  quality_info,
                        "liveness": liveness_info,
                    }

                    # ── V2 Anti-Spoof Gate ──────────────────
                    # Layer 1: Moiré check (passive — always runs)
                    is_screen = mr.get("is_screen", False)

                    # Layer 2: Challenge check (active — triggered when moiré suspicious)
                    challenge_blocked = False
                    if is_screen and not challenge.passed:
                        challenge_blocked = True
                    if challenge.failed and challenge.fail_count >= 2:
                        challenge_blocked = True

                    # ── Status determination ────────────────
                    if challenge_blocked:
                        entry["status"] = "spoof"
                        entry["name"]   = "SPOOF (V2)"
                        entry["label"]  = f"Moire: {mr.get('moire_score', 0):.0%}"
                        entry["extra_label"] = f"SCREEN detected — challenge required"

                    elif lv is not None and lv.is_live is None:
                        entry["status"] = "unknown"
                        entry["name"]   = "Checking..."
                        entry["label"]  = f"Blinks: {lv.blinks} | EAR: {lv.ear:.2f}"
                        entry["extra_label"] = f"{lv.reason}"

                    elif lv is not None and not lv.is_live:
                        entry["status"] = "spoof"
                        entry["name"]   = "SPOOF"
                        entry["label"]  = f"Blinks: {lv.blinks} | Move: {lv.movement}px"
                        entry["extra_label"] = f"{lv.reason} | {lv.track_time:.0f}s"

                    else:
                        # LIVE or liveness disabled — proceed with recognition
                        if emb is not None and len(emb) > 0:
                            match = engine.match(emb)
                            if match.matched:
                                entry["name"]   = match.name
                                entry["sid"]    = match.student_id
                                entry["conf"]   = float(match.score)
                                entry["status"] = "present"
                                entry["label"]  = f"ID: {match.student_id}  Conf: {match.score:.0%}"

                        blink_info = f"Blinks: {lv.blinks} | {lv.track_time:.0f}s" if lv else ""
                        moire_info = f"M:{mr.get('moire_score', 0):.0%}"
                        chall_info = "V2:OK" if challenge.passed else ""
                        entry["extra_label"] = f"LIVE | {blink_info} {moire_info} {chall_info}".strip()

                    last_results.append(entry)

            except Exception as e:
                print(f"Detect error: {e}")
                import traceback
                traceback.print_exc()

        # ── Draw cached results ─────────────────────────────
        for ri, r in enumerate(last_results):
            x1, y1, x2, y2 = r["bbox"]
            status = r["status"]
            color  = C_PRESENT if status == "present" else (C_SPOOF if status == "spoof" else C_UNKNOWN)

            draw_rounded_rect(frame, (x1, y1), (x2, y2), color, thickness=2)

            if show_landmarks:
                draw_landmarks(frame, r.get("kps"), color=C_CYAN)

            draw_label_box(frame, r["name"], r["label"], r.get("extra_label"), x1, y1, color)
            conf_bar(frame, x1, y2, x2, r["conf"], color)

            # Quality indicator dot
            if r.get("quality"):
                q_color = C_PRESENT if r["quality"]["passed"] else C_SPOOF
                cv2.circle(frame, (x2 - 8, y1 + 8), 6, q_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (x2 - 8, y1 + 8), 6, C_WHITE, 1, cv2.LINE_AA)

            # Moiré badge + spectrum (NEW)
            if show_moire_overlay:
                draw_moire_badge(frame, x2, y2, moire_results.get(ri))
                # FFT spectrum thumbnail
                face_roi = frame[max(0,y1):y2, max(0,x1):x2]
                draw_moire_spectrum(frame, face_roi, x1, y1)

        # ── Challenge overlay (NEW) ─────────────────────────
        draw_challenge_overlay(frame, challenge)

        # ── HUD ────────────────────────────────────────────
        total_students = len(db.get_all_students())
        draw_hud(frame, display_fps, len(last_results), total_students,
                 show_landmarks, show_info_panel, show_moire_overlay)

        # ── FPS calculation ─────────────────────────────────
        fps_counter += 1
        if now - fps_start >= 1.0:
            display_fps = fps_counter / (now - fps_start)
            fps_counter = 0
            fps_start   = now

        # ── Compose final frame ─────────────────────────────
        if show_info_panel:
            display_frame = draw_info_panel(frame, last_results, moire_results, challenge)
        else:
            display_frame = frame

        cv2.imshow("Live Face Detection V2 — Anti-Spoof", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit.")
            break
        elif key == ord('s'):
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOT_DIR, f"detect_v2_{ts}.jpg")
            cv2.imwrite(path, display_frame)
            print(f"Screenshot saved: {path}")
        elif key == ord('r'):
            print("Reloading face embeddings cache...")
            engine.reload_cache()
            print("Cache reloaded.")
        elif key == ord('l'):
            show_landmarks = not show_landmarks
            print(f"Landmarks: {'ON' if show_landmarks else 'OFF'}")
        elif key == ord('i'):
            show_info_panel = not show_info_panel
            print(f"Info panel: {'ON' if show_info_panel else 'OFF'}")
        elif key == ord('m'):
            show_moire_overlay = not show_moire_overlay
            print(f"Moire overlay: {'ON' if show_moire_overlay else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
