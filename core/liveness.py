"""
Multi-Frame Liveness Detection
================================
Phát hiện giả mạo khuôn mặt bằng phân tích đa frame:

1. Eye Blink Detection (EAR) — ảnh 2D không thể chớp mắt
2. Head Micro-Movement — ảnh 2D hoàn toàn tĩnh

Sử dụng MediaPipe FaceLandmarker (Tasks API) cho 478 landmark tracking.
Cần tối thiểu ~2 giây quan sát trước khi đưa ra kết luận.
"""

import os
import cv2
import numpy as np
import time
from collections import deque
from dataclasses import dataclass, field
from loguru import logger

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python as mp_python

import config

# ── MediaPipe eye landmark indices for EAR ──────────────────
# Right eye (6 points: outer corner, top-1, top-2, inner corner, bottom-2, bottom-1)
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
# Left eye
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Model path
FACE_LANDMARKER_MODEL = str(config.MODELS_DIR / "face_landmarker.task")


@dataclass
class LivenessStatus:
    """Result of multi-frame liveness analysis."""
    is_live: object       # True/False/None (None = undetermined)
    score: float          # 0.0 - 1.0
    reason: str           # human-readable reason
    blinks: int           # number of blinks detected
    ear: float            # current Eye Aspect Ratio
    movement: float       # head movement (pixels std)
    track_time: float     # seconds since first seen


@dataclass
class _FaceTrack:
    """Internal per-face tracking state."""
    ear_history: deque = field(default_factory=lambda: deque(maxlen=200))
    pos_history: deque = field(default_factory=lambda: deque(maxlen=200))
    blink_count: int = 0
    last_ear: float = 0.30
    blink_state: bool = False   # True = eye currently closed
    first_seen: float = 0.0
    last_seen: float = 0.0
    center_x: float = 0.0
    center_y: float = 0.0


class MultiFrameLiveness:
    """
    Multi-frame liveness checker using MediaPipe FaceLandmarker.

    Logic:
    - Track mắt (EAR) qua nhiều frame → phát hiện chớp mắt
    - Track vị trí mũi qua nhiều frame → phát hiện chuyển động nhỏ
    - Sau MIN_TRACK_TIME giây:
        - Có chớp mắt ≥ 1 lần → LIVE
        - Không chớp mắt + không chuyển động → SPOOF
    """

    # ── Thresholds ──────────────────────────────────────────
    EAR_BLINK_THRESHOLD = 0.21   # EAR below this = eye closed
    MIN_TRACK_TIME      = 2.0    # seconds before making judgment
    MOVEMENT_THRESHOLD  = 2.0    # pixels std to count as "moving"
    MATCH_DISTANCE      = 120    # max pixels to match face across frames
    STALE_TIMEOUT       = 3.0    # seconds before removing stale tracker

    def __init__(self):
        if not os.path.exists(FACE_LANDMARKER_MODEL):
            raise FileNotFoundError(
                f"FaceLandmarker model not found: {FACE_LANDMARKER_MODEL}\n"
                f"Download: https://storage.googleapis.com/mediapipe-models/"
                f"face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        base_options = mp_python.BaseOptions(
            model_asset_path=FACE_LANDMARKER_MODEL
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=5,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._tracks: dict[int, _FaceTrack] = {}
        self._next_id = 0
        self._frame_ts = 0  # monotonic timestamp for VIDEO mode
        logger.info("MultiFrameLiveness initialized (EAR blink + movement)")

    # ── Public API ──────────────────────────────────────────

    def process_frame(self, frame: np.ndarray):
        """Run on EVERY frame to continuously track eyes & movement.

        This runs MediaPipe FaceLandmarker and updates internal trackers.
        Call this before get_liveness().
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires increasing timestamps (in ms)
        self._frame_ts += 33  # ~30fps
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        now = time.time()

        if not result.face_landmarks:
            self._cleanup(now)
            return

        for face_lm in result.face_landmarks:
            # Nose tip (landmark 1) for position tracking
            nose = face_lm[1]
            cx, cy = nose.x * w, nose.y * h

            # Calculate EAR (average of both eyes)
            left_ear = self._calc_ear(face_lm, LEFT_EYE_IDX, w, h)
            right_ear = self._calc_ear(face_lm, RIGHT_EYE_IDX, w, h)
            ear = (left_ear + right_ear) / 2.0

            # Find or create tracker
            tid = self._match_track(cx, cy)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _FaceTrack(first_seen=now)

            track = self._tracks[tid]
            track.last_seen = now
            track.center_x = cx
            track.center_y = cy
            track.ear_history.append(ear)
            track.pos_history.append((cx, cy))

            # Blink detection: EAR drops below threshold then rises back
            if ear < self.EAR_BLINK_THRESHOLD:
                track.blink_state = True   # eye closing
            elif track.blink_state and ear >= self.EAR_BLINK_THRESHOLD:
                track.blink_state = False  # eye opening → blink complete
                track.blink_count += 1

            track.last_ear = ear

        self._cleanup(now)

    def get_liveness(self, bbox) -> LivenessStatus:
        """Get liveness status for a face matched by its bounding box.

        Args:
            bbox: (x1, y1, x2, y2) bounding box

        Returns:
            LivenessStatus with is_live=True/False/None
        """
        x1, y1, x2, y2 = bbox[:4]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        tid = self._match_track(cx, cy)
        if tid is None:
            return LivenessStatus(
                is_live=None, score=0.5, reason="no_track",
                blinks=0, ear=0.3, movement=0.0, track_time=0.0
            )

        track = self._tracks[tid]
        now = time.time()
        track_time = now - track.first_seen

        # Calculate movement (std of nose positions)
        movement = 0.0
        if len(track.pos_history) > 10:
            positions = np.array(list(track.pos_history))
            movement = np.std(positions, axis=0).mean()

        # Still collecting data
        if track_time < self.MIN_TRACK_TIME:
            return LivenessStatus(
                is_live=None, score=0.5,
                reason=f"analyzing ({track_time:.1f}/{self.MIN_TRACK_TIME:.0f}s)",
                blinks=track.blink_count, ear=track.last_ear,
                movement=round(movement, 2), track_time=round(track_time, 1)
            )

        # ── Make judgment ───────────────────────────────────
        has_blink = track.blink_count > 0
        has_movement = movement > self.MOVEMENT_THRESHOLD

        # Score: blinks are the ONLY definitive signal
        # Movement is informational only — hand tremor while holding
        # a phone creates false movement that must NOT bypass blink check
        blink_score = min(1.0, track.blink_count / 2.0)
        move_score = min(1.0, movement / (self.MOVEMENT_THRESHOLD * 3))
        score = blink_score * 0.85 + move_score * 0.15

        # CRITICAL: Only blinks confirm liveness!
        # Movement alone CANNOT confirm live (hand holding phone = movement)
        is_live = has_blink

        reasons = []
        if not has_blink:
            reasons.append("no_blinks")
        if not has_movement:
            reasons.append("no_movement")

        return LivenessStatus(
            is_live=is_live, score=round(score, 3),
            reason="; ".join(reasons) if reasons else "pass",
            blinks=track.blink_count, ear=track.last_ear,
            movement=round(movement, 2), track_time=round(track_time, 1)
        )

    # ── Private helpers ─────────────────────────────────────

    def _calc_ear(self, landmarks, indices, w, h) -> float:
        """Calculate Eye Aspect Ratio from 6 landmark points.

        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

        Open eye: EAR ≈ 0.25-0.35
        Closed eye: EAR < 0.20
        """
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]

        def dist(a, b):
            return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        vert1 = dist(pts[1], pts[5])   # p2 - p6
        vert2 = dist(pts[2], pts[4])   # p3 - p5
        horiz = dist(pts[0], pts[3])   # p1 - p4

        if horiz < 1e-6:
            return 0.3
        return (vert1 + vert2) / (2.0 * horiz)

    def _match_track(self, cx, cy) -> int | None:
        """Find the closest existing tracker within MATCH_DISTANCE."""
        best_id = None
        best_dist = self.MATCH_DISTANCE
        for tid, track in self._tracks.items():
            d = np.sqrt((cx - track.center_x) ** 2 + (cy - track.center_y) ** 2)
            if d < best_dist:
                best_dist = d
                best_id = tid
        return best_id

    def _cleanup(self, now: float):
        """Remove stale trackers that haven't been seen recently."""
        stale = [k for k, v in self._tracks.items()
                 if now - v.last_seen > self.STALE_TIMEOUT]
        for k in stale:
            del self._tracks[k]


# ── Singleton ───────────────────────────────────────────────

_liveness = None

def get_liveness() -> MultiFrameLiveness:
    global _liveness
    if _liveness is None:
        _liveness = MultiFrameLiveness()
    return _liveness
