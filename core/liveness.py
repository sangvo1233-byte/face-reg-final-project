"""
Multi-frame liveness tracking for Detect V3.

The module exposes two APIs:
- MultiFrameLiveness: compatibility API used by older Detect V3 code.
- StreamingLivenessTracker: stateful stream API used by browser_ws/local_direct.
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from loguru import logger
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config

RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
FACE_LANDMARKER_MODEL = str(config.MODELS_DIR / "face_landmarker.task")


@dataclass
class LivenessStatus:
    is_live: object
    score: float
    reason: str
    blinks: int
    ear: float
    movement: float
    track_time: float


@dataclass
class _FaceTrack:
    ear_history: deque = field(default_factory=lambda: deque(maxlen=200))
    pos_history: deque = field(default_factory=lambda: deque(maxlen=200))
    blink_count: int = 0
    last_ear: float = 0.30
    open_ear_baseline: float = 0.30
    min_ear_since_open: float = 0.30
    blink_state: bool = False
    first_seen: float = 0.0
    last_seen: float = 0.0
    center_x: float = 0.0
    center_y: float = 0.0
    frame_count: int = 0


class MultiFrameLiveness:
    """MediaPipe FaceLandmarker based multi-frame liveness tracker."""

    MATCH_DISTANCE = 120
    STALE_TIMEOUT = 3.0

    def __init__(self):
        self._landmarker = None
        self._available = False
        self._tracks: dict[int, _FaceTrack] = {}
        self._next_id = 0
        self._frame_ts = 0
        self._init_landmarker()

    def close(self):
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

    def process_frame(self, frame: np.ndarray):
        if not self._available or self._landmarker is None:
            return

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)
        now = time.time()

        if not result.face_landmarks:
            self._cleanup(now)
            return

        for face_lm in result.face_landmarks:
            nose = face_lm[1]
            cx, cy = nose.x * w, nose.y * h
            ear = (
                self._calc_ear(face_lm, LEFT_EYE_IDX, w, h)
                + self._calc_ear(face_lm, RIGHT_EYE_IDX, w, h)
            ) / 2.0

            tid = self._match_track(cx, cy)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _FaceTrack(first_seen=now)

            track = self._tracks[tid]
            track.last_seen = now
            track.center_x = cx
            track.center_y = cy
            track.frame_count += 1
            track.last_ear = ear
            track.ear_history.append(ear)
            track.pos_history.append((cx, cy))

            baseline = self._update_open_ear_baseline(track, ear)
            track.min_ear_since_open = min(track.min_ear_since_open, ear)
            ear_drop = baseline - ear
            closed = (
                ear <= config.DETECT_V3_BLINK_EAR_CLOSED_THRESHOLD
                or ear_drop >= config.DETECT_V3_STREAM_BLINK_MIN_DROP
            )
            reopened = (
                ear >= config.DETECT_V3_BLINK_EAR_OPEN_THRESHOLD
                and ear >= baseline - (config.DETECT_V3_STREAM_BLINK_MIN_DROP * 0.4)
            )

            if closed:
                track.blink_state = True
            elif track.blink_state and reopened:
                track.blink_state = False
                if baseline - track.min_ear_since_open >= config.DETECT_V3_BLINK_EAR_DELTA * 0.7:
                    track.blink_count += 1
                track.min_ear_since_open = ear

        self._cleanup(now)

    def get_liveness(self, bbox) -> LivenessStatus:
        track = self._track_for_bbox(bbox)
        return self._status(track)

    def primary_status(self) -> dict[str, Any]:
        track = self._primary_track()
        status = self._status(track)
        return liveness_status_to_dict(status)

    def _init_landmarker(self):
        if not os.path.exists(FACE_LANDMARKER_MODEL):
            logger.warning(f"FaceLandmarker model not found: {FACE_LANDMARKER_MODEL}")
            return
        try:
            base_options = mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL)
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
            self._available = True
            logger.info("MultiFrameLiveness initialized")
        except Exception as exc:
            logger.warning(f"MultiFrameLiveness unavailable: {exc}")
            self._landmarker = None
            self._available = False

    def _calc_ear(self, landmarks, indices, w: int, h: int) -> float:
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]

        def dist(a, b):
            return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

        vertical_1 = dist(pts[1], pts[5])
        vertical_2 = dist(pts[2], pts[4])
        horizontal = dist(pts[0], pts[3])
        if horizontal < 1e-6:
            return 0.3
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def _update_open_ear_baseline(self, track: _FaceTrack, ear: float) -> float:
        if track.frame_count <= 5:
            track.open_ear_baseline = max(config.DETECT_V3_BLINK_EAR_OPEN_THRESHOLD, float(ear))
            track.min_ear_since_open = min(track.min_ear_since_open, ear)
            return track.open_ear_baseline

        if len(track.ear_history) >= 8:
            candidate = float(np.percentile(np.array(track.ear_history), 80))
        else:
            candidate = max(track.open_ear_baseline, ear)

        track.open_ear_baseline = max(
            config.DETECT_V3_BLINK_EAR_OPEN_THRESHOLD,
            track.open_ear_baseline * 0.85 + candidate * 0.15,
        )
        return track.open_ear_baseline

    def _match_track(self, cx: float, cy: float) -> int | None:
        best_id = None
        best_dist = self.MATCH_DISTANCE
        for tid, track in self._tracks.items():
            distance = float(np.sqrt((cx - track.center_x) ** 2 + (cy - track.center_y) ** 2))
            if distance < best_dist:
                best_dist = distance
                best_id = tid
        return best_id

    def _track_for_bbox(self, bbox) -> _FaceTrack | None:
        x1, y1, x2, y2 = bbox[:4]
        return self._tracks.get(self._match_track((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    def _primary_track(self) -> _FaceTrack | None:
        if not self._tracks:
            return None
        return max(self._tracks.values(), key=lambda track: track.last_seen)

    def _cleanup(self, now: float):
        stale = [tid for tid, track in self._tracks.items() if now - track.last_seen > self.STALE_TIMEOUT]
        for tid in stale:
            del self._tracks[tid]

    def _movement(self, track: _FaceTrack | None) -> float:
        if track is None or len(track.pos_history) < 3:
            return 0.0
        arr = np.array(list(track.pos_history), dtype=np.float32)
        return float(np.std(arr, axis=0).mean())

    def _track_time(self, track: _FaceTrack | None, now: float) -> float:
        return max(0.0, now - track.first_seen) if track and track.first_seen else 0.0

    def _status(self, track: _FaceTrack | None) -> LivenessStatus:
        now = time.time()
        if not self._available:
            return LivenessStatus(None, 0.0, "unavailable", 0, 0.3, 0.0, 0.0)
        if track is None:
            return LivenessStatus(None, 0.5, "waiting_face", 0, 0.3, 0.0, 0.0)

        track_time = self._track_time(track, now)
        movement = self._movement(track)
        blink_score = min(1.0, track.blink_count / 1.0)
        move_score = min(1.0, movement / max(config.DETECT_V3_STREAM_MOVEMENT_THRESHOLD * 3, 1e-6))
        score = round(float(blink_score * 0.85 + move_score * 0.15), 3)

        if track.blink_count > 0:
            return LivenessStatus(True, score, "pass", track.blink_count, track.last_ear, movement, track_time)
        if track_time < config.DETECT_V3_STREAM_MIN_TRACK_SECONDS:
            return LivenessStatus(None, score, "checking", track.blink_count, track.last_ear, movement, track_time)
        if track_time < config.DETECT_V3_STREAM_MAX_CHECK_SECONDS:
            return LivenessStatus(None, score, "blink_required", track.blink_count, track.last_ear, movement, track_time)
        return LivenessStatus(False, score, "no_blinks", track.blink_count, track.last_ear, movement, track_time)


def liveness_status_to_dict(status: LivenessStatus) -> dict[str, Any]:
    if status.reason == "waiting_face":
        state, message = "waiting_face", "Move your face into the frame"
    elif status.reason == "checking":
        state, message = "checking", "Tracking liveness..."
    elif status.reason == "blink_required":
        state, message = "blink_required", "Blink once to confirm liveness"
    elif status.is_live is True:
        state, message = "live", "Live face confirmed"
    elif status.is_live is False:
        state, message = "spoof", "No blink detected"
    else:
        state, message = "unavailable", "FaceMesh liveness is unavailable"

    return {
        "state": state,
        "message": message,
        "score": round(float(status.score), 3),
        "blinks": status.blinks,
        "ear": round(float(status.ear), 3),
        "movement": round(float(status.movement), 2),
        "track_time": round(float(status.track_time), 2),
    }


class StreamingLivenessTracker:
    """Streaming API used by DetectV3RuntimeSession."""

    def __init__(self):
        self._tracker = MultiFrameLiveness()

    def close(self):
        self._tracker.close()

    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        self._tracker.process_frame(frame)
        return self._tracker.primary_status()

    def get_liveness(self, bbox) -> dict[str, Any]:
        return liveness_status_to_dict(self._tracker.get_liveness(bbox))


_liveness = None


def get_liveness() -> MultiFrameLiveness:
    global _liveness
    if _liveness is None:
        _liveness = MultiFrameLiveness()
    return _liveness
