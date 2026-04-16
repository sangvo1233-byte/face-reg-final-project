"""
Active challenge service for Detect V3.

The normal /api/scan/v3 path stays passive. When a frame is suspicious but not
bad enough to block, Detect V3 creates a short-lived challenge. The browser then
captures multiple frames while the user turns left/right, and this service
verifies both identity and pose before recording attendance.
"""
from __future__ import annotations

import random
import secrets
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

import config
from core.anti_spoof import get_anti_spoof
from core.database import get_db
from core.face_engine import get_engine
from core.moire import get_moire_detector


CHALLENGE_COPY = {
    "turn_left": {
        "label": "Turn Left",
        "instruction": "Turn your face to the LEFT",
    },
    "turn_right": {
        "label": "Turn Right",
        "instruction": "Turn your face to the RIGHT",
    },
    "blink": {
        "label": "Blink",
        "instruction": "Blink once, then keep looking at the camera",
    },
}

RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]


class ChallengeV3Service:
    """In-memory active challenge coordinator.

    This is intentionally process-local. It is enough for the current single
    FastAPI process. If the app is later deployed with multiple workers, move
    this store to Redis or another shared cache.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._blink_supported: bool | None = None

    def create_challenge(
        self,
        *,
        session_id: int,
        candidate: dict[str, Any],
        reason: str,
        moire_score: float,
        liveness_score: float,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._cleanup_expired()

        challenge_type = random.choice(self._available_challenge_types())
        challenge_id = secrets.token_urlsafe(18)
        expires_at = time.time() + config.DETECT_V3_CHALLENGE_TTL_SECONDS

        record = {
            "id": challenge_id,
            "type": challenge_type,
            "session_id": session_id,
            "candidate": candidate,
            "reason": reason,
            "moire_score": moire_score,
            "liveness_score": liveness_score,
            "diagnostics": diagnostics or {},
            "expires_at": expires_at,
            "created_at": time.time(),
        }

        with self._lock:
            self._store[challenge_id] = record

        copy = CHALLENGE_COPY[challenge_type]
        return {
            "id": challenge_id,
            "type": challenge_type,
            "label": copy["label"],
            "instruction": copy["instruction"],
            "expires_in": config.DETECT_V3_CHALLENGE_TTL_SECONDS,
            "required_frames": config.DETECT_V3_CHALLENGE_MIN_FRAMES,
        }

    def verify_challenge(
        self,
        challenge_id: str,
        frames: list[np.ndarray],
    ) -> dict[str, Any]:
        record = self._pop_challenge(challenge_id)
        if not record:
            return self._failed("Challenge expired or not found")

        if len(frames) < config.DETECT_V3_CHALLENGE_MIN_FRAMES:
            return self._failed("Not enough challenge frames")

        db = get_db()
        active = db.get_active_session()
        if not active or int(active["id"]) != int(record["session_id"]):
            return self._failed("Attendance session changed during challenge")

        candidate = record["candidate"]
        student_id = candidate["student_id"]
        engine = get_engine()
        engine._ensure_model()

        good_frames: list[np.ndarray] = []
        pose_values: list[float] = []
        identity_scores: list[float] = []
        moire_scores: list[float] = []
        liveness_scores: list[float] = []
        max_frames = config.DETECT_V3_CHALLENGE_MAX_FRAMES
        moire_detector = get_moire_detector()
        anti_spoof = get_anti_spoof()

        for frame in frames[:max_frames]:
            face = engine.detect_largest(frame)
            if face is None or face.embedding is None or len(face.embedding) == 0:
                continue

            match = engine.match_with_threshold(
                face.embedding,
                config.DETECT_V3_COSINE_THRESHOLD,
            )
            if not match.matched or match.student_id != student_id:
                continue

            metrics = engine.get_face_metrics(frame, face)
            bbox = [int(v) for v in face.bbox[:4]]
            x1, y1, x2, y2 = bbox
            face_roi = frame[max(0, y1):y2, max(0, x1):x2]
            moire = moire_detector.analyze_single(face_roi)
            liveness = anti_spoof.check(frame, face.bbox)

            pose_values.append(float(metrics.get("nose_x_disp", 0.0)))
            identity_scores.append(float(match.score))
            moire_scores.append(float(moire.get("moire_score", 1.0)))
            liveness_scores.append(float(liveness.score))
            good_frames.append(frame)

        if len(good_frames) < config.DETECT_V3_CHALLENGE_MIN_FRAMES:
            return self._failed(
                "Challenge failed: identity was not stable",
                details={
                    "good_frames": len(good_frames),
                    "required_frames": config.DETECT_V3_CHALLENGE_MIN_FRAMES,
                },
            )

        if self._has_hard_spoof_signal(moire_scores, liveness_scores):
            return self._failed(
                "Challenge failed: spoof signal remained during verification",
                details={
                    "moire_scores": [round(v, 3) for v in moire_scores],
                    "liveness_scores": [round(v, 3) for v in liveness_scores],
                },
            )

        if record["type"] == "blink":
            action_ok = self._verify_blink(good_frames)
            if not action_ok["passed"]:
                return self._failed(
                    action_ok["message"],
                    details={
                        "good_frames": len(good_frames),
                        "ear_values": action_ok.get("ear_values", []),
                        "closed_frames": action_ok.get("closed_frames", 0),
                        "open_frames": action_ok.get("open_frames", 0),
                        "challenge_type": record["type"],
                    },
                )
        else:
            action_ok = self._verify_pose(record["type"], pose_values)
            if not action_ok["passed"]:
                return self._failed(
                    action_ok["message"],
                    details={
                        "good_frames": len(good_frames),
                        "pose_frames": action_ok["pose_frames"],
                        "pose_values": [round(v, 3) for v in pose_values],
                        "challenge_type": record["type"],
                    },
                )

        confidence = max(identity_scores) if identity_scores else candidate["confidence"]
        evidence_frame = good_frames[-1]
        result = self._record_attendance(
            session_id=record["session_id"],
            candidate=candidate,
            confidence=confidence,
            evidence_frame=evidence_frame,
            challenge=record,
            pose_values=pose_values,
            action_details=action_ok,
            moire_scores=moire_scores,
            liveness_scores=liveness_scores,
        )
        return {
            "success": True,
            "challenge_passed": True,
            "scan_version": "v3_challenge",
            "results": [result],
        }

    def _verify_pose(self, challenge_type: str, pose_values: list[float]) -> dict[str, Any]:
        threshold = config.ENROLL_V2_POSE_TURN_THRESHOLD
        required = config.DETECT_V3_CHALLENGE_POSE_FRAMES

        if challenge_type == "turn_left":
            pose_frames = sum(1 for value in pose_values if value >= threshold)
            message = "Challenge failed: turn your face further LEFT"
        else:
            pose_frames = sum(1 for value in pose_values if value <= -threshold)
            message = "Challenge failed: turn your face further RIGHT"

        return {
            "passed": pose_frames >= required,
            "pose_frames": pose_frames,
            "message": message,
        }

    def _verify_blink(self, frames: list[np.ndarray]) -> dict[str, Any]:
        """Verify a single blink using MediaPipe FaceMesh EAR values."""
        if not self._is_blink_supported():
            return {
                "passed": False,
                "message": "Blink verification is unavailable on this server",
                "ear_values": [],
            }

        try:
            import mediapipe as mp
        except Exception:
            return {
                "passed": False,
                "message": "Blink verification is unavailable on this server",
                "ear_values": [],
            }

        ear_values: list[float] = []
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        ) as face_mesh:
            for frame in frames:
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)
                if not result.multi_face_landmarks:
                    continue

                landmarks = result.multi_face_landmarks[0].landmark
                left_ear = self._calc_ear(landmarks, LEFT_EYE_IDX, w, h)
                right_ear = self._calc_ear(landmarks, RIGHT_EYE_IDX, w, h)
                ear_values.append((left_ear + right_ear) / 2.0)

        if len(ear_values) < config.DETECT_V3_CHALLENGE_MIN_FRAMES:
            return {
                "passed": False,
                "message": "Challenge failed: eyes were not tracked clearly",
                "ear_values": [round(v, 3) for v in ear_values],
            }

        closed_threshold = config.DETECT_V3_BLINK_EAR_CLOSED_THRESHOLD
        open_threshold = config.DETECT_V3_BLINK_EAR_OPEN_THRESHOLD
        delta_threshold = config.DETECT_V3_BLINK_EAR_DELTA
        closed_frames = sum(1 for value in ear_values if value <= closed_threshold)
        open_frames = sum(1 for value in ear_values if value >= open_threshold)
        ear_delta = max(ear_values) - min(ear_values)
        passed = (
            closed_frames >= 1
            and open_frames >= 2
            and ear_delta >= delta_threshold
        )

        return {
            "passed": passed,
            "message": (
                "Challenge failed: blink once clearly"
                if not passed else ""
            ),
            "ear_values": [round(v, 3) for v in ear_values],
            "closed_frames": closed_frames,
            "open_frames": open_frames,
            "ear_delta": round(ear_delta, 3),
        }

    def _calc_ear(self, landmarks, indices: list[int], w: int, h: int) -> float:
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]

        def dist(a, b):
            return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

        vertical_1 = dist(pts[1], pts[5])
        vertical_2 = dist(pts[2], pts[4])
        horizontal = dist(pts[0], pts[3])
        if horizontal < 1e-6:
            return 0.3
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def _available_challenge_types(self) -> tuple[str, ...]:
        if self._is_blink_supported():
            return ("turn_left", "turn_right", "blink")
        return ("turn_left", "turn_right")

    def _is_blink_supported(self) -> bool:
        if self._blink_supported is not None:
            return self._blink_supported
        try:
            import mediapipe as mp
            self._blink_supported = hasattr(mp, "solutions") and hasattr(
                mp.solutions, "face_mesh"
            )
        except Exception:
            self._blink_supported = False
        return self._blink_supported

    def _has_hard_spoof_signal(
        self,
        moire_scores: list[float],
        liveness_scores: list[float],
    ) -> bool:
        if not moire_scores or not liveness_scores:
            return True

        avg_moire = sum(moire_scores) / len(moire_scores)
        avg_liveness = sum(liveness_scores) / len(liveness_scores)
        return (
            avg_moire < config.DETECT_V3_MOIRE_BLOCK_THRESHOLD
            or avg_liveness < config.DETECT_V3_LIVENESS_BLOCK_THRESHOLD
        )

    def _record_attendance(
        self,
        *,
        session_id: int,
        candidate: dict[str, Any],
        confidence: float,
        evidence_frame: np.ndarray,
        challenge: dict[str, Any],
        pose_values: list[float],
        action_details: dict[str, Any],
        moire_scores: list[float],
        liveness_scores: list[float],
    ) -> dict[str, Any]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        student_id = candidate["student_id"]
        evidence = str(config.EVIDENCE_DIR / f"{student_id}_challenge_{ts}.jpg")
        cv2.imwrite(evidence, evidence_frame)

        db = get_db()
        db_result = db.mark_attendance(session_id, student_id, confidence, evidence)
        session = db.get_session(session_id)
        student = db.get_student(student_id) or {}
        emb_count = db.get_embedding_count(student_id)
        evidence_filename = Path(evidence).name

        return {
            "name": candidate.get("name") or student.get("name") or student_id,
            "student_id": student_id,
            "class_name": student.get("class_name", candidate.get("class_name", "")),
            "session_id": session_id,
            "session_name": (
                session.get("name", f"Session #{session_id}")
                if session else f"Session #{session_id}"
            ),
            "confidence": confidence,
            "status": "present" if db_result["success"] else "already",
            "message": db_result["message"],
            "bbox": candidate.get("bbox", []),
            "evidence_url": f"/api/evidence/{evidence_filename}",
            "moire_score": (
                sum(moire_scores) / len(moire_scores)
                if moire_scores else challenge.get("moire_score")
            ),
            "moire_is_screen": False,
            "liveness_score": (
                sum(liveness_scores) / len(liveness_scores)
                if liveness_scores else challenge.get("liveness_score")
            ),
            "embedding_count": emb_count,
            "enroll_type": "multi_angle_v2" if emb_count >= 3 else "single",
            "challenge_passed": True,
            "challenge_type": challenge["type"],
            "challenge_pose_values": [round(v, 3) for v in pose_values],
            "challenge_action_details": action_details,
        }

    def _pop_challenge(self, challenge_id: str) -> dict[str, Any] | None:
        now = time.time()
        with self._lock:
            record = self._store.pop(challenge_id, None)

        if not record:
            return None
        if record["expires_at"] < now:
            return None
        return record

    def _cleanup_expired(self) -> None:
        now = time.time()
        with self._lock:
            expired = [
                challenge_id
                for challenge_id, record in self._store.items()
                if record["expires_at"] < now
            ]
            for challenge_id in expired:
                self._store.pop(challenge_id, None)

    def _failed(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        logger.warning(f"Detect V3 challenge failed: {message}")
        return {
            "success": False,
            "challenge_passed": False,
            "scan_version": "v3_challenge",
            "results": [{
                "name": "Unknown",
                "student_id": "",
                "confidence": 0,
                "status": "challenge_failed",
                "message": message,
                "details": details or {},
            }],
        }


_challenge_v3_service: ChallengeV3Service | None = None


def get_challenge_v3_service() -> ChallengeV3Service:
    global _challenge_v3_service
    if _challenge_v3_service is None:
        _challenge_v3_service = ChallengeV3Service()
    return _challenge_v3_service
