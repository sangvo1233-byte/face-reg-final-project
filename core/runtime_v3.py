"""
Transport-agnostic Detect V3 stream runtime.

Both browser_ws and local_direct feed BGR frames into this class and receive
the same event family back.
"""
from __future__ import annotations

import random
import time
import uuid
from collections import deque
from typing import Any

import numpy as np
from loguru import logger

import config
from core.anti_spoof import get_anti_spoof
from core.database import get_db
from core.detect_v3 import get_detect_v3_service
from core.face_engine import get_engine
from core.liveness import StreamingLivenessTracker
from core.moire import MoireDetector

CHALLENGE_TYPES = ["blink", "turn_left", "turn_right"]
CHALLENGE_LABELS = {
    "blink": "Blink Once",
    "turn_left": "Turn Left",
    "turn_right": "Turn Right",
}
CHALLENGE_INSTRUCTIONS = {
    "blink": "Blink once while looking at the camera",
    "turn_left": "Turn your face to the LEFT",
    "turn_right": "Turn your face to the RIGHT",
}
CHALLENGE_TIMEOUT = float(getattr(config, "DETECT_V3_CHALLENGE_TTL_SECONDS", 10))
CHALLENGE_BASELINE_FRAMES = 5
CHALLENGE_PASS_COOLDOWN = 15.0
TURN_THRESHOLD = 0.06


class ActiveChallenge:
    def __init__(self, challenge_type: str, candidate: dict[str, Any]):
        self.id = str(uuid.uuid4())[:8]
        self.type = challenge_type
        self.label = CHALLENGE_LABELS.get(challenge_type, challenge_type)
        self.instruction = CHALLENGE_INSTRUCTIONS.get(challenge_type, self.label)
        self.candidate = candidate
        self.started_at = time.time()
        self.expires_at = self.started_at + CHALLENGE_TIMEOUT
        self.frames_processed = 0
        self.identity_frames = 0
        self.blink_baseline_count = 0
        self.blink_detected = False
        self.pose_baseline = None
        self.pose_baseline_frames = 0
        self.pose_frames_passed = 0

    @property
    def remaining_ms(self) -> int:
        return max(0, int((self.expires_at - time.time()) * 1000))

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.expires_at

    @property
    def required_frames(self) -> int:
        if self.type == "blink":
            return 1
        return int(config.DETECT_V3_CHALLENGE_POSE_FRAMES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "instruction": self.instruction,
            "candidate": self.candidate,
            "expires_in": round(max(0.0, self.expires_at - time.time()), 1),
            "remaining_ms": self.remaining_ms,
            "required_frames": self.required_frames,
            "frames_processed": self.frames_processed,
            "identity_frames": self.identity_frames,
        }


class DetectV3RuntimeSession:
    def __init__(self, session_id: int):
        self.session_id = session_id
        self.db = get_db()
        self.session = self.db.get_session(session_id)
        self.engine = get_engine()
        self.anti_spoof = get_anti_spoof()
        self.detect_service = get_detect_v3_service()
        self.liveness = StreamingLivenessTracker()
        self.moire_detector = MoireDetector()
        self.started_at = time.time()
        self.last_detect_at = 0.0
        self.detect_paused_until = 0.0
        self.detect_cycle_count = 0
        self.detect_interval = 1.0 / max(config.DETECT_V3_STREAM_DETECT_FPS, 1)
        self.attendance_cooldown = 2.5
        self.frame_timestamps: deque[float] = deque(maxlen=60)
        self.moire_results: dict[int, dict[str, Any]] = {}
        self.active_challenge: ActiveChallenge | None = None
        self.challenge_passed = False
        self.challenge_passed_at = 0.0
        self.challenge_fail_count = 0
        self.last_challenge_student_id: str | None = None

    def close(self):
        self.liveness.close()

    def process_frame(self, frame: np.ndarray) -> list[dict[str, Any]]:
        now = time.time()
        self.frame_timestamps.append(now)
        live_summary = self.liveness.process_frame(frame)

        if self.active_challenge is not None:
            return self._process_challenge_frame(frame, live_summary, now)

        if now < self.detect_paused_until:
            return [self._state_event(live_summary, status="cooldown", message="Attendance recorded. Continue scanning...")]

        if now - self.last_detect_at < self.detect_interval:
            return [self._state_event(live_summary)]

        return self._detect_cycle(frame, live_summary, now)

    def _detect_cycle(self, frame: np.ndarray, live: dict[str, Any], now: float) -> list[dict[str, Any]]:
        self.last_detect_at = now
        self.detect_cycle_count += 1
        run_moire = self.detect_cycle_count % max(config.DETECT_V3_STREAM_MOIRE_EVERY_N_DETECT, 1) == 0
        if run_moire:
            self.moire_results = {}

        faces = self.engine.detect(frame)
        if not faces:
            return [self._state_event(live, status="waiting_face", message="Move your face into the frame", faces_detected=0)]

        events = []
        for index, face in enumerate(faces):
            if face.embedding is None or len(face.embedding) == 0:
                continue
            event = self._process_face(frame, face, index, len(faces), run_moire, now)
            if event:
                events.append(event)

        return events or [self._state_event(live, status="unknown", message="No usable face embedding", faces_detected=len(faces))]

    def _process_face(self, frame, face, index: int, faces_detected: int, run_moire: bool, now: float) -> dict[str, Any] | None:
        bbox = self.detect_service.bbox_list(face.bbox)
        x1, y1, x2, y2 = bbox
        face_roi = frame[max(0, y1):y2, max(0, x1):x2]
        face_live = self.liveness.get_liveness(bbox)
        moire = self._moire_for_face(index, face_roi, run_moire)
        moire_score = float(moire.get("moire_score", 1.0))
        moire_decision = self.detect_service.moire_decision(moire_score, moire)
        passive = self.anti_spoof.check(frame, face.bbox)
        passive_score = float(passive.score)
        passive_decision = self.detect_service.liveness_decision(passive_score, passive.reason)

        diagnostics = {
            "faces_detected": faces_detected,
            "liveness": face_live,
            "moire": {**moire, "decision": moire_decision["action"], "reason": moire_decision.get("reason", "")},
            "passive_liveness": {
                "score": round(passive_score, 3),
                "reason": passive.reason,
                "decision": passive_decision["action"],
            },
            "match": {
                "matched": False,
                "confidence": 0,
                "student_id": "",
                "name": "",
                "threshold": config.DETECT_V3_COSINE_THRESHOLD,
            },
        }

        if moire_decision["action"] == "block":
            result = self.detect_service.spoof_result(
                bbox=bbox,
                message=f"Screen detected (moire: {moire_score:.0%})",
                moire_score=moire_score,
                moire_is_screen=True,
                liveness_score=passive_score,
            )
            return self._result_event(result, diagnostics)

        if face_live["state"] == "spoof":
            result = self.detect_service.spoof_result(
                bbox=bbox,
                message=face_live["message"],
                moire_score=moire_score,
                moire_is_screen=False,
                liveness_score=face_live.get("score", 0),
            )
            return self._result_event(result, diagnostics)

        if face_live["state"] != "live":
            return self._state_event(face_live, status=face_live["state"], message=face_live["message"], **diagnostics)

        if passive_decision["action"] == "block":
            result = self.detect_service.spoof_result(
                bbox=bbox,
                message=f"Liveness check failed ({passive.reason})",
                moire_score=moire_score,
                moire_is_screen=False,
                liveness_score=passive_score,
            )
            return self._result_event(result, diagnostics)

        match = self.engine.match_with_threshold(face.embedding, config.DETECT_V3_COSINE_THRESHOLD)
        diagnostics["match"] = {
            "matched": bool(match.matched),
            "confidence": float(match.score),
            "student_id": match.student_id,
            "name": match.name,
            "threshold": config.DETECT_V3_COSINE_THRESHOLD,
        }

        if not match.matched:
            result = self.detect_service.unknown_result(
                bbox=bbox,
                match=match,
                moire_score=moire_score,
                liveness_score=passive_score,
            )
            return self._result_event(result, diagnostics)

        student = self.db.get_student(match.student_id) or {}
        emb_count = self.db.get_embedding_count(match.student_id)
        candidate = self.detect_service.candidate_from_match(
            match=match,
            student=student,
            emb_count=emb_count,
            bbox=bbox,
        )
        suspicion = self.detect_service.challenge_reason(moire_decision, passive_decision)
        if suspicion and config.DETECT_V3_CHALLENGE_ENABLED:
            if self._challenge_passed_valid(now) and self.last_challenge_student_id == match.student_id:
                pass
            else:
                return self._start_challenge(candidate, diagnostics)

        result = self.detect_service.record_attendance_result(
            frame=frame,
            session_id=self.session_id,
            session=self.session,
            match=match,
            student=student,
            emb_count=emb_count,
            bbox=bbox,
            moire_score=moire_score,
            liveness_score=passive_score,
        )
        self.detect_paused_until = now + self.attendance_cooldown
        return self._result_event(result, diagnostics)

    def _start_challenge(self, candidate: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
        challenge_type = random.choice(CHALLENGE_TYPES)
        self.active_challenge = ActiveChallenge(challenge_type, candidate)
        self.active_challenge.blink_baseline_count = diagnostics.get("liveness", {}).get("blinks", 0)
        logger.info(f"Challenge started: {challenge_type} for {candidate.get('student_id')}")
        return {
            "type": "challenge_required",
            "status": "challenge_required",
            "message": "Additional verification required",
            "challenge": self.active_challenge.to_dict(),
            "challenge_progress": self._challenge_progress_payload(self.active_challenge, "collecting", "Follow the instruction shown on screen"),
            **diagnostics,
            "fps": self._fps(),
        }

    def _process_challenge_frame(self, frame: np.ndarray, live: dict[str, Any], now: float) -> list[dict[str, Any]]:
        challenge = self.active_challenge
        if challenge is None:
            return [self._state_event(live)]
        if challenge.is_expired:
            return self._end_challenge("timeout", "Challenge timed out")
        if live.get("state") == "spoof":
            return self._end_challenge("failed", "Spoof detected during challenge")

        challenge.frames_processed += 1
        faces = self.engine.detect(frame)
        if not faces:
            return [self._challenge_progress_event("No face detected")]

        valid_faces = [f for f in faces if f.embedding is not None and len(f.embedding) > 0]
        if not valid_faces:
            return [self._challenge_progress_event("No usable face embedding")]

        face = max(valid_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        match = self.engine.match_with_threshold(face.embedding, config.DETECT_V3_COSINE_THRESHOLD)
        if not (match.matched and match.student_id == challenge.candidate.get("student_id")):
            return [self._challenge_progress_event("Identity mismatch")]
        challenge.identity_frames += 1

        bbox = self.detect_service.bbox_list(face.bbox)
        x1, y1, x2, y2 = bbox
        face_roi = frame[max(0, y1):y2, max(0, x1):x2]
        moire = self.moire_detector.analyze(face_roi)
        if float(moire.get("moire_score", 1.0)) < config.DETECT_V3_MOIRE_BLOCK_THRESHOLD:
            return self._end_challenge("failed", "Screen detected during challenge")

        completed = False
        if challenge.type == "blink":
            completed = live.get("blinks", 0) > challenge.blink_baseline_count
            challenge.blink_detected = completed
        else:
            metrics = self.engine.get_face_metrics(frame, face)
            completed = self._check_turn(challenge, float(metrics.get("nose_x_disp", 0.0)))

        if completed:
            return self._end_challenge_pass(frame, match, face, now)
        return [self._challenge_progress_event(f"Keep going: {challenge.label}")]

    def _check_turn(self, challenge: ActiveChallenge, nose_x_disp: float) -> bool:
        if challenge.pose_baseline_frames < CHALLENGE_BASELINE_FRAMES:
            challenge.pose_baseline = nose_x_disp if challenge.pose_baseline is None else challenge.pose_baseline * 0.7 + nose_x_disp * 0.3
            challenge.pose_baseline_frames += 1
            return False

        baseline = challenge.pose_baseline or 0.0
        dx = nose_x_disp - baseline
        if challenge.type == "turn_left" and dx >= TURN_THRESHOLD:
            challenge.pose_frames_passed += 1
        elif challenge.type == "turn_right" and dx <= -TURN_THRESHOLD:
            challenge.pose_frames_passed += 1
        else:
            challenge.pose_frames_passed = max(0, challenge.pose_frames_passed - 1)
        return challenge.pose_frames_passed >= config.DETECT_V3_CHALLENGE_POSE_FRAMES

    def _end_challenge_pass(self, frame, match, face, now: float) -> list[dict[str, Any]]:
        challenge = self.active_challenge
        self.challenge_passed = True
        self.challenge_passed_at = now
        self.challenge_fail_count = 0
        self.last_challenge_student_id = challenge.candidate.get("student_id") if challenge else None
        self.active_challenge = None

        student = self.db.get_student(match.student_id) or {}
        emb_count = self.db.get_embedding_count(match.student_id)
        bbox = self.detect_service.bbox_list(face.bbox)
        result = self.detect_service.record_attendance_result(
            frame=frame,
            session_id=self.session_id,
            session=self.session,
            match=match,
            student=student,
            emb_count=emb_count,
            bbox=bbox,
            moire_score=1.0,
            liveness_score=1.0,
        )
        self.detect_paused_until = now + self.attendance_cooldown
        return [{"type": "attendance", "status": result.get("status", "present"), "message": result.get("message", "Attendance recorded"), "challenge_passed": True, "fps": self._fps(), **result}]

    def _end_challenge(self, reason: str, message: str) -> list[dict[str, Any]]:
        challenge = self.active_challenge
        self.challenge_fail_count += 1
        self.active_challenge = None
        payload = self._challenge_progress_payload(challenge, "failed", message) if challenge else {}
        return [{
            "type": "scan_state",
            "status": "challenge_failed",
            "message": message,
            "challenge_result": reason,
            "challenge_fail_count": self.challenge_fail_count,
            "challenge_progress": payload,
            "fps": self._fps(),
        }]

    def _challenge_progress_event(self, feedback: str) -> dict[str, Any]:
        challenge = self.active_challenge
        return {
            "type": "scan_state",
            "status": "challenge_active",
            "message": feedback,
            "challenge": challenge.to_dict() if challenge else {},
            "challenge_progress": self._challenge_progress_payload(challenge, "collecting", feedback) if challenge else {},
            "fps": self._fps(),
        }

    def _moire_for_face(self, index: int, face_roi: np.ndarray, run_moire: bool) -> dict[str, Any]:
        if run_moire or index not in self.moire_results:
            self.moire_results[index] = self.moire_detector.analyze(face_roi)
        return self.moire_results[index]

    def _result_event(self, result: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
        status = result.get("status", "unknown")
        event_type = "attendance" if status in ("present", "already") else "scan_state"
        if status == "challenge_required":
            event_type = "challenge_required"
        return {
            "type": event_type,
            "status": status,
            "message": result.get("message", ""),
            "fps": self._fps(),
            **diagnostics,
            "result": result,
            **result,
        }

    def _state_event(self, live: dict[str, Any], *, status: str | None = None, message: str | None = None, **extra) -> dict[str, Any]:
        return {
            "type": "scan_state",
            "status": status or live.get("state", "unknown"),
            "message": message or live.get("message", ""),
            "liveness": live,
            "faces_detected": extra.pop("faces_detected", 0),
            "moire": extra.pop("moire", {}),
            "passive_liveness": extra.pop("passive_liveness", {}),
            "match": extra.pop("match", {}),
            "fps": self._fps(),
            **extra,
        }

    def _challenge_progress_payload(self, challenge: ActiveChallenge, status: str, feedback: str) -> dict[str, Any]:
        collected = 1 if challenge.type == "blink" and challenge.blink_detected else challenge.pose_frames_passed
        return {
            "id": challenge.id,
            "type": challenge.type,
            "label": challenge.label,
            "instruction": challenge.instruction,
            "status": status,
            "feedback": feedback,
            "collected_frames": collected,
            "required_frames": challenge.required_frames,
            "frames_processed": challenge.frames_processed,
            "identity_frames": challenge.identity_frames,
            "elapsed_ms": int(max(0.0, (time.time() - challenge.started_at) * 1000)),
            "remaining_ms": challenge.remaining_ms,
        }

    def _challenge_passed_valid(self, now: float) -> bool:
        return self.challenge_passed and (now - self.challenge_passed_at) < CHALLENGE_PASS_COOLDOWN

    def _fps(self) -> float:
        if len(self.frame_timestamps) < 2:
            return 0.0
        elapsed = self.frame_timestamps[-1] - self.frame_timestamps[0]
        if elapsed <= 0:
            return 0.0
        return round((len(self.frame_timestamps) - 1) / elapsed, 1)
