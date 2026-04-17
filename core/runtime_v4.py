"""
Transport-agnostic Detect V4.4 stream runtime.

Browser WebSocket and local-direct scanner both feed BGR frames into this
runtime. The runtime owns all anti-spoof, challenge, matching, and attendance
logic; the UI only renders returned events.
"""
from __future__ import annotations

import random
import time
import uuid
from collections import defaultdict, deque
from typing import Any

import numpy as np
from loguru import logger

import config
from core.anti_spoof import get_anti_spoof
from core.database import get_db
from core.detect_v4 import (
    CHALLENGE_BASELINE_FRAMES,
    CHALLENGE_COOLDOWN,
    CHALLENGE_POSE_FRAMES,
    CHALLENGE_TIMEOUT,
    CHALLENGE_TYPES,
    MOIRE_BLOCK_THRESHOLD,
    MOIRE_EVERY_N_DETECT,
    TURN_THRESHOLD,
    V4_COSINE_THRESHOLD,
    DETECT_VERSION,
    FaceMoireTrackStore,
    MoireDetectorV4,
    PhoneRectangleDetectorV42,
    RollingMoireDecision,
    RollingPhoneRectDecision,
    ScreenContextDetectorV41,
    bbox_list,
    collect_suspicious_reasons,
    expanded_roi,
    get_detect_v4_service,
    passive_decision,
)
from core.face_engine import get_engine
from core.liveness import StreamingLivenessTracker


API_CHALLENGE_TYPES = {
    "BLINK": "blink",
    "TURN_LEFT": "turn_left",
    "TURN_RIGHT": "turn_right",
}
CHALLENGE_LABELS = {
    "BLINK": "Blink Once",
    "TURN_LEFT": "Turn Left",
    "TURN_RIGHT": "Turn Right",
}
CHALLENGE_INSTRUCTIONS = {
    "BLINK": "Blink once while looking at the camera",
    "TURN_LEFT": "Turn your face to the LEFT",
    "TURN_RIGHT": "Turn your face to the RIGHT",
}


class ActiveChallengeV4:
    def __init__(self, challenge_type: str, candidate: dict[str, Any], live: dict[str, Any], reason: str):
        self.id = str(uuid.uuid4())[:8]
        self.internal_type = challenge_type
        self.type = API_CHALLENGE_TYPES[challenge_type]
        self.label = CHALLENGE_LABELS[challenge_type]
        self.instruction = CHALLENGE_INSTRUCTIONS[challenge_type]
        self.candidate = candidate
        self.reason = reason
        self.started_at = time.time()
        self.expires_at = self.started_at + CHALLENGE_TIMEOUT
        self.frames_processed = 0
        self.identity_frames = 0
        self.blink_baseline = int(live.get("blinks", 0))
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
        if self.internal_type == "BLINK":
            return 1
        return CHALLENGE_POSE_FRAMES

    @property
    def text(self) -> str:
        return self.label

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


class DetectV4RuntimeSession:
    def __init__(self, session_id: int):
        self.session_id = session_id
        self.db = get_db()
        self.session = self.db.get_session(session_id)
        self.engine = get_engine()
        self.anti_spoof = get_anti_spoof()
        self.service = get_detect_v4_service()
        self.liveness = StreamingLivenessTracker()
        self.moire_detector = MoireDetectorV4()
        self.moire_tracks = FaceMoireTrackStore()
        self.context_detector = ScreenContextDetectorV41()
        self.phone_rect_detector = PhoneRectangleDetectorV42()
        self.rolling_by_face: dict[int, RollingMoireDecision] = defaultdict(RollingMoireDecision)
        self.phone_rolling_by_face: dict[int, RollingPhoneRectDecision] = defaultdict(RollingPhoneRectDecision)
        self.started_at = time.time()
        self.last_detect_at = 0.0
        self.detect_paused_until = 0.0
        self.detect_cycle_count = 0
        self.detect_interval = 1.0 / max(config.DETECT_V3_STREAM_DETECT_FPS, 1)
        self.attendance_cooldown = 2.5
        self.frame_timestamps: deque[float] = deque(maxlen=60)
        self.active_challenge: ActiveChallengeV4 | None = None
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
        run_moire = self.detect_cycle_count % max(MOIRE_EVERY_N_DETECT, 1) == 0

        faces = self.engine.detect(frame)
        self.moire_tracks.begin_cycle()
        if not faces:
            active_track_ids = self.moire_tracks.finish_cycle()
            self._drop_inactive_tracks(active_track_ids)
            return [self._state_event(live, status="waiting_face", message="Move your face into the frame", faces_detected=0)]

        events = []
        try:
            for face in faces:
                if face.embedding is None or len(face.embedding) == 0:
                    continue
                event = self._process_face(frame, face, len(faces), run_moire, now)
                if event:
                    events.append(event)
        finally:
            active_track_ids = self.moire_tracks.finish_cycle()
            self._drop_inactive_tracks(active_track_ids)

        return events or [self._state_event(live, status="unknown", message="No usable face embedding", faces_detected=len(faces))]

    def _process_face(
        self,
        frame: np.ndarray,
        face: Any,
        faces_detected: int,
        run_moire: bool,
        now: float,
    ) -> dict[str, Any] | None:
        bbox = bbox_list(face.bbox)
        moire, rolling, moire_roi_bbox = self._moire_for_face(frame, bbox, run_moire)
        face_live = self.liveness.get_liveness(bbox)
        passive = self.anti_spoof.check(frame, face.bbox)
        passive_status = passive_decision(float(passive.score))
        screen_context = self.context_detector.analyze(frame, bbox, moire_roi_bbox)
        phone_rect = self.phone_rect_detector.analyze(frame, bbox)
        track_id = int(moire.get("track_id") or 0)
        phone_rect_rolling = self.phone_rolling_by_face[track_id].update(phone_rect)
        metrics = self.engine.get_face_metrics(frame, face)
        match = self.engine.match_with_threshold(face.embedding, V4_COSINE_THRESHOLD)

        moire_decision = moire.get("decision_hint", rolling.get("decision", "clean"))
        diagnostics = self._diagnostics(
            faces_detected=faces_detected,
            face_live=face_live,
            moire=moire,
            rolling=rolling,
            screen_context=screen_context,
            phone_rect=phone_rect,
            phone_rect_rolling=phone_rect_rolling,
            passive=passive,
            passive_status=passive_status,
            match=match,
        )

        if rolling.get("decision") == "block" or moire_decision == "block":
            result = self.service.spoof_result(
                bbox=bbox,
                message=f"SCREEN detected ({moire.get('moire_score', 0):.0%})",
                moire_score=float(moire.get("moire_score", 0.0)),
                moire_is_screen=True,
                liveness_score=float(passive.score),
                diagnostics=diagnostics,
            )
            return self._result_event(result, diagnostics)

        if face_live.get("state") == "spoof":
            result = self.service.spoof_result(
                bbox=bbox,
                message=face_live.get("message", "Liveness failed"),
                moire_score=float(moire.get("moire_score", 0.0)),
                moire_is_screen=False,
                liveness_score=float(face_live.get("score", 0.0)),
                diagnostics=diagnostics,
            )
            return self._result_event(result, diagnostics)

        if (
            screen_context.get("decision") == "strong"
            and (rolling.get("decision") == "suspicious" or passive_status == "suspicious")
        ):
            result = self.service.spoof_result(
                bbox=bbox,
                message="Screen context strong",
                moire_score=float(moire.get("moire_score", 0.0)),
                moire_is_screen=True,
                liveness_score=float(passive.score),
                diagnostics=diagnostics,
            )
            return self._result_event(result, diagnostics)

        if phone_rect_rolling.get("decision") == "block":
            result = self.service.spoof_result(
                bbox=bbox,
                message="Phone rectangle detected",
                moire_score=float(moire.get("moire_score", 0.0)),
                moire_is_screen=True,
                liveness_score=float(passive.score),
                diagnostics=diagnostics,
            )
            return self._result_event(result, diagnostics)

        if (
            phone_rect.get("decision") == "strong"
            and (
                moire_decision == "suspicious"
                or screen_context.get("decision") in {"suspicious", "strong"}
                or passive_status == "suspicious"
            )
        ):
            result = self.service.spoof_result(
                bbox=bbox,
                message="Phone rectangle strong",
                moire_score=float(moire.get("moire_score", 0.0)),
                moire_is_screen=True,
                liveness_score=float(passive.score),
                diagnostics=diagnostics,
            )
            return self._result_event(result, diagnostics)

        if passive_status == "block":
            result = self.service.spoof_result(
                bbox=bbox,
                message=f"Passive failed ({passive.score:.2f})",
                moire_score=float(moire.get("moire_score", 0.0)),
                moire_is_screen=False,
                liveness_score=float(passive.score),
                diagnostics=diagnostics,
            )
            return self._result_event(result, diagnostics)

        if face_live.get("state") != "live":
            return self._state_event(
                face_live,
                status=face_live.get("state"),
                message=face_live.get("message", "Tracking liveness"),
                **diagnostics,
            )

        if not match.matched:
            result = self.service.unknown_result(
                bbox=bbox,
                match=match,
                moire_score=float(moire.get("moire_score", 0.0)),
                liveness_score=float(passive.score),
                diagnostics=diagnostics,
            )
            return self._result_event(result, diagnostics)

        student = self.db.get_student(match.student_id) or {}
        emb_count = self.db.get_embedding_count(match.student_id)
        candidate = self.service.candidate_from_match(
            match=match,
            student=student,
            emb_count=emb_count,
            bbox=bbox,
        )
        suspicious_reasons = collect_suspicious_reasons(
            moire_decision=moire_decision,
            screen_context=screen_context,
            phone_rect=phone_rect,
            phone_rect_rolling=phone_rect_rolling,
            passive_status=passive_status,
            passive_score=float(passive.score),
            challenge_fail_count=self.challenge_fail_count,
        )
        if suspicious_reasons and not self._challenge_passed_valid(now, match.student_id):
            return self._start_challenge(candidate, face_live, diagnostics, "; ".join(suspicious_reasons))

        result = self.service.record_attendance_result(
            frame=frame,
            session_id=self.session_id,
            session=self.session,
            match=match,
            student=student,
            emb_count=emb_count,
            bbox=bbox,
            moire_score=float(moire.get("moire_score", 0.0)),
            liveness_score=float(passive.score),
            diagnostics=diagnostics,
        )
        self.detect_paused_until = now + self.attendance_cooldown
        return self._result_event(result, diagnostics)

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
        match = self.engine.match_with_threshold(face.embedding, V4_COSINE_THRESHOLD)
        if not (match.matched and match.student_id == challenge.candidate.get("student_id")):
            return [self._challenge_progress_event("Identity mismatch")]
        challenge.identity_frames += 1

        bbox = bbox_list(face.bbox)
        self.moire_tracks.begin_cycle()
        try:
            moire, rolling, _ = self._moire_for_face(frame, bbox, True)
        finally:
            active_track_ids = self.moire_tracks.finish_cycle()
            self._drop_inactive_tracks(active_track_ids)

        if rolling.get("decision") == "block" or float(moire.get("moire_score", 1.0)) < MOIRE_BLOCK_THRESHOLD:
            return self._end_challenge("failed", "Screen detected during challenge")

        face_live = self.liveness.get_liveness(bbox)
        completed = False
        if challenge.internal_type == "BLINK":
            completed = int(face_live.get("blinks", 0)) > challenge.blink_baseline
            challenge.blink_detected = completed
        else:
            metrics = self.engine.get_face_metrics(frame, face)
            completed = self._check_turn(challenge, float(metrics.get("nose_x_disp", 0.0)))

        if completed:
            return self._end_challenge_pass(frame, match, face, now, moire, face_live)
        return [self._challenge_progress_event(f"Keep going: {challenge.label}")]

    def _moire_for_face(
        self,
        frame: np.ndarray,
        bbox: list[int],
        run_moire: bool,
    ) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
        roi, moire_roi_bbox = expanded_roi(frame, bbox)
        moire_track = self.moire_tracks.match_and_update(bbox)
        track_id = moire_track.track_id

        if run_moire or not moire_track.last_result:
            moire = self.moire_detector.analyze(roi, moire_track)
            rolling = self.rolling_by_face[track_id].update(moire)
        else:
            moire = moire_track.last_result
            rolling = self.rolling_by_face[track_id].summary()
        return moire, rolling, moire_roi_bbox

    def _start_challenge(
        self,
        candidate: dict[str, Any],
        live: dict[str, Any],
        diagnostics: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        challenge_type = random.choice(CHALLENGE_TYPES)
        self.active_challenge = ActiveChallengeV4(challenge_type, candidate, live, reason)
        self.last_challenge_student_id = candidate.get("student_id")
        logger.info(f"Detect V4 challenge started: {challenge_type} for {candidate.get('student_id')}: {reason}")
        return {
            "type": "challenge_required",
            "status": "challenge_required",
            "message": "Additional verification required",
            "challenge": self.active_challenge.to_dict(),
            "challenge_progress": self._challenge_progress_payload(
                self.active_challenge,
                "collecting",
                "Follow the instruction shown on screen",
            ),
            **diagnostics,
            "fps": self._fps(),
        }

    def _check_turn(self, challenge: ActiveChallengeV4, nose_x_disp: float) -> bool:
        if challenge.pose_baseline_frames < CHALLENGE_BASELINE_FRAMES:
            challenge.pose_baseline = (
                nose_x_disp
                if challenge.pose_baseline is None
                else challenge.pose_baseline * 0.7 + nose_x_disp * 0.3
            )
            challenge.pose_baseline_frames += 1
            return False

        baseline = challenge.pose_baseline or 0.0
        dx = nose_x_disp - baseline
        if challenge.internal_type == "TURN_LEFT" and dx >= TURN_THRESHOLD:
            challenge.pose_frames_passed += 1
        elif challenge.internal_type == "TURN_RIGHT" and dx <= -TURN_THRESHOLD:
            challenge.pose_frames_passed += 1
        else:
            challenge.pose_frames_passed = max(0, challenge.pose_frames_passed - 1)
        return challenge.pose_frames_passed >= CHALLENGE_POSE_FRAMES

    def _end_challenge_pass(
        self,
        frame: np.ndarray,
        match: Any,
        face: Any,
        now: float,
        moire: dict[str, Any],
        live: dict[str, Any],
    ) -> list[dict[str, Any]]:
        challenge = self.active_challenge
        self.challenge_passed_at = now
        self.challenge_fail_count = 0
        self.last_challenge_student_id = challenge.candidate.get("student_id") if challenge else None
        self.active_challenge = None

        student = self.db.get_student(match.student_id) or {}
        emb_count = self.db.get_embedding_count(match.student_id)
        bbox = bbox_list(face.bbox)
        result = self.service.record_attendance_result(
            frame=frame,
            session_id=self.session_id,
            session=self.session,
            match=match,
            student=student,
            emb_count=emb_count,
            bbox=bbox,
            moire_score=float(moire.get("moire_score", 1.0)),
            liveness_score=float(live.get("score", 1.0)),
        )
        self.detect_paused_until = now + self.attendance_cooldown
        return [{
            "type": "attendance",
            "status": result.get("status", "present"),
            "message": result.get("message", "Attendance recorded"),
            "challenge_passed": True,
            "fps": self._fps(),
            **result,
        }]

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

    def _challenge_progress_payload(
        self,
        challenge: ActiveChallengeV4,
        status: str,
        feedback: str,
    ) -> dict[str, Any]:
        collected = (
            1 if challenge.internal_type == "BLINK" and challenge.blink_detected
            else challenge.pose_frames_passed
        )
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

    def _diagnostics(
        self,
        *,
        faces_detected: int,
        face_live: dict[str, Any],
        moire: dict[str, Any],
        rolling: dict[str, Any],
        screen_context: dict[str, Any],
        phone_rect: dict[str, Any],
        phone_rect_rolling: dict[str, Any],
        passive: Any,
        passive_status: str,
        match: Any,
    ) -> dict[str, Any]:
        return {
            "faces_detected": faces_detected,
            "liveness": face_live,
            "moire": {
                **moire,
                "decision": rolling.get("decision", moire.get("decision_hint", "")),
            },
            "moire_rolling": rolling,
            "screen_context": screen_context,
            "phone_rect": phone_rect,
            "phone_rect_rolling": phone_rect_rolling,
            "passive_liveness": {
                "score": round(float(passive.score), 3),
                "reason": passive.reason,
                "decision": passive_status,
            },
            "match": {
                "matched": bool(match.matched),
                "confidence": float(match.score),
                "student_id": match.student_id,
                "name": match.name,
                "threshold": V4_COSINE_THRESHOLD,
            },
            "scan_version": DETECT_VERSION,
        }

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

    def _state_event(
        self,
        live: dict[str, Any],
        *,
        status: str | None = None,
        message: str | None = None,
        **extra,
    ) -> dict[str, Any]:
        return {
            "type": "scan_state",
            "status": status or live.get("state", "unknown"),
            "message": message or live.get("message", ""),
            "liveness": live,
            "faces_detected": extra.pop("faces_detected", 0),
            "moire": extra.pop("moire", {}),
            "moire_rolling": extra.pop("moire_rolling", {}),
            "screen_context": extra.pop("screen_context", {}),
            "phone_rect": extra.pop("phone_rect", {}),
            "phone_rect_rolling": extra.pop("phone_rect_rolling", {}),
            "passive_liveness": extra.pop("passive_liveness", {}),
            "match": extra.pop("match", {}),
            "scan_version": DETECT_VERSION,
            "fps": self._fps(),
            **extra,
        }

    def _drop_inactive_tracks(self, active_track_ids: set[int]):
        for track_id in list(self.rolling_by_face.keys()):
            if track_id not in active_track_ids:
                self.rolling_by_face.pop(track_id, None)
                self.phone_rolling_by_face.pop(track_id, None)

    def _challenge_passed_valid(self, now: float, student_id: str) -> bool:
        return (
            bool(student_id)
            and self.last_challenge_student_id == student_id
            and (now - self.challenge_passed_at) < CHALLENGE_COOLDOWN
        )

    def _fps(self) -> float:
        if len(self.frame_timestamps) < 2:
            return 0.0
        elapsed = self.frame_timestamps[-1] - self.frame_timestamps[0]
        if elapsed <= 0:
            return 0.0
        return round((len(self.frame_timestamps) - 1) / elapsed, 1)
