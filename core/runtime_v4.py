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
    CHALLENGE_CENTER_HOLD_FRAMES,
    CHALLENGE_CENTER_MAX_YAW_ANGLE,
    CHALLENGE_COOLDOWN,
    CHALLENGE_MOUTH_FRAMES,
    CHALLENGE_MIN_YAW_ANGLE,
    CHALLENGE_POSE_FRAMES,
    CHALLENGE_RECENTER_PITCH_THRESHOLD,
    CHALLENGE_TIMEOUT,
    CHALLENGE_TYPES,
    CENTER_PITCH_THRESHOLD,
    CENTER_SCALE_DELTA,
    CENTER_YAW_THRESHOLD,
    LOOK_DOWN_THRESHOLD,
    LOOK_PITCH_CENTER_YAW_THRESHOLD,
    LOOK_UP_THRESHOLD,
    MOIRE_BLOCK_THRESHOLD,
    MOIRE_EVERY_N_DETECT,
    OPEN_MOUTH_DELTA,
    OPEN_MOUTH_MIN_RATIO,
    TURN_THRESHOLD,
    V4_COSINE_THRESHOLD,
    DETECT_VERSION,
    FaceMoireTrackStore,
    MoireDetectorV4,
    PhoneRectangleDetectorV42,
    RollingMoireDecision,
    RollingPhoneRectDecision,
    ScreenContextDetectorV41,
    assess_challenge_need,
    bbox_list,
    expanded_roi,
    get_detect_v4_service,
    passive_decision,
)
from core.face_engine import get_engine
from core.liveness import StreamingLivenessTracker


API_CHALLENGE_TYPES = {
    "CENTER": "center",
    "CENTER_HOLD": "center_hold",
    "TURN_LEFT": "turn_left",
    "TURN_RIGHT": "turn_right",
    "LOOK_UP": "look_up",
    "LOOK_DOWN": "look_down",
    "OPEN_MOUTH": "open_mouth",
}
CHALLENGE_LABELS = {
    "CENTER": "Return to Center",
    "CENTER_HOLD": "Center Hold",
    "TURN_LEFT": "Turn Left",
    "TURN_RIGHT": "Turn Right",
    "LOOK_UP": "Look Up",
    "LOOK_DOWN": "Look Down",
    "OPEN_MOUTH": "Open Mouth",
}
CHALLENGE_INSTRUCTIONS = {
    "CENTER": "Look straight at the camera before the next step",
    "CENTER_HOLD": "Look straight at the camera and hold still",
    "TURN_LEFT": "Turn your face to the LEFT",
    "TURN_RIGHT": "Turn your face to the RIGHT",
    "LOOK_UP": "Lift your chin and LOOK UP",
    "LOOK_DOWN": "Lower your chin and LOOK DOWN",
    "OPEN_MOUTH": "Open your mouth clearly",
}
MEDIUM_CHALLENGE_SEQUENCES = [
    ("TURN_LEFT", "OPEN_MOUTH"),
    ("TURN_RIGHT", "OPEN_MOUTH"),
    ("LOOK_UP", "TURN_LEFT"),
    ("LOOK_UP", "TURN_RIGHT"),
]
STRONG_CHALLENGE_SEQUENCES = [
    *MEDIUM_CHALLENGE_SEQUENCES,
    ("CENTER_HOLD", "TURN_RIGHT"),
    ("TURN_RIGHT", "LOOK_UP"),
    ("LOOK_UP", "OPEN_MOUTH"),
    ("LOOK_DOWN", "TURN_LEFT"),
    ("LOOK_DOWN", "OPEN_MOUTH"),
]
CHALLENGE_RECENTER_THRESHOLD = 0.035
CHALLENGE_RECENTER_FRAMES = 2


class ActiveChallengeV4:
    def __init__(self, challenge_steps: tuple[str, ...], candidate: dict[str, Any], live: dict[str, Any], reason: str):
        self.id = str(uuid.uuid4())[:8]
        self.steps = tuple(challenge_steps)
        self.candidate = candidate
        self.reason = reason
        self.started_at = time.time()
        self.expires_at = self.started_at + CHALLENGE_TIMEOUT
        self.frames_processed = 0
        self.identity_frames = 0
        self.step_index = 0
        self.awaiting_recenter = False
        self.recenter_frames = 0
        self.neutral_yaw_baseline: float | None = None
        self.neutral_pitch_baseline: float | None = None
        self.neutral_scale_baseline: float | None = None
        self.pose_baseline = None
        self.pitch_baseline: float | None = None
        self.mouth_baseline: float | None = None
        self.scale_baseline: float | None = None
        self.pose_baseline_frames = 0
        self.pose_frames_passed = 0

    @property
    def remaining_ms(self) -> int:
        return max(0, int((self.expires_at - time.time()) * 1000))

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.expires_at

    @property
    def current_internal_type(self) -> str:
        return self.steps[min(self.step_index, len(self.steps) - 1)]

    @property
    def display_internal_type(self) -> str:
        if self.awaiting_recenter:
            return "CENTER"
        return self.current_internal_type

    @property
    def type(self) -> str:
        return API_CHALLENGE_TYPES[self.display_internal_type]

    @property
    def label(self) -> str:
        return CHALLENGE_LABELS[self.display_internal_type]

    @property
    def instruction(self) -> str:
        return CHALLENGE_INSTRUCTIONS[self.display_internal_type]

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        return self.step_index + (1 if self.awaiting_recenter else 0)

    @property
    def step_number(self) -> int:
        if self.total_steps <= 1:
            return 1
        return min(
            self.step_index + 1 + (1 if self.awaiting_recenter and self.step_index < self.total_steps - 1 else 0),
            self.total_steps,
        )

    @property
    def required_frames(self) -> int:
        if self.awaiting_recenter:
            return CHALLENGE_RECENTER_FRAMES
        if self.current_internal_type == "OPEN_MOUTH":
            return CHALLENGE_MOUTH_FRAMES
        if self.current_internal_type == "CENTER_HOLD":
            return CHALLENGE_CENTER_HOLD_FRAMES
        return CHALLENGE_POSE_FRAMES

    @property
    def text(self) -> str:
        return self.label

    def reset_pose_tracking(self):
        self.pose_baseline = None
        self.pitch_baseline = None
        self.mouth_baseline = None
        self.scale_baseline = None
        self.pose_baseline_frames = 0
        self.pose_frames_passed = 0

    def begin_recenter(self):
        self.awaiting_recenter = True
        self.recenter_frames = 0
        self.reset_pose_tracking()

    def advance_step(self):
        self.awaiting_recenter = False
        self.recenter_frames = 0
        if self.step_index < self.total_steps - 1:
            self.step_index += 1
        self.reset_pose_tracking()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "instruction": self.instruction,
            "candidate": self.candidate,
            "step_number": self.step_number,
            "step_total": self.total_steps,
            "completed_steps": self.completed_steps,
            "phase": "recenter" if self.awaiting_recenter else "action",
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
        frame_size = self._frame_size(frame)
        live_summary = self.liveness.process_frame(frame)

        if self.active_challenge is not None:
            return self._process_challenge_frame(frame, live_summary, now)

        if now < self.detect_paused_until:
            return [self._state_event(
                live_summary,
                status="cooldown",
                message="Attendance recorded. Continue scanning...",
                frame_size=frame_size,
            )]

        if now - self.last_detect_at < self.detect_interval:
            return [self._state_event(live_summary, frame_size=frame_size)]

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
            return [self._state_event(
                live,
                status="waiting_face",
                message="Move your face into the frame",
                faces_detected=0,
                frame_size=self._frame_size(frame),
            )]

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

        return events or [self._state_event(
            live,
            status="unknown",
            message="No usable face embedding",
            faces_detected=len(faces),
            frame_size=self._frame_size(frame),
        )]

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
            frame_size=self._frame_size(frame),
            face_bbox=bbox,
            moire_roi_bbox=moire_roi_bbox,
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
        challenge_assessment = assess_challenge_need(
            moire_decision=moire_decision,
            screen_context=screen_context,
            phone_rect=phone_rect,
            phone_rect_rolling=phone_rect_rolling,
            passive_status=passive_status,
            passive_score=float(passive.score),
            challenge_fail_count=self.challenge_fail_count,
        )
        if challenge_assessment["should_challenge"] and not self._challenge_passed_valid(now, match.student_id):
            return self._start_challenge(
                candidate,
                face_live,
                diagnostics,
                "; ".join(challenge_assessment["reasons"]),
                challenge_assessment["severity"],
            )

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
            return [self._challenge_progress_event("No face detected", frame_size=self._frame_size(frame))]

        valid_faces = [f for f in faces if f.embedding is not None and len(f.embedding) > 0]
        if not valid_faces:
            return [self._challenge_progress_event(
                "No usable face embedding",
                frame_size=self._frame_size(frame),
            )]

        face = max(valid_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        bbox = bbox_list(face.bbox)
        match = self.engine.match_with_threshold(face.embedding, V4_COSINE_THRESHOLD)
        if not (match.matched and match.student_id == challenge.candidate.get("student_id")):
            return [self._challenge_progress_event(
                "Identity mismatch",
                frame_size=self._frame_size(frame),
                face_bbox=bbox,
            )]
        challenge.identity_frames += 1

        self.moire_tracks.begin_cycle()
        try:
            moire, rolling, moire_roi_bbox = self._moire_for_face(frame, bbox, True)
        finally:
            active_track_ids = self.moire_tracks.finish_cycle()
            self._drop_inactive_tracks(active_track_ids)

        if rolling.get("decision") == "block" or float(moire.get("moire_score", 1.0)) < MOIRE_BLOCK_THRESHOLD:
            return self._end_challenge("failed", "Screen detected during challenge")

        face_live = self.liveness.get_liveness(bbox)
        challenge_metrics = self.liveness.get_challenge_metrics(bbox)
        metrics = self.engine.get_face_metrics(frame, face)
        nose_x_disp = float(metrics.get("nose_x_disp", 0.0))
        progress_extra = {
            "frame_size": self._frame_size(frame),
            "face_bbox": bbox,
            "liveness": face_live,
            "moire": {
                **moire,
                "decision": rolling.get("decision", moire.get("decision_hint", "")),
            },
            "moire_rolling": rolling,
            "moire_roi_bbox": moire_roi_bbox,
        }

        if challenge.awaiting_recenter:
            recentered = self._check_recenter(challenge, nose_x_disp, challenge_metrics)
            if recentered:
                challenge.advance_step()
                return [self._challenge_progress_event(
                    f"Step {challenge.step_number}/{challenge.total_steps}: {challenge.label}",
                    **progress_extra,
                )]
            return [self._challenge_progress_event(
                "Return to center before the next step",
                **progress_extra,
            )]

        completed = self._check_action(
            challenge,
            nose_x_disp,
            float(metrics.get("yaw_angle", 0.0)),
            challenge_metrics,
        )

        if completed:
            if challenge.step_index < challenge.total_steps - 1:
                challenge.begin_recenter()
                return [self._challenge_progress_event(
                    "Step complete. Return to center",
                    **progress_extra,
                )]
            return self._end_challenge_pass(frame, match, face, now, moire, face_live)
        return [self._challenge_progress_event(
            f"Keep going: {challenge.label}",
            **progress_extra,
        )]

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
        severity: str,
    ) -> dict[str, Any]:
        challenge_pool = (
            STRONG_CHALLENGE_SEQUENCES
            if severity == "strong"
            else MEDIUM_CHALLENGE_SEQUENCES
        )
        challenge_pool = challenge_pool or [(challenge,) for challenge in CHALLENGE_TYPES]
        challenge_steps = tuple(random.choice(challenge_pool))
        self.active_challenge = ActiveChallengeV4(challenge_steps, candidate, live, reason)
        self.last_challenge_student_id = candidate.get("student_id")
        logger.info(
            f"Detect V4 {severity} challenge started: {' -> '.join(challenge_steps)} "
            f"for {candidate.get('student_id')}: {reason}"
        )
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

    def _collect_action_baseline(
        self,
        challenge: ActiveChallengeV4,
        nose_x_disp: float,
        challenge_metrics: dict[str, Any],
    ) -> bool:
        pitch_value = float(challenge_metrics.get("pitch_value", 0.0))
        mouth_ratio = float(challenge_metrics.get("mouth_ratio", 0.0))
        face_scale = float(challenge_metrics.get("face_scale", 0.0))

        challenge.pose_baseline = (
            nose_x_disp
            if challenge.pose_baseline is None
            else challenge.pose_baseline * 0.7 + nose_x_disp * 0.3
        )
        challenge.pitch_baseline = (
            pitch_value
            if challenge.pitch_baseline is None
            else challenge.pitch_baseline * 0.7 + pitch_value * 0.3
        )
        challenge.mouth_baseline = (
            mouth_ratio
            if challenge.mouth_baseline is None
            else challenge.mouth_baseline * 0.8 + mouth_ratio * 0.2
        )
        challenge.scale_baseline = (
            face_scale
            if challenge.scale_baseline is None
            else challenge.scale_baseline * 0.8 + face_scale * 0.2
        )

        if challenge.neutral_yaw_baseline is None:
            challenge.neutral_yaw_baseline = challenge.pose_baseline
        if challenge.neutral_pitch_baseline is None:
            challenge.neutral_pitch_baseline = challenge.pitch_baseline
        if challenge.neutral_scale_baseline is None:
            challenge.neutral_scale_baseline = challenge.scale_baseline

        challenge.pose_baseline_frames += 1
        return challenge.pose_baseline_frames >= CHALLENGE_BASELINE_FRAMES

    def _check_action(
        self,
        challenge: ActiveChallengeV4,
        nose_x_disp: float,
        yaw_angle: float,
        challenge_metrics: dict[str, Any],
    ) -> bool:
        if not bool(challenge_metrics.get("available")) or int(challenge_metrics.get("lighting_cooldown", 0)) > 0:
            challenge.pose_frames_passed = 0
            return False
        if challenge.pose_baseline_frames < CHALLENGE_BASELINE_FRAMES:
            self._collect_action_baseline(challenge, nose_x_disp, challenge_metrics)
            return False

        baseline_yaw = float(challenge.pose_baseline or 0.0)
        baseline_pitch = float(challenge.pitch_baseline or 0.0)
        baseline_mouth = float(challenge.mouth_baseline or 0.0)
        baseline_scale = float(challenge.scale_baseline or 0.0)
        neutral_yaw = float(
            challenge.neutral_yaw_baseline
            if challenge.neutral_yaw_baseline is not None
            else baseline_yaw
        )
        neutral_pitch = float(
            challenge.neutral_pitch_baseline
            if challenge.neutral_pitch_baseline is not None
            else baseline_pitch
        )
        neutral_scale = float(
            challenge.neutral_scale_baseline
            if challenge.neutral_scale_baseline is not None
            else baseline_scale
        )

        pitch_value = float(challenge_metrics.get("pitch_value", 0.0))
        mouth_ratio = float(challenge_metrics.get("mouth_ratio", 0.0))
        face_scale = float(challenge_metrics.get("face_scale", 0.0))

        dx = nose_x_disp - baseline_yaw
        pitch_delta = pitch_value - baseline_pitch
        mouth_delta = max(0.0, mouth_ratio - baseline_mouth)
        neutral_scale_delta = 0.0
        if neutral_scale > 1e-6:
            neutral_scale_delta = abs((face_scale - neutral_scale) / neutral_scale)

        action = challenge.current_internal_type
        yaw_ok = abs(yaw_angle) >= CHALLENGE_MIN_YAW_ANGLE
        pitch_centered = abs(nose_x_disp - neutral_yaw) <= LOOK_PITCH_CENTER_YAW_THRESHOLD
        valid = False
        if action == "TURN_LEFT":
            valid = dx >= TURN_THRESHOLD and yaw_ok
        elif action == "TURN_RIGHT":
            valid = dx <= -TURN_THRESHOLD and yaw_ok
        elif action == "LOOK_UP":
            valid = pitch_delta <= -LOOK_UP_THRESHOLD and pitch_centered
        elif action == "LOOK_DOWN":
            valid = pitch_delta >= LOOK_DOWN_THRESHOLD and pitch_centered
        elif action == "OPEN_MOUTH":
            valid = mouth_delta >= OPEN_MOUTH_DELTA or mouth_ratio >= OPEN_MOUTH_MIN_RATIO
        elif action == "CENTER_HOLD":
            valid = (
                abs(nose_x_disp - neutral_yaw) <= CENTER_YAW_THRESHOLD
                and abs(pitch_value - neutral_pitch) <= CENTER_PITCH_THRESHOLD
                and abs(yaw_angle) <= CHALLENGE_CENTER_MAX_YAW_ANGLE
                and neutral_scale_delta <= CENTER_SCALE_DELTA
            )

        if valid:
            challenge.pose_frames_passed += 1
        else:
            challenge.pose_frames_passed = 0
        return challenge.pose_frames_passed >= challenge.required_frames

    def _check_recenter(
        self,
        challenge: ActiveChallengeV4,
        nose_x_disp: float,
        challenge_metrics: dict[str, Any],
    ) -> bool:
        neutral_yaw = float(challenge.neutral_yaw_baseline or 0.0)
        neutral_pitch = float(challenge.neutral_pitch_baseline or 0.0)
        pitch_value = float(challenge_metrics.get("pitch_value", 0.0))

        yaw_centered = abs(nose_x_disp - neutral_yaw) <= CHALLENGE_RECENTER_THRESHOLD
        pitch_centered = abs(pitch_value - neutral_pitch) <= CHALLENGE_RECENTER_PITCH_THRESHOLD
        if yaw_centered and pitch_centered:
            challenge.recenter_frames += 1
        else:
            challenge.recenter_frames = 0
        return challenge.recenter_frames >= CHALLENGE_RECENTER_FRAMES

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
            "frame_size": self._frame_size(frame),
            "face_bbox": bbox,
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

    def _challenge_progress_event(self, feedback: str, **extra) -> dict[str, Any]:
        challenge = self.active_challenge
        return {
            "type": "scan_state",
            "status": "challenge_active",
            "message": feedback,
            "challenge": challenge.to_dict() if challenge else {},
            "challenge_progress": self._challenge_progress_payload(challenge, "collecting", feedback) if challenge else {},
            "fps": self._fps(),
            "scan_version": DETECT_VERSION,
            **extra,
        }

    def _challenge_progress_payload(
        self,
        challenge: ActiveChallengeV4,
        status: str,
        feedback: str,
    ) -> dict[str, Any]:
        if challenge.awaiting_recenter:
            collected = challenge.recenter_frames
            progress_ratio = challenge.completed_steps / max(challenge.total_steps, 1)
        else:
            collected = challenge.pose_frames_passed
            progress_ratio = (
                challenge.step_index + min(collected / max(challenge.required_frames, 1), 1.0)
            ) / max(challenge.total_steps, 1)
        return {
            "id": challenge.id,
            "type": challenge.type,
            "label": challenge.label,
            "instruction": challenge.instruction,
            "status": status,
            "feedback": feedback,
            "step_number": challenge.step_number,
            "step_total": challenge.total_steps,
            "completed_steps": challenge.completed_steps,
            "collected_frames": collected,
            "required_frames": challenge.required_frames,
            "progress_pct": int(round(progress_ratio * 100)),
            "frames_processed": challenge.frames_processed,
            "identity_frames": challenge.identity_frames,
            "elapsed_ms": int(max(0.0, (time.time() - challenge.started_at) * 1000)),
            "remaining_ms": challenge.remaining_ms,
        }

    def _diagnostics(
        self,
        *,
        frame_size: dict[str, int],
        face_bbox: list[int],
        moire_roi_bbox: list[int],
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
            "frame_size": frame_size,
            "face_bbox": face_bbox,
            "moire_roi_bbox": moire_roi_bbox,
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

    @staticmethod
    def _frame_size(frame: np.ndarray) -> dict[str, int]:
        height, width = frame.shape[:2]
        return {"width": int(width), "height": int(height)}

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
