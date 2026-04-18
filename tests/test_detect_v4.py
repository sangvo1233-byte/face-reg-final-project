"""
Focused Detect V4.4 backend tests.
"""
import asyncio
import os
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))


def test_moire_v4_empty_input():
    from core.detect_v4 import MoireDetectorV4

    detector = MoireDetectorV4()

    result = detector.analyze(None)
    assert result["moire_score"] == 0.5
    assert result["decision_hint"] == "clean"

    result = detector.analyze(np.array([]))
    assert result["moire_score"] == 0.5
    assert result["pipeline_mode"] == "enhanced"


def test_rolling_moire_uses_screen_threshold():
    from core.detect_v4 import MOIRE_SCREEN_THRESHOLD, RollingMoireDecision

    rolling = RollingMoireDecision()
    summary = rolling.update({"moire_score": MOIRE_SCREEN_THRESHOLD - 0.01, "decision_hint": "clean"})

    assert MOIRE_SCREEN_THRESHOLD == 0.60
    assert summary["decision"] == "suspicious"


def test_phone_rect_rolling_blocks_after_two_strong_samples():
    from core.detect_v4 import RollingPhoneRectDecision

    rolling = RollingPhoneRectDecision()
    rolling.update({"score": 0.60, "decision": "strong"})
    summary = rolling.update({"score": 0.61, "decision": "strong"})

    assert summary["decision"] == "block"
    assert summary["strong_count"] == 2


def test_collect_suspicious_reasons():
    from core.detect_v4 import collect_suspicious_reasons

    reasons = collect_suspicious_reasons(
        moire_decision="suspicious",
        screen_context={"decision": "strong"},
        phone_rect={"decision": "clean"},
        phone_rect_rolling={"decision": "suspicious"},
        passive_status="suspicious",
        passive_score=0.42,
        challenge_fail_count=2,
    )

    assert "moire suspicious" in reasons
    assert "context strong" in reasons
    assert "phone rectangle rolling" in reasons
    assert "passive 0.42" in reasons
    assert "recent challenge failures" in reasons


def test_assess_challenge_need_requires_two_suspicious_or_one_strong():
    from core.detect_v4 import assess_challenge_need

    none = assess_challenge_need(
        moire_decision="suspicious",
        screen_context={"decision": "clean"},
        phone_rect={"decision": "clean"},
        phone_rect_rolling={"decision": "clean"},
        passive_status="pass",
        passive_score=0.55,
        challenge_fail_count=0,
    )
    assert none["should_challenge"] is False
    assert none["severity"] == "none"

    medium = assess_challenge_need(
        moire_decision="suspicious",
        screen_context={"decision": "suspicious"},
        phone_rect={"decision": "clean"},
        phone_rect_rolling={"decision": "clean"},
        passive_status="pass",
        passive_score=0.55,
        challenge_fail_count=0,
    )
    assert medium["should_challenge"] is True
    assert medium["severity"] == "medium"

    strong = assess_challenge_need(
        moire_decision="clean",
        screen_context={"decision": "strong"},
        phone_rect={"decision": "clean"},
        phone_rect_rolling={"decision": "clean"},
        passive_status="pass",
        passive_score=0.55,
        challenge_fail_count=0,
    )
    assert strong["should_challenge"] is True
    assert strong["severity"] == "strong"
    assert "context strong" in strong["reasons"]


def test_runtime_diagnostics_expose_overlay_geometry():
    from core.runtime_v4 import DetectV4RuntimeSession

    runtime = DetectV4RuntimeSession.__new__(DetectV4RuntimeSession)
    passive = MagicMock(score=0.81, reason="ok")
    match = MagicMock(matched=True, score=0.88, student_id="HS001", name="Sample")

    diagnostics = runtime._diagnostics(
        frame_size={"width": 960, "height": 540},
        face_bbox=[100, 80, 220, 240],
        moire_roi_bbox=[70, 40, 250, 280],
        faces_detected=1,
        face_live={"state": "live"},
        moire={"moire_score": 0.73, "decision_hint": "clean"},
        rolling={"decision": "clean"},
        screen_context={"score": 0.12, "decision": "clean", "roi_bbox": [70, 40, 250, 280]},
        phone_rect={"score": 0.44, "decision": "suspicious", "roi_bbox": [20, 10, 320, 400]},
        phone_rect_rolling={"decision": "suspicious", "strong_count": 1},
        passive=passive,
        passive_status="pass",
        match=match,
    )

    assert diagnostics["frame_size"] == {"width": 960, "height": 540}
    assert diagnostics["face_bbox"] == [100, 80, 220, 240]
    assert diagnostics["moire_roi_bbox"] == [70, 40, 250, 280]
    assert diagnostics["phone_rect"]["roi_bbox"] == [20, 10, 320, 400]


def test_liveness_ignores_brightness_flicker_as_fake_blink():
    from core.liveness import MultiFrameLiveness

    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(388)]
    landmarks[1] = SimpleNamespace(x=0.5, y=0.5)
    for index, x, y in [
        (33, 0.42, 0.40), (160, 0.45, 0.38), (158, 0.47, 0.38),
        (133, 0.50, 0.40), (153, 0.47, 0.42), (144, 0.45, 0.42),
        (362, 0.50, 0.40), (385, 0.53, 0.38), (387, 0.55, 0.38),
        (263, 0.58, 0.40), (373, 0.55, 0.42), (380, 0.53, 0.42),
    ]:
        landmarks[index] = SimpleNamespace(x=x, y=y)

    tracker = MultiFrameLiveness.__new__(MultiFrameLiveness)
    tracker._landmarker = MagicMock()
    tracker._landmarker.detect_for_video.return_value = MagicMock(face_landmarks=[landmarks])
    tracker._available = True
    tracker._tracks = {}
    tracker._next_id = 0
    tracker._frame_ts = 0
    tracker._calc_ear = MagicMock(side_effect=[0.30, 0.30, 0.23, 0.23, 0.30, 0.30])

    for level in (120, 220, 120):
        frame = np.full((120, 120, 3), level, dtype=np.uint8)
        tracker.process_frame(frame)

    track = next(iter(tracker._tracks.values()))
    assert track.blink_count == 0


def test_liveness_counts_blink_with_shorter_closed_open_windows():
    from core.liveness import MultiFrameLiveness

    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(388)]
    landmarks[1] = SimpleNamespace(x=0.5, y=0.5)

    tracker = MultiFrameLiveness.__new__(MultiFrameLiveness)
    tracker._landmarker = MagicMock()
    tracker._landmarker.detect_for_video.return_value = MagicMock(face_landmarks=[landmarks])
    tracker._available = True
    tracker._tracks = {}
    tracker._next_id = 0
    tracker._frame_ts = 0
    tracker._calc_ear = MagicMock(side_effect=[0.30, 0.30, 0.20, 0.20, 0.30, 0.30, 0.30, 0.30])

    frame = np.full((120, 120, 3), 120, dtype=np.uint8)
    for _ in range(4):
        tracker.process_frame(frame)

    track = next(iter(tracker._tracks.values()))
    assert track.blink_count == 1


def test_liveness_no_longer_requires_motion_after_blink():
    import config
    from core.liveness import MultiFrameLiveness, _FaceTrack

    tracker = MultiFrameLiveness.__new__(MultiFrameLiveness)
    tracker._available = True

    track = _FaceTrack(first_seen=time.time() - 3.0)
    track.blink_count = 1
    track.frame_count = config.DETECT_V3_STREAM_LIVE_MIN_FRAMES
    track.last_ear = 0.30
    track.pos_history.extend([(100.0, 100.0)] * track.frame_count)

    status = tracker._status(track)

    assert status.is_live is True
    assert status.reason == "pass"


def test_liveness_exposes_challenge_metrics_from_landmarks():
    from core.liveness import MultiFrameLiveness

    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(388)]
    for index, x, y in [
        (1, 0.50, 0.50),
        (13, 0.50, 0.58),
        (14, 0.50, 0.62),
        (78, 0.43, 0.60),
        (308, 0.57, 0.60),
        (33, 0.42, 0.40), (160, 0.45, 0.38), (158, 0.47, 0.38),
        (133, 0.50, 0.40), (153, 0.47, 0.42), (144, 0.45, 0.42),
        (362, 0.50, 0.40), (385, 0.53, 0.38), (387, 0.55, 0.38),
        (263, 0.58, 0.40), (373, 0.55, 0.42), (380, 0.53, 0.42),
    ]:
        landmarks[index] = SimpleNamespace(x=x, y=y)

    tracker = MultiFrameLiveness.__new__(MultiFrameLiveness)
    tracker._landmarker = MagicMock()
    tracker._landmarker.detect_for_video.return_value = MagicMock(face_landmarks=[landmarks])
    tracker._available = True
    tracker._tracks = {}
    tracker._next_id = 0
    tracker._frame_ts = 0
    tracker._calc_ear = MagicMock(return_value=0.30)

    frame = np.full((120, 120, 3), 120, dtype=np.uint8)
    tracker.process_frame(frame)

    metrics = tracker.get_challenge_metrics([40, 30, 80, 90])

    assert metrics["available"] is True
    assert metrics["pitch_value"] > 0.0
    assert metrics["mouth_ratio"] > 0.0
    assert metrics["face_scale"] > 0.0


def test_v4_challenge_uses_medium_pool_for_medium_severity():
    from core.runtime_v4 import DetectV4RuntimeSession

    runtime = DetectV4RuntimeSession.__new__(DetectV4RuntimeSession)
    runtime.active_challenge = None
    runtime.last_challenge_student_id = None
    runtime.frame_timestamps = deque(maxlen=60)

    with patch("core.runtime_v4.random.choice", return_value=("TURN_LEFT", "OPEN_MOUTH")) as mock_choice:
        result = runtime._start_challenge(
            {"student_id": "HS001", "name": "Sample"},
            {"blinks": 0},
            {"frame_size": {"width": 640, "height": 480}},
            "moire suspicious",
            "medium",
        )

    assert runtime.active_challenge.type == "turn_left"
    assert runtime.active_challenge.steps == ("TURN_LEFT", "OPEN_MOUTH")
    assert result["challenge"]["type"] == "turn_left"
    assert result["challenge"]["step_total"] == 2
    assert result["challenge"]["step_number"] == 1
    assert "sequence" not in result["challenge"]
    assert mock_choice.call_args.args[0] == [
        ("TURN_LEFT", "OPEN_MOUTH"),
        ("TURN_RIGHT", "OPEN_MOUTH"),
        ("LOOK_UP", "TURN_LEFT"),
        ("LOOK_UP", "TURN_RIGHT"),
    ]


def test_v4_challenge_uses_strong_pool_for_strong_severity():
    from core.runtime_v4 import DetectV4RuntimeSession

    runtime = DetectV4RuntimeSession.__new__(DetectV4RuntimeSession)
    runtime.active_challenge = None
    runtime.last_challenge_student_id = None
    runtime.frame_timestamps = deque(maxlen=60)

    with patch("core.runtime_v4.random.choice", return_value=("CENTER_HOLD", "TURN_RIGHT")) as mock_choice:
        runtime._start_challenge(
            {"student_id": "HS001", "name": "Sample"},
            {"blinks": 0},
            {"frame_size": {"width": 640, "height": 480}},
            "context strong",
            "strong",
        )

    assert ("CENTER_HOLD", "TURN_RIGHT") in mock_choice.call_args.args[0]
    assert ("LOOK_DOWN", "OPEN_MOUTH") in mock_choice.call_args.args[0]


def test_v4_challenge_requires_recentering_before_second_step():
    from core.runtime_v4 import ActiveChallengeV4, DetectV4RuntimeSession

    runtime = DetectV4RuntimeSession.__new__(DetectV4RuntimeSession)
    challenge = ActiveChallengeV4(("TURN_LEFT", "OPEN_MOUTH"), {"student_id": "HS001"}, {"blinks": 0}, "test")
    challenge.neutral_yaw_baseline = 0.0
    challenge.neutral_pitch_baseline = 0.0
    challenge.begin_recenter()

    assert challenge.type == "center"
    assert runtime._check_recenter(challenge, 0.08, {"pitch_value": 0.0}) is False
    assert runtime._check_recenter(challenge, 0.01, {"pitch_value": 0.08}) is False
    assert runtime._check_recenter(challenge, 0.0, {"pitch_value": 0.01}) is False
    assert runtime._check_recenter(challenge, 0.0, {"pitch_value": 0.0}) is True

    challenge.advance_step()

    assert challenge.type == "open_mouth"
    assert challenge.step_number == 2


def test_v4_turn_challenge_requires_yaw_and_consecutive_hold():
    import core.detect_v4 as detect_v4
    from core.runtime_v4 import ActiveChallengeV4, DetectV4RuntimeSession

    runtime = DetectV4RuntimeSession.__new__(DetectV4RuntimeSession)
    challenge = ActiveChallengeV4(("TURN_LEFT",), {"student_id": "HS001"}, {"blinks": 0}, "test")
    challenge.pose_baseline = 0.0
    challenge.pitch_baseline = 0.5
    challenge.mouth_baseline = 0.08
    challenge.scale_baseline = 100.0
    challenge.pose_baseline_frames = detect_v4.CHALLENGE_BASELINE_FRAMES
    metrics = {
        "available": True,
        "lighting_cooldown": 0,
        "pitch_value": 0.5,
        "mouth_ratio": 0.08,
        "face_scale": 100.0,
    }

    assert runtime._check_action(challenge, 0.11, detect_v4.CHALLENGE_MIN_YAW_ANGLE - 1, metrics) is False
    assert challenge.pose_frames_passed == 0

    assert runtime._check_action(challenge, 0.11, detect_v4.CHALLENGE_MIN_YAW_ANGLE + 1, metrics) is False
    assert runtime._check_action(challenge, 0.11, detect_v4.CHALLENGE_MIN_YAW_ANGLE + 1, metrics) is False
    assert challenge.pose_frames_passed == detect_v4.CHALLENGE_POSE_FRAMES - 1

    assert runtime._check_action(challenge, 0.04, detect_v4.CHALLENGE_MIN_YAW_ANGLE + 2, metrics) is False
    assert challenge.pose_frames_passed == 0

    assert runtime._check_action(challenge, 0.11, detect_v4.CHALLENGE_MIN_YAW_ANGLE + 2, metrics) is False
    assert runtime._check_action(challenge, 0.11, detect_v4.CHALLENGE_MIN_YAW_ANGLE + 2, metrics) is False
    assert runtime._check_action(challenge, 0.11, detect_v4.CHALLENGE_MIN_YAW_ANGLE + 2, metrics) is True


def test_v4_look_up_requires_negative_pitch_delta_and_centered_yaw():
    import core.detect_v4 as detect_v4
    from core.runtime_v4 import ActiveChallengeV4, DetectV4RuntimeSession

    runtime = DetectV4RuntimeSession.__new__(DetectV4RuntimeSession)
    challenge = ActiveChallengeV4(("LOOK_UP",), {"student_id": "HS001"}, {"blinks": 0}, "test")
    challenge.pose_baseline = 0.0
    challenge.pitch_baseline = 0.50
    challenge.mouth_baseline = 0.08
    challenge.scale_baseline = 100.0
    challenge.neutral_yaw_baseline = 0.0
    challenge.pose_baseline_frames = detect_v4.CHALLENGE_BASELINE_FRAMES

    metrics = {
        "available": True,
        "lighting_cooldown": 0,
        "pitch_value": 0.40,
        "mouth_ratio": 0.08,
        "face_scale": 100.0,
    }
    assert runtime._check_action(challenge, 0.0, 2.0, metrics) is False
    assert runtime._check_action(challenge, 0.0, 2.0, metrics) is False
    assert runtime._check_action(challenge, 0.0, 2.0, metrics) is True
    assert runtime._check_action(challenge, 0.09, 2.0, metrics) is False


def test_v4_open_mouth_requires_mouth_delta_and_hold():
    from core.runtime_v4 import ActiveChallengeV4, DetectV4RuntimeSession
    import core.detect_v4 as detect_v4

    runtime = DetectV4RuntimeSession.__new__(DetectV4RuntimeSession)
    challenge = ActiveChallengeV4(("OPEN_MOUTH",), {"student_id": "HS001"}, {"blinks": 0}, "test")
    challenge.pose_baseline = 0.0
    challenge.pitch_baseline = 0.50
    challenge.mouth_baseline = 0.07
    challenge.scale_baseline = 100.0
    challenge.pose_baseline_frames = detect_v4.CHALLENGE_BASELINE_FRAMES

    weak_metrics = {
        "available": True,
        "lighting_cooldown": 0,
        "pitch_value": 0.50,
        "mouth_ratio": 0.12,
        "face_scale": 100.0,
    }
    strong_metrics = {
        "available": True,
        "lighting_cooldown": 0,
        "pitch_value": 0.50,
        "mouth_ratio": 0.24,
        "face_scale": 100.0,
    }

    assert runtime._check_action(challenge, 0.0, 1.0, weak_metrics) is False
    assert challenge.pose_frames_passed == 0
    assert runtime._check_action(challenge, 0.0, 1.0, strong_metrics) is False
    assert runtime._check_action(challenge, 0.0, 1.0, strong_metrics) is False
    assert runtime._check_action(challenge, 0.0, 1.0, strong_metrics) is True


def test_v4_center_hold_uses_neutral_baseline_not_step_baseline():
    import core.detect_v4 as detect_v4
    from core.runtime_v4 import ActiveChallengeV4, DetectV4RuntimeSession

    runtime = DetectV4RuntimeSession.__new__(DetectV4RuntimeSession)
    challenge = ActiveChallengeV4(("CENTER_HOLD",), {"student_id": "HS001"}, {"blinks": 0}, "test")
    challenge.pose_baseline = 0.09
    challenge.pitch_baseline = 0.58
    challenge.mouth_baseline = 0.08
    challenge.scale_baseline = 100.0
    challenge.neutral_yaw_baseline = 0.0
    challenge.neutral_pitch_baseline = 0.50
    challenge.neutral_scale_baseline = 100.0
    challenge.pose_baseline_frames = detect_v4.CHALLENGE_BASELINE_FRAMES

    off_center_metrics = {
        "available": True,
        "lighting_cooldown": 0,
        "pitch_value": 0.58,
        "mouth_ratio": 0.08,
        "face_scale": 100.0,
    }
    centered_metrics = {
        "available": True,
        "lighting_cooldown": 0,
        "pitch_value": 0.50,
        "mouth_ratio": 0.08,
        "face_scale": 100.0,
    }

    assert runtime._check_action(challenge, 0.09, 1.0, off_center_metrics) is False
    assert runtime._check_action(challenge, 0.0, 1.0, centered_metrics) is False
    assert runtime._check_action(challenge, 0.0, 1.0, centered_metrics) is False
    assert runtime._check_action(challenge, 0.0, 1.0, centered_metrics) is True


def test_system_capabilities_include_v4():
    from app.routes.system import system_capabilities

    data = asyncio.run(system_capabilities())

    assert "v4" in data["scan_versions"]
    assert "v4_ws" in data["scan_versions"]
    assert "local_direct_v4" in data["scan_versions"]
    assert data["features"]["detect_v4"] is True
    assert data["features"]["phone_rectangle"] is True
    assert data["features"]["portrait_phone_roi"] is True
    assert data["thresholds"]["v4_moire_screen"] == 0.60


def test_scan_v4_no_session():
    from app.routes.scan_v4 import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with patch("app.routes.scan_v4.get_db") as mock_get_db:
        db = MagicMock()
        db.get_active_session.return_value = None
        mock_get_db.return_value = db

        response = client.post(
            "/api/scan/v4",
            files={"image": ("frame.jpg", b"not-needed", "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "No active session"
    assert data["scan_version"] == "v4.4"
