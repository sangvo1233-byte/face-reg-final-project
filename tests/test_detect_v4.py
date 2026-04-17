"""
Focused Detect V4.4 backend tests.
"""
import asyncio
import os
import sys
from pathlib import Path

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
