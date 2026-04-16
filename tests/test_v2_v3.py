"""
Tests for Enroll V2 + Detect V3 integration.

Covers:
  - MoireDetector (real image, empty input)
  - FaceEngine helpers (match_with_threshold, get_face_metrics)
  - EnrollmentV2Service (with mock engine/db)
  - DetectV3Service (spoof + match scenarios)
  - API routes (FastAPI TestClient)
"""
import sys
import os
import io
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))

import pytest


# ═══════════════════════════════════════════════════════════
#  PHASE 1: MoireDetector Tests
# ═══════════════════════════════════════════════════════════

class TestMoireDetector:
    """Test MoireDetector FFT analysis."""

    def test_empty_input(self):
        """MoireDetector handles None/empty input gracefully."""
        from core.moire import MoireDetector
        detector = MoireDetector()

        result = detector.analyze(None)
        assert result["moire_score"] == 0.5
        assert result["is_screen"] is None

        empty = np.array([])
        result = detector.analyze(empty)
        assert result["moire_score"] == 0.5

    def test_synthetic_real_face(self):
        """Synthetic smooth gradient should score as 'real' (no moiré)."""
        from core.moire import MoireDetector
        detector = MoireDetector()

        # Create a smooth gradient image (simulates real face — organic spectrum)
        face_roi = np.zeros((128, 128, 3), dtype=np.uint8)
        for i in range(128):
            for j in range(128):
                face_roi[i, j] = [
                    int(100 + 50 * np.sin(i / 20.0)),
                    int(120 + 30 * np.cos(j / 25.0)),
                    int(110 + 40 * np.sin((i + j) / 30.0)),
                ]

        result = detector.analyze_single(face_roi)
        assert "moire_score" in result
        assert "is_screen" in result
        assert "peak_ratio" in result
        assert "energy_ratio" in result
        assert "periodicity" in result
        # Smooth gradient should not trigger moiré detection
        assert result["moire_score"] >= 0.0
        assert result["moire_score"] <= 1.0

    def test_analyze_single_no_history_leak(self):
        """analyze_single should not pollute rolling history."""
        from core.moire import MoireDetector
        detector = MoireDetector()

        face_roi = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        # First, populate some history
        detector.analyze(face_roi)
        history_before = len(detector.history)

        # analyze_single should not change history
        detector.analyze_single(face_roi)
        assert len(detector.history) == history_before

    def test_reset(self):
        """reset() clears history and last_result."""
        from core.moire import MoireDetector
        detector = MoireDetector()

        face_roi = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        detector.analyze(face_roi)
        assert len(detector.history) > 0

        detector.reset()
        assert len(detector.history) == 0
        assert detector.last_result == {}

    def test_singleton(self):
        """get_moire_detector returns singleton."""
        from core.moire import get_moire_detector
        d1 = get_moire_detector()
        d2 = get_moire_detector()
        assert d1 is d2


# ═══════════════════════════════════════════════════════════
#  PHASE 1: FaceEngine Helper Tests
# ═══════════════════════════════════════════════════════════

class TestFaceEngineHelpers:
    """Test new FaceEngine helpers: match_with_threshold, get_face_metrics."""

    def _make_engine_with_cache(self):
        """Create a FaceEngine with a fake embedding cache."""
        from core.face_engine import FaceEngine

        engine = FaceEngine()
        # Inject fake cache directly (bypass DB)
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 /= np.linalg.norm(emb1)
        emb2 = np.random.randn(512).astype(np.float32)
        emb2 /= np.linalg.norm(emb2)

        engine._embeddings = np.stack([emb1, emb2])
        engine._identities = [
            {"student_id": "S001", "name": "Alice"},
            {"student_id": "S002", "name": "Bob"},
        ]
        engine._cache_loaded = True
        return engine, emb1, emb2

    def test_match_with_threshold_above(self):
        """match_with_threshold succeeds when score >= threshold."""
        engine, emb1, _ = self._make_engine_with_cache()

        # Match with the exact same embedding — score should be ~1.0
        result = engine.match_with_threshold(emb1, threshold=0.5)
        assert result.matched is True
        assert result.student_id == "S001"
        assert result.score >= 0.99

    def test_match_with_threshold_below(self):
        """match_with_threshold fails when threshold is very high."""
        engine, emb1, _ = self._make_engine_with_cache()

        # Random embedding — score will be much lower than 0.99
        random_emb = np.random.randn(512).astype(np.float32)
        random_emb /= np.linalg.norm(random_emb)

        result = engine.match_with_threshold(random_emb, threshold=0.99)
        assert result.matched is False

    def test_match_with_threshold_empty_cache(self):
        """match_with_threshold returns unmatched when cache is empty."""
        from core.face_engine import FaceEngine
        engine = FaceEngine()
        engine._embeddings = np.empty((0, 512))
        engine._identities = []
        engine._cache_loaded = True

        emb = np.random.randn(512).astype(np.float32)
        result = engine.match_with_threshold(emb, threshold=0.3)
        assert result.matched is False
        assert result.score == 0.0

    def test_get_face_metrics_returns_expected_keys(self):
        """get_face_metrics returns dict with all expected keys."""
        from core.face_engine import FaceEngine
        from core.schemas import DetectedFace

        engine = FaceEngine()
        # Create a simple test face with landmarks
        face = DetectedFace(
            bbox=np.array([100, 100, 300, 300]),
            landmarks=np.array([
                [150, 180],  # left eye
                [250, 180],  # right eye
                [200, 220],  # nose
                [160, 260],  # left mouth
                [240, 260],  # right mouth
            ], dtype=np.float32),
            confidence=0.99,
            aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
            embedding=np.random.randn(512).astype(np.float32),
        )

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        metrics = engine.get_face_metrics(frame, face)

        assert "passed" in metrics
        assert "face_size" in metrics
        assert "blur_score" in metrics
        assert "brightness" in metrics
        assert "yaw_angle" in metrics
        assert "nose_x_disp" in metrics
        assert "reasons" in metrics

    def test_get_face_metrics_centered_nose(self):
        """Centered nose should have near-zero displacement."""
        from core.face_engine import FaceEngine
        from core.schemas import DetectedFace

        engine = FaceEngine()
        face = DetectedFace(
            bbox=np.array([100, 100, 300, 300]),
            landmarks=np.array([
                [150, 180],  # left eye
                [250, 180],  # right eye
                [200, 220],  # nose (centered between eyes)
                [160, 260],
                [240, 260],
            ], dtype=np.float32),
            confidence=0.99,
            aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
        )

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        metrics = engine.get_face_metrics(frame, face)

        # Nose is exactly centered, displacement should be 0
        assert abs(metrics["nose_x_disp"]) < 0.01


# ═══════════════════════════════════════════════════════════
#  PHASE 2: EnrollmentV2Service Tests
# ═══════════════════════════════════════════════════════════

class TestEnrollmentV2Service:
    """Test EnrollmentV2Service with mock engine/db."""

    def _make_mock_face(self, nose_x_disp=0.0):
        """Create a mock DetectedFace."""
        from core.schemas import DetectedFace

        # Position nose based on desired displacement
        eye_left_x = 150
        eye_right_x = 250
        eye_center = (eye_left_x + eye_right_x) / 2
        eye_dist = abs(eye_right_x - eye_left_x)
        nose_x = eye_center + nose_x_disp * eye_dist

        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        return DetectedFace(
            bbox=np.array([100, 100, 300, 300]),
            landmarks=np.array([
                [eye_left_x, 180],
                [eye_right_x, 180],
                [nose_x, 220],
                [160, 260],
                [240, 260],
            ], dtype=np.float32),
            confidence=0.99,
            aligned_face=np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8),
            embedding=emb,
        )

    @patch("core.enrollment_v2.get_db")
    @patch("core.enrollment_v2.get_engine")
    def test_enroll_success(self, mock_get_engine, mock_get_db):
        """Successful 3-angle enrollment saves 3 embeddings."""
        from core.enrollment_v2 import EnrollmentV2Service

        # Mock engine
        engine = MagicMock()
        engine._ensure_model = MagicMock()

        # Return centered face for front, shifted for left/right
        def mock_detect_largest(image):
            # Use a simple heuristic: check the image mean to differentiate
            return self._make_mock_face(nose_x_disp=0.0)

        engine.detect_largest = mock_detect_largest

        def mock_get_face_metrics(frame, face):
            return {
                "passed": True,
                "face_size": 200,
                "blur_score": 150.0,  # above ENROLL_V2_BLUR_MIN (80)
                "brightness": 120.0,
                "yaw_angle": 5.0,
                "nose_x_disp": 0.0,
                "reasons": [],
            }

        engine.get_face_metrics = mock_get_face_metrics
        engine.reload_cache = MagicMock()
        mock_get_engine.return_value = engine

        # Mock DB
        db = MagicMock()
        db.replace_student_embeddings.return_value = {"saved": 3, "old_count": 0}
        mock_get_db.return_value = db

        service = EnrollmentV2Service()
        images = {
            "front": np.zeros((480, 640, 3), dtype=np.uint8),
            "left": np.zeros((480, 640, 3), dtype=np.uint8),
            "right": np.zeros((480, 640, 3), dtype=np.uint8),
        }

        # Override _verify_pose to always pass for this test
        original_verify = service._verify_pose
        service._verify_pose = lambda disp, verify: (True, "")

        result = service.enroll_multi_angle("HS001", "Test User", "12A1", images)

        service._verify_pose = original_verify

        assert result["success"] is True
        assert result["total_saved"] == 3
        assert db.replace_student_embeddings.call_count == 1
        assert len(db.replace_student_embeddings.call_args.kwargs["embeddings"]) == 3

    @patch("core.enrollment_v2.get_db")
    @patch("core.enrollment_v2.get_engine")
    def test_enroll_missing_images(self, mock_get_engine, mock_get_db):
        """Missing images returns failure."""
        from core.enrollment_v2 import EnrollmentV2Service

        service = EnrollmentV2Service()
        result = service.enroll_multi_angle(
            "HS001", "Test", "", {"front": np.zeros((100, 100, 3), dtype=np.uint8)}
        )

        assert result["success"] is False
        assert "Missing" in result["message"]

    def test_verify_pose_center(self):
        """Verify pose: centered face passes center check."""
        from core.enrollment_v2 import EnrollmentV2Service
        service = EnrollmentV2Service()

        ok, reason = service._verify_pose(0.0, "center")
        assert ok is True

        ok, reason = service._verify_pose(0.15, "center")
        assert ok is False

    def test_verify_pose_left(self):
        """Verify pose: face turned left passes left check."""
        from core.enrollment_v2 import EnrollmentV2Service
        service = EnrollmentV2Service()

        # User-facing LEFT in the enrollment camera feed means positive displacement.
        ok, reason = service._verify_pose(0.08, "left")
        assert ok is True

        ok, reason = service._verify_pose(0.0, "left")
        assert ok is False

    def test_verify_pose_right(self):
        """Verify pose: face turned right passes right check."""
        from core.enrollment_v2 import EnrollmentV2Service
        service = EnrollmentV2Service()

        ok, reason = service._verify_pose(-0.08, "right")
        assert ok is True

        ok, reason = service._verify_pose(0.0, "right")
        assert ok is False

    @patch("core.enrollment_v2.get_db")
    @patch("core.enrollment_v2.get_engine")
    def test_enroll_multi_frame_average(self, mock_get_engine, mock_get_db):
        """Multi-frame enrollment averages good embeddings per angle."""
        from core.enrollment_v2 import EnrollmentV2Service

        engine = MagicMock()
        engine._ensure_model = MagicMock()
        engine.reload_cache = MagicMock()
        mock_get_engine.return_value = engine

        db = MagicMock()
        db.replace_student_embeddings.return_value = {"saved": 3, "old_count": 0}
        mock_get_db.return_value = db

        service = EnrollmentV2Service()

        def fake_process_angle(_engine, image, phase):
            emb = np.zeros(512, dtype=np.float32)
            emb[int(image[0, 0, 0])] = 1.0
            return {
                "name": phase["name"],
                "success": True,
                "reason": "",
                "embedding": emb,
                "quality": 120.0,
                "blur_score": 120.0,
                "best_aligned": None,
            }

        service._process_angle = fake_process_angle
        images = {
            "front": [np.full((4, 4, 3), i, dtype=np.uint8) for i in (0, 1, 2)],
            "left": [np.full((4, 4, 3), i, dtype=np.uint8) for i in (3, 4, 5)],
            "right": [np.full((4, 4, 3), i, dtype=np.uint8) for i in (6, 7, 8)],
        }

        result = service.enroll_multi_angle("HS002", "Multi", "12A1", images)

        assert result["success"] is True
        assert result["phase_results"][0]["frame_count"] == 3
        assert result["phase_results"][0]["frames_required"] == 3
        assert db.replace_student_embeddings.call_count == 1
        saved_embedding = db.replace_student_embeddings.call_args.kwargs["embeddings"][0]["embedding"]
        assert np.isclose(np.linalg.norm(saved_embedding), 1.0)

    def test_replace_student_embeddings_rolls_back_on_failure(self, monkeypatch):
        """Embedding replacement keeps old data if the transaction fails."""
        import config
        from core.database import Database

        db_path = Path("database") / "test_tx.db"
        db_path.unlink(missing_ok=True)
        monkeypatch.setattr(config, "SQLITE_DB_PATH", db_path)
        db = Database()
        old_emb = np.ones(512, dtype=np.float32)
        old_emb /= np.linalg.norm(old_emb)

        db.add_student("HS_TX", "Old Name", "12A1")
        db.save_embedding("HS_TX", old_emb, 99.0, "old")

        with pytest.raises(KeyError):
            db.replace_student_embeddings(
                student_id="HS_TX",
                name="New Name",
                class_name="12A2",
                embeddings=[
                    {"embedding": old_emb, "quality": 100.0, "source": "new_front"},
                    {"quality": 100.0, "source": "broken"},
                ],
            )

        assert db.get_embedding_count("HS_TX") == 1
        assert db.get_student_any("HS_TX")["name"] == "Old Name"
        db_path.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════
#  PHASE 2: DetectV3Service Tests
# ═══════════════════════════════════════════════════════════

class TestDetectV3Service:
    """Test DetectV3Service with mock engine/db/moiré."""

    @patch("core.detect_v3.get_db")
    @patch("core.detect_v3.get_anti_spoof")
    @patch("core.detect_v3.get_moire_detector")
    @patch("core.detect_v3.get_engine")
    def test_scan_spoof_via_moire(self, mock_engine, mock_moire, mock_anti_spoof, mock_db):
        """Moiré-detected screen face is marked as spoof."""
        from core.detect_v3 import DetectV3Service
        from core.schemas import DetectedFace, MatchResult

        # Mock engine
        engine = MagicMock()
        engine._ensure_model = MagicMock()

        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        face = DetectedFace(
            bbox=np.array([100, 100, 300, 300]),
            landmarks=None,
            confidence=0.99,
            aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
            embedding=emb,
        )
        engine.detect.return_value = [face]
        mock_engine.return_value = engine

        # Mock moiré → screen detected
        detector = MagicMock()
        detector.analyze_single.return_value = {
            "moire_score": 0.2,
            "is_screen": True,
            "peak_ratio": 4.0,
            "energy_ratio": 0.5,
            "periodicity": 5.0,
        }
        mock_moire.return_value = detector

        service = DetectV3Service()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = service.scan_attendance(frame, session_id=1)

        assert result["faces_detected"] == 1
        assert result["recognized"] == 0
        assert result["results"][0]["status"] == "spoof"
        assert "Screen" in result["results"][0]["message"]

    @patch("core.detect_v3.get_db")
    @patch("core.detect_v3.get_anti_spoof")
    @patch("core.detect_v3.get_moire_detector")
    @patch("core.detect_v3.get_engine")
    def test_scan_match_success(self, mock_engine, mock_moire, mock_anti_spoof, mock_db):
        """Real face that matches is recorded as present."""
        from core.detect_v3 import DetectV3Service
        from core.schemas import DetectedFace, MatchResult, LivenessResult

        engine = MagicMock()
        engine._ensure_model = MagicMock()

        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        face = DetectedFace(
            bbox=np.array([100, 100, 300, 300]),
            landmarks=None,
            confidence=0.99,
            aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
            embedding=emb,
        )
        engine.detect.return_value = [face]
        engine.match_with_threshold.return_value = MatchResult(
            matched=True, name="Alice", student_id="S001", score=0.85, index=0
        )
        mock_engine.return_value = engine

        # Mock moiré → real face
        detector = MagicMock()
        detector.analyze_single.return_value = {
            "moire_score": 0.85,
            "is_screen": False,
        }
        mock_moire.return_value = detector

        # Mock liveness → live
        anti_spoof = MagicMock()
        anti_spoof.check.return_value = LivenessResult(is_live=True, score=1.0)
        mock_anti_spoof.return_value = anti_spoof

        # Mock DB
        db = MagicMock()
        db.mark_attendance.return_value = {"success": True, "message": "Alice - Present"}
        db.get_embedding_count.return_value = 3
        mock_db.return_value = db

        service = DetectV3Service()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = service.scan_attendance(frame, session_id=1)

        assert result["faces_detected"] == 1
        assert result["recognized"] == 1
        assert result["results"][0]["status"] == "present"
        assert result["results"][0]["student_id"] == "S001"
        assert result["results"][0]["embedding_count"] == 3
        assert result["scan_version"] == "v3"

    @patch("core.detect_v3.get_challenge_v3_service")
    @patch("core.detect_v3.get_db")
    @patch("core.detect_v3.get_anti_spoof")
    @patch("core.detect_v3.get_moire_detector")
    @patch("core.detect_v3.get_engine")
    def test_scan_suspicious_match_requires_challenge(
        self,
        mock_engine,
        mock_moire,
        mock_anti_spoof,
        mock_db,
        mock_challenge_service,
    ):
        """Suspicious-but-not-blocked frames require challenge before attendance."""
        from core.detect_v3 import DetectV3Service
        from core.schemas import DetectedFace, MatchResult, LivenessResult

        engine = MagicMock()
        engine._ensure_model = MagicMock()

        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        face = DetectedFace(
            bbox=np.array([100, 100, 300, 300]),
            landmarks=None,
            confidence=0.99,
            aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
            embedding=emb,
        )
        engine.detect.return_value = [face]
        engine.match_with_threshold.return_value = MatchResult(
            matched=True, name="Alice", student_id="S001", score=0.85, index=0
        )
        mock_engine.return_value = engine

        detector = MagicMock()
        detector.analyze_single.return_value = {
            "moire_score": 0.4,
            "is_screen": False,
            "peak_ratio": 2.0,
            "periodicity": 1.0,
        }
        mock_moire.return_value = detector

        anti_spoof = MagicMock()
        anti_spoof.check.return_value = LivenessResult(is_live=True, score=1.0)
        mock_anti_spoof.return_value = anti_spoof

        db = MagicMock()
        db.get_student.return_value = {"id": "S001", "name": "Alice", "class_name": "12A1"}
        db.get_embedding_count.return_value = 3
        db.get_session.return_value = {"id": 1, "name": "Math"}
        mock_db.return_value = db

        challenge_service = MagicMock()
        challenge_service.create_challenge.return_value = {
            "id": "challenge-1",
            "type": "turn_left",
            "label": "Turn Left",
            "instruction": "Turn your face to the LEFT",
            "expires_in": 10,
            "required_frames": 3,
        }
        mock_challenge_service.return_value = challenge_service

        service = DetectV3Service()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = service.scan_attendance(frame, session_id=1)

        assert result["recognized"] == 0
        assert result["results"][0]["status"] == "challenge_required"
        assert result["results"][0]["challenge"]["id"] == "challenge-1"
        db.mark_attendance.assert_not_called()


# ═══════════════════════════════════════════════════════════
#  PHASE 3: API Route Tests
# ═══════════════════════════════════════════════════════════

class TestAPIRoutes:
    """Test API endpoints with FastAPI TestClient."""

    def test_capabilities_endpoint(self):
        """GET /api/system/capabilities returns expected structure."""
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        response = client.get("/api/system/capabilities")

        assert response.status_code == 200
        data = response.json()

        assert "enroll_versions" in data
        assert "v1" in data["enroll_versions"]
        assert "v2" in data["enroll_versions"]

        assert "scan_versions" in data
        assert "v1" in data["scan_versions"]
        assert "v3" in data["scan_versions"]

        assert "features" in data
        assert data["features"]["moire_detection"] is True
        assert data["features"]["multi_angle_enrollment"] is True

        assert "thresholds" in data
        assert data["thresholds"]["v3_cosine"] == 0.52

    @patch("app.routes.scan_v3.get_detect_v3_service")
    @patch("app.routes.scan_v3.get_db")
    def test_scan_v3_no_session(self, mock_db, mock_service):
        """POST /api/scan/v3 without active session returns error."""
        from fastapi.testclient import TestClient
        from main import app

        db = MagicMock()
        db.get_active_session.return_value = None
        mock_db.return_value = db

        client = TestClient(app)

        # Create a simple test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', img)
        response = client.post(
            "/api/scan/v3",
            files={"image": ("test.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    @patch("app.routes.enrollment_v2.get_enrollment_v2_service")
    def test_enroll_v2_endpoint(self, mock_service):
        """POST /api/enroll/v2 calls EnrollmentV2Service."""
        from fastapi.testclient import TestClient
        from main import app

        service = MagicMock()
        service.enroll_multi_angle.return_value = {
            "success": True,
            "message": "Enrolled: Test (3/3 angles)",
            "phase_results": [],
            "total_saved": 3,
            "embeddings_saved": 3,
        }
        mock_service.return_value = service

        client = TestClient(app)

        # Create 3 test images
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', img)
        img_bytes = buf.tobytes()

        response = client.post(
            "/api/enroll/v2",
            data={"student_id": "HS001", "name": "Test", "class_name": "12A1"},
            files=[
                ("image_front", ("front.jpg", io.BytesIO(img_bytes), "image/jpeg")),
                ("image_left", ("left.jpg", io.BytesIO(img_bytes), "image/jpeg")),
                ("image_right", ("right.jpg", io.BytesIO(img_bytes), "image/jpeg")),
            ],
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_saved"] == 3

    @patch("app.routes.enrollment_v2.get_enrollment_v2_service")
    def test_enroll_v2_endpoint_accepts_multi_frame_fields(self, mock_service):
        """POST /api/enroll/v2 accepts repeated image fields per angle."""
        from fastapi.testclient import TestClient
        from main import app

        service = MagicMock()
        service.enroll_multi_angle.return_value = {
            "success": True,
            "message": "Enrolled: Test (3/3 angles)",
            "phase_results": [],
            "total_saved": 3,
            "embeddings_saved": 3,
        }
        mock_service.return_value = service

        client = TestClient(app)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', img)
        img_bytes = buf.tobytes()

        files = []
        for angle in ("front", "left", "right"):
            for idx in range(3):
                files.append(
                    (f"image_{angle}", (f"{angle}_{idx}.jpg", io.BytesIO(img_bytes), "image/jpeg"))
                )

        response = client.post(
            "/api/enroll/v2",
            data={"student_id": "HS002", "name": "Multi", "class_name": "12A1"},
            files=files,
        )

        assert response.status_code == 200
        images_arg = service.enroll_multi_angle.call_args.args[3]
        assert len(images_arg["front"]) == 3
        assert len(images_arg["left"]) == 3
        assert len(images_arg["right"]) == 3

    @patch("app.routes.enrollment.get_db")
    def test_student_photo_rejects_path_outside_face_crops(self, mock_get_db):
        """Student photo endpoint must not serve arbitrary filesystem paths."""
        from fastapi.testclient import TestClient
        from main import app

        db = MagicMock()
        db.get_student_any.return_value = {
            "id": "HS_BAD",
            "name": "Bad Path",
            "photo_path": str(Path("README.md").resolve()),
        }
        mock_get_db.return_value = db

        client = TestClient(app)
        response = client.get("/api/students/HS_BAD/photo")

        assert response.status_code == 403

    @patch("app.routes.scan_v3.get_challenge_v3_service")
    def test_scan_v3_challenge_endpoint(self, mock_service):
        """POST /api/scan/v3/challenge decodes repeated frame files."""
        from fastapi.testclient import TestClient
        from main import app

        service = MagicMock()
        service.verify_challenge.return_value = {
            "success": True,
            "challenge_passed": True,
            "results": [{"status": "present", "student_id": "S001"}],
        }
        mock_service.return_value = service

        client = TestClient(app)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', img)
        img_bytes = buf.tobytes()

        response = client.post(
            "/api/scan/v3/challenge",
            data={"challenge_id": "challenge-1"},
            files=[
                ("frames", ("frame1.jpg", io.BytesIO(img_bytes), "image/jpeg")),
                ("frames", ("frame2.jpg", io.BytesIO(img_bytes), "image/jpeg")),
            ],
        )

        assert response.status_code == 200
        assert response.json()["challenge_passed"] is True
        assert service.verify_challenge.call_count == 1


# ═══════════════════════════════════════════════════════════
#  Config Tests
# ═══════════════════════════════════════════════════════════

class TestConfig:
    """Verify V2/V3 config constants exist."""

    def test_v2_config(self):
        import config
        assert hasattr(config, "ENROLL_V2_MIN_FRAMES_PER_PHASE")
        assert hasattr(config, "ENROLL_V2_BLUR_MIN")
        assert hasattr(config, "ENROLL_V2_POSE_FRONT_MAX_DISP")
        assert hasattr(config, "ENROLL_V2_POSE_TURN_THRESHOLD")
        assert config.ENROLL_V2_BLUR_MIN == 80.0

    def test_v3_config(self):
        import config
        assert hasattr(config, "DETECT_V3_COSINE_THRESHOLD")
        assert hasattr(config, "DETECT_V3_MOIRE_SCREEN_THRESHOLD")
        assert hasattr(config, "DETECT_V3_MOIRE_BLOCK_THRESHOLD")
        assert hasattr(config, "DETECT_V3_MOIRE_CHALLENGE_THRESHOLD")
        assert hasattr(config, "DETECT_V3_LIVENESS_BLOCK_THRESHOLD")
        assert hasattr(config, "DETECT_V3_LIVENESS_CHALLENGE_THRESHOLD")
        assert hasattr(config, "DETECT_V3_CHALLENGE_ENABLED")
        assert hasattr(config, "DETECT_V3_BLINK_EAR_CLOSED_THRESHOLD")
        assert hasattr(config, "DETECT_V3_BLINK_EAR_OPEN_THRESHOLD")
        assert hasattr(config, "DETECT_V3_BLINK_EAR_DELTA")
        assert config.DETECT_V3_COSINE_THRESHOLD == 0.52
        assert config.DETECT_V3_MOIRE_SCREEN_THRESHOLD == 0.30
        assert config.DETECT_V3_MOIRE_BLOCK_THRESHOLD < config.DETECT_V3_MOIRE_CHALLENGE_THRESHOLD
        assert config.DETECT_V3_BLINK_EAR_CLOSED_THRESHOLD < config.DETECT_V3_BLINK_EAR_OPEN_THRESHOLD


# ═══════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import cv2
    pytest.main([__file__, "-v", "--tb=short"])
