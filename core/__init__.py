# Core package — Face Attendance System
from core.schemas import DetectedFace, MatchResult, QualityResult, LivenessResult
from core.camera import CameraManager, get_camera
from core.anti_spoof import AntiSpoof, get_anti_spoof
from core.database import Database, get_db
from core.face_engine import FaceEngine, get_engine
from core.moire import MoireDetector, get_moire_detector
from core.enrollment_v2 import EnrollmentV2Service, get_enrollment_v2_service
from core.detect_v3 import DetectV3Service, get_detect_v3_service

__all__ = [
    "DetectedFace", "MatchResult", "QualityResult", "LivenessResult",
    "CameraManager", "get_camera",
    "AntiSpoof", "get_anti_spoof",
    "Database", "get_db",
    "FaceEngine", "get_engine",
    "MoireDetector", "get_moire_detector",
    "EnrollmentV2Service", "get_enrollment_v2_service",
    "DetectV3Service", "get_detect_v3_service",
]
