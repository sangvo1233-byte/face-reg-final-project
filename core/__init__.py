# Core package — Face Attendance System
from core.schemas import DetectedFace, MatchResult, QualityResult, LivenessResult
from core.camera import CameraManager, get_camera
from core.anti_spoof import AntiSpoof, get_anti_spoof
from core.database import Database, get_db
from core.face_engine import FaceEngine, get_engine

__all__ = [
    "DetectedFace", "MatchResult", "QualityResult", "LivenessResult",
    "CameraManager", "get_camera",
    "AntiSpoof", "get_anti_spoof",
    "Database", "get_db",
    "FaceEngine", "get_engine",
]
