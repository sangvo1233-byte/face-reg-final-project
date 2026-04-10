"""
Data Models — Dataclasses cho hệ thống điểm danh học sinh.
"""
# === CUDA SETUP (MUST be before any onnxruntime import) ===
import os, glob
try:
    import nvidia
    nvidia_base = os.path.dirname(nvidia.__path__[0])
    for bin_dir in glob.glob(os.path.join(nvidia_base, '*', 'bin')):
        if os.path.isdir(bin_dir):
            os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')
            os.add_dll_directory(bin_dir)
except ImportError:
    pass

try:
    import onnxruntime as ort
    ort.preload_dlls()
except Exception:
    pass
# === END CUDA SETUP ===

import numpy as np
from dataclasses import dataclass, field


@dataclass
class DetectedFace:
    """1 khuôn mặt đã detect."""
    bbox: np.ndarray
    landmarks: np.ndarray
    confidence: float
    aligned_face: np.ndarray
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class MatchResult:
    """Kết quả matching."""
    matched: bool
    name: str
    student_id: str
    score: float
    index: int


@dataclass
class QualityResult:
    """Kết quả kiểm tra chất lượng ảnh khuôn mặt."""
    passed: bool
    face_size: int
    blur_score: float
    brightness: float
    yaw_angle: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class LivenessResult:
    """Kết quả kiểm tra liveness (chống giả mạo)."""
    is_live: bool
    score: float
    reason: str = ""
