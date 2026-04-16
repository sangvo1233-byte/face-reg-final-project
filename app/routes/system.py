"""
System Routes — /api/system/*
"""
from fastapi import APIRouter

import config
from core.face_engine import get_engine
from core.database import get_db

router = APIRouter(tags=["system"])


@router.get("/api/system/status")
async def system_status():
    engine = get_engine()
    db = get_db()

    gpu_available = False
    try:
        import onnxruntime as ort
        gpu_available = 'CUDAExecutionProvider' in ort.get_available_providers()
    except Exception:
        pass

    return {
        'status': 'running',
        'gpu': 'available' if gpu_available else 'cpu_only',
        'model': config.INSIGHTFACE_MODEL,
        'students': db.get_student_count(),
        'embeddings': db.get_embedding_count(),
    }


@router.get("/api/system/capabilities")
async def system_capabilities():
    """Report available API versions and V2/V3 features."""
    return {
        "enroll_versions": ["v1", "v2"],
        "scan_versions": ["v1", "v3"],
        "features": {
            "moire_detection": True,
            "multi_angle_enrollment": True,
            "strict_threshold": config.DETECT_V3_COSINE_THRESHOLD,
            "default_threshold": config.COSINE_THRESHOLD,
            "enroll_v2_angles": ["front", "left", "right"],
        },
        "thresholds": {
            "v1_cosine": config.COSINE_THRESHOLD,
            "v3_cosine": config.DETECT_V3_COSINE_THRESHOLD,
            "v3_moire_screen": config.DETECT_V3_MOIRE_SCREEN_THRESHOLD,
            "v2_blur_min": config.ENROLL_V2_BLUR_MIN,
            "v2_pose_front_max": config.ENROLL_V2_POSE_FRONT_MAX_DISP,
            "v2_pose_turn_min": config.ENROLL_V2_POSE_TURN_THRESHOLD,
        },
    }

