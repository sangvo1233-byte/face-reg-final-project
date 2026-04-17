"""
System Routes — /api/system/*
"""
from fastapi import APIRouter

import config
from core.face_engine import get_engine
from core.database import get_db
from core.detect_v4 import (
    MOIRE_BLOCK_THRESHOLD,
    MOIRE_SCREEN_THRESHOLD,
    PHONE_RECT_CONTEXT_SCALE,
    PHONE_RECT_STRONG_THRESHOLD,
    PHONE_RECT_SUSPICIOUS_THRESHOLD,
    PHONE_RECT_VERTICAL_RATIO,
    V4_COSINE_THRESHOLD,
)

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
        "scan_versions": ["v1", "v3", "v3_ws", "local_direct", "v4", "v4_ws", "local_direct_v4"],
        "scan_mode_default": "auto",
        "scan_modes": ["auto", "local_direct", "browser_ws"],
        "local_direct_available": True,
        "browser_stream_available": bool(config.DETECT_V3_STREAM_ENABLED),
        "features": {
            "moire_detection": True,
            "multi_angle_enrollment": True,
            "stream_liveness": config.DETECT_V3_STREAM_ENABLED,
            "strict_threshold": config.DETECT_V3_COSINE_THRESHOLD,
            "default_threshold": config.COSINE_THRESHOLD,
            "enroll_v2_angles": ["front", "left", "right"],
            "server_camera_scan": True,
            "detect_v4": True,
            "screen_context": True,
            "phone_rectangle": True,
            "portrait_phone_roi": True,
            "enhanced_moire": True,
        },
        "thresholds": {
            "v1_cosine": config.COSINE_THRESHOLD,
            "v3_cosine": config.DETECT_V3_COSINE_THRESHOLD,
            "v3_moire_screen": config.DETECT_V3_MOIRE_SCREEN_THRESHOLD,
            "v3_stream_target_fps": config.DETECT_V3_STREAM_TARGET_FPS,
            "v3_stream_detect_fps": config.DETECT_V3_STREAM_DETECT_FPS,
            "v3_stream_min_track_seconds": config.DETECT_V3_STREAM_MIN_TRACK_SECONDS,
            "v2_blur_min": config.ENROLL_V2_BLUR_MIN,
            "v2_pose_front_max": config.ENROLL_V2_POSE_FRONT_MAX_DISP,
            "v2_pose_turn_min": config.ENROLL_V2_POSE_TURN_THRESHOLD,
            "v4_cosine": V4_COSINE_THRESHOLD,
            "v4_moire_screen": MOIRE_SCREEN_THRESHOLD,
            "v4_moire_block": MOIRE_BLOCK_THRESHOLD,
            "v4_phone_rect_suspicious": PHONE_RECT_SUSPICIOUS_THRESHOLD,
            "v4_phone_rect_strong": PHONE_RECT_STRONG_THRESHOLD,
            "v4_phone_rect_context_scale": PHONE_RECT_CONTEXT_SCALE,
            "v4_phone_rect_vertical_ratio": PHONE_RECT_VERTICAL_RATIO,
        },
    }
