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
