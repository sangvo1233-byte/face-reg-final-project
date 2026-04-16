"""
Scan V3 Routes — POST /api/scan/v3

Face recognition with moiré anti-spoof + strict threshold.
"""
import cv2
import numpy as np

from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger

from core.database import get_db
from core.detect_v3 import get_detect_v3_service

router = APIRouter(tags=["scan_v3"])


@router.post("/api/scan/v3")
async def scan_v3(image: UploadFile = File(...)):
    """Scan with V3 pipeline: moiré anti-spoof + strict matching.

    Requires an active attendance session.
    Uses stricter cosine threshold (0.52) and moiré pattern
    detection to reject screen-displayed faces.
    """
    db = get_db()
    active = db.get_active_session()
    if not active:
        return {"success": False, "error": "No active session"}

    contents = await image.read()
    frame = cv2.imdecode(
        np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR
    )
    if frame is None:
        raise HTTPException(400, "Invalid image")

    service = get_detect_v3_service()
    result = service.scan_attendance(frame, active["id"])
    return result
