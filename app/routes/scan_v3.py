"""
Scan V3 Routes — POST /api/scan/v3

Face recognition with moiré anti-spoof + strict threshold.
"""
import cv2
import numpy as np

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from loguru import logger

from core.database import get_db
from core.challenge_v3 import get_challenge_v3_service
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

    frame = await _decode_upload(image)
    if frame is None:
        raise HTTPException(400, "Invalid image")

    service = get_detect_v3_service()
    result = service.scan_attendance(frame, active["id"])
    return result


@router.post("/api/scan/v3/challenge")
async def scan_v3_challenge(
    challenge_id: str = Form(...),
    frames: list[UploadFile] = File(...),
):
    """Verify a Detect V3 active challenge with multiple browser frames."""
    decoded = []
    for upload in frames:
        frame = await _decode_upload(upload)
        if frame is not None:
            decoded.append(frame)

    if not decoded:
        raise HTTPException(400, "No valid challenge frames")

    service = get_challenge_v3_service()
    return service.verify_challenge(challenge_id, decoded)


async def _decode_upload(upload: UploadFile):
    contents = await upload.read()
    return cv2.imdecode(
        np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR
    )
