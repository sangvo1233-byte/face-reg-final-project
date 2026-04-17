"""
Scan V3 routes.

The browser dashboard uses the WebSocket stream for continuous liveness state.
The single-frame HTTP endpoint remains available for API compatibility.
"""
import cv2
import numpy as np

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from loguru import logger

import config
from core.challenge_v3 import get_challenge_v3_service
from core.database import get_db
from core.detect_v3 import get_detect_v3_service
from core.stream_scan_v3 import ScanV3StreamSession

router = APIRouter(tags=["scan_v3"])


@router.post("/api/scan/v3")
async def scan_v3(image: UploadFile = File(...)):
    """Scan with V3 pipeline: moire anti-spoof + strict matching.

    Requires an active attendance session. This endpoint scans one frame and is
    kept for API compatibility; the web dashboard uses /ws/scan-v3.
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


@router.websocket("/ws/scan-v3")
async def scan_v3_stream(websocket: WebSocket):
    """Continuous browser-camera Scan V3 stream.

    The client sends JPEG bytes. The server keeps liveness state across frames
    and runs recognition on a throttled cadence while matching liveness back to
    each detected bbox.
    """
    await websocket.accept()

    if not config.DETECT_V3_STREAM_ENABLED:
        await websocket.send_json({
            "type": "error",
            "status": "stream_disabled",
            "message": "Scan V3 stream is disabled",
        })
        await websocket.close(code=1008)
        return

    db = get_db()
    active = db.get_active_session()
    if not active:
        await websocket.send_json({
            "type": "error",
            "status": "no_session",
            "message": "No active session",
        })
        await websocket.close(code=1008)
        return

    stream = ScanV3StreamSession(active["id"])
    logger.info(f"Scan V3 WebSocket connected: session={active['id']}")

    try:
        await websocket.send_json({
            "type": "stream_ready",
            "status": "ready",
            "message": "Browser camera stream connected",
            "mode": "browser_ws",
            "session_id": active["id"],
            "target_fps": config.DETECT_V3_STREAM_TARGET_FPS,
            "detect_fps": config.DETECT_V3_STREAM_DETECT_FPS,
            "client_frame_width": config.DETECT_V3_STREAM_CLIENT_FRAME_WIDTH,
            "client_jpeg_quality": config.DETECT_V3_STREAM_CLIENT_JPEG_QUALITY,
        })

        while True:
            contents = await websocket.receive_bytes()
            if len(contents) > config.DETECT_V3_STREAM_JPEG_MAX_BYTES:
                await websocket.send_json({
                    "type": "scan_state",
                    "status": "frame_too_large",
                    "message": "Frame too large",
                })
                continue

            frame = _decode_bytes(contents)
            if frame is None:
                await websocket.send_json({
                    "type": "scan_state",
                    "status": "bad_frame",
                    "message": "Invalid frame",
                })
                continue

            for message in stream.process_frame(frame):
                await websocket.send_json(message)
    except WebSocketDisconnect:
        logger.info(f"Scan V3 WebSocket disconnected: session={active['id']}")
    except Exception as exc:
        logger.exception(f"Scan V3 WebSocket error: {exc}")
        try:
            await websocket.send_json({
                "type": "error",
                "status": "server_error",
                "message": "Scan stream error",
            })
        except Exception:
            pass
    finally:
        stream.close()


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
    return _decode_bytes(contents)


def _decode_bytes(contents: bytes):
    return cv2.imdecode(
        np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR
    )
