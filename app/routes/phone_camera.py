"""
Phone Camera Routes — /ws/phone-camera, /api/phone/*

Nhận frame từ camera điện thoại qua WebSocket,
cung cấp MJPEG stream và snapshot cho dashboard.
"""
import time
import asyncio
import threading
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from loguru import logger

router = APIRouter(tags=["phone-camera"])

# ── Frame Buffer ────────────────────────────────────────────
_lock = threading.Lock()
_latest_frame: bytes | None = None
_last_frame_time: float = 0
_phone_connected: bool = False
_phone_info: dict = {}


def _store_frame(data: bytes):
    global _latest_frame, _last_frame_time
    with _lock:
        _latest_frame = data
        _last_frame_time = time.time()


def _get_frame() -> bytes | None:
    with _lock:
        return _latest_frame


def _is_fresh(max_age: float = 5.0) -> bool:
    with _lock:
        if _latest_frame is None:
            return False
        return (time.time() - _last_frame_time) < max_age


# ── WebSocket endpoint ─────────────────────────────────────
@router.websocket("/ws/phone-camera")
async def phone_camera_ws(websocket: WebSocket):
    global _phone_connected, _phone_info

    await websocket.accept()
    _phone_connected = True
    _phone_info = {
        "connected_at": datetime.now().isoformat(),
        "client": websocket.client.host if websocket.client else "unknown",
    }
    logger.info(f"Phone camera connected from {_phone_info['client']}")

    try:
        while True:
            data = await websocket.receive_bytes()
            _store_frame(data)
    except WebSocketDisconnect:
        logger.info("Phone camera disconnected")
    except Exception as e:
        logger.warning(f"Phone camera WS error: {e}")
    finally:
        _phone_connected = False
        _phone_info = {}
        # Clear the frame buffer so stale frames cannot be scanned
        # after the phone disconnects.
        global _latest_frame
        with _lock:
            _latest_frame = None
        logger.info("Phone frame buffer cleared")


# ── REST endpoints ──────────────────────────────────────────
@router.get("/api/phone/status")
async def phone_status():
    return {
        "connected": _phone_connected,
        "has_frame": _latest_frame is not None,
        "fresh": _is_fresh(),
        "info": _phone_info,
    }


@router.get("/api/phone/latest")
async def phone_latest():
    """Return the latest phone frame only if it is fresh (< 3s old).

    Returns 204 No Content when the phone is disconnected or the last
    frame is stale, preventing stale frames from being scanned.
    """
    if not _is_fresh(max_age=3.0):
        return Response(status_code=204, content=b"")
    frame = _get_frame()
    if frame is None:
        return Response(status_code=204, content=b"")
    return Response(content=frame, media_type="image/jpeg")


def _generate_phone_mjpeg():
    """MJPEG generator that streams from the phone camera frame buffer."""
    import numpy as np
    import cv2

    blank = None
    while True:
        frame = _get_frame()
        if frame and _is_fresh():
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(1.0 / 15)
        else:
            # No phone connected yet — serve a blank placeholder frame
            if blank is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    img, "Waiting for phone...", (130, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2,
                )
                cv2.putText(
                    img, "Open /phone on your phone", (120, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1,
                )
                _, buf = cv2.imencode(".jpg", img)
                blank = buf.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + blank + b"\r\n"
            )
            time.sleep(2.0)


@router.get("/api/live/phone-stream")
async def phone_stream():
    return StreamingResponse(
        _generate_phone_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
