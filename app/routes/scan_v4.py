"""
Detect V4.4 routes.

V4 runs side by side with V3. The browser and local-direct clients send frames;
all anti-spoof and attendance decisions stay in the backend runtime.
"""
import asyncio

import cv2
import numpy as np

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from loguru import logger

import config
from core.database import get_db
from core.local_runner_v4 import get_local_runner_v4
from core.stream_scan_v4 import ScanV4StreamSession

router = APIRouter(tags=["scan_v4"])


@router.post("/api/scan/v4")
async def scan_v4(image: UploadFile = File(...)):
    """Single-frame V4 compatibility endpoint.

    The production UI uses /ws/scan-v4 because V4 relies on stream liveness
    and rolling anti-spoof state.
    """
    active = get_db().get_active_session()
    if not active:
        return {"success": False, "error": "No active session", "scan_version": "v4.4"}

    frame = await _decode_upload(image)
    if frame is None:
        raise HTTPException(400, "Invalid image")

    stream = ScanV4StreamSession(active["id"])
    try:
        events = stream.process_frame(frame)
    finally:
        stream.close()

    return {
        "success": True,
        "scan_version": "v4.4",
        "events": events,
        "results": [event.get("result", event) for event in events],
    }


@router.websocket("/ws/scan-v4")
async def scan_v4_stream(websocket: WebSocket):
    await websocket.accept()

    if not config.DETECT_V3_STREAM_ENABLED:
        await websocket.send_json({
            "type": "error",
            "status": "stream_disabled",
            "message": "Scan V4 stream is disabled",
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

    stream = ScanV4StreamSession(active["id"])
    logger.info(f"Scan V4 WebSocket connected: session={active['id']}")

    try:
        await websocket.send_json({
            "type": "stream_ready",
            "status": "ready",
            "message": "Browser camera V4 stream connected",
            "mode": "browser_ws_v4",
            "session_id": active["id"],
            "target_fps": config.DETECT_V3_STREAM_TARGET_FPS,
            "detect_fps": config.DETECT_V3_STREAM_DETECT_FPS,
            "client_frame_width": config.DETECT_V3_STREAM_CLIENT_FRAME_WIDTH,
            "client_jpeg_quality": config.DETECT_V3_STREAM_CLIENT_JPEG_QUALITY,
            "scan_version": "v4.4",
        })

        while True:
            contents = await websocket.receive_bytes()
            if len(contents) > config.DETECT_V3_STREAM_JPEG_MAX_BYTES:
                await websocket.send_json({
                    "type": "scan_state",
                    "status": "frame_too_large",
                    "message": "Frame too large",
                    "scan_version": "v4.4",
                })
                continue

            frame = _decode_bytes(contents)
            if frame is None:
                await websocket.send_json({
                    "type": "scan_state",
                    "status": "bad_frame",
                    "message": "Invalid frame",
                    "scan_version": "v4.4",
                })
                continue

            for message in stream.process_frame(frame):
                await websocket.send_json(message)
    except WebSocketDisconnect:
        logger.info(f"Scan V4 WebSocket disconnected: session={active['id']}")
    except Exception as exc:
        logger.exception(f"Scan V4 WebSocket error: {exc}")
        try:
            await websocket.send_json({
                "type": "error",
                "status": "server_error",
                "message": "Scan V4 stream error",
            })
        except Exception:
            pass
    finally:
        stream.close()


@router.post("/api/scan/v4/local/start")
async def local_scan_v4_start():
    active = get_db().get_active_session()
    if not active:
        return {"success": False, "message": "No active session", "state": "no_session"}
    return get_local_runner_v4().start(active["id"])


@router.post("/api/scan/v4/local/stop")
async def local_scan_v4_stop():
    return get_local_runner_v4().stop()


@router.get("/api/scan/v4/local/status")
async def local_scan_v4_status():
    return get_local_runner_v4().get_status()


@router.websocket("/ws/scan-v4/local")
async def local_scan_v4_ws(websocket: WebSocket):
    await websocket.accept()
    runner = get_local_runner_v4()
    queue = runner.subscribe()

    try:
        status = runner.get_status()
        await websocket.send_json({
            "type": "stream_ready",
            "status": "ready",
            "mode": "local_direct_v4",
            "message": "Local Direct V4 event stream connected",
            "runner_state": status["runner_state"],
            "session_id": status["session_id"],
            "camera_status": status["camera_status"],
            "last_event": status["last_event"],
            "scan_version": "v4.4",
        })

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=5.0)
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "heartbeat",
                    "runner_state": runner.state,
                    "session_id": runner.session_id,
                    "scan_version": "v4.4",
                })
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning(f"Local scan V4 websocket error: {exc}")
    finally:
        runner.unsubscribe(queue)


async def _decode_upload(upload: UploadFile):
    contents = await upload.read()
    return _decode_bytes(contents)


def _decode_bytes(contents: bytes):
    return cv2.imdecode(
        np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR
    )
