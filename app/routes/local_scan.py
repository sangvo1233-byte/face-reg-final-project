"""
Local-direct Detect V3 scan routes.
"""
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from core.database import get_db
from core.local_runner import get_local_runner

router = APIRouter(tags=["local_scan"])


@router.post("/api/scan/v3/local/start")
async def local_scan_start():
    active = get_db().get_active_session()
    if not active:
        return {"success": False, "message": "No active session", "state": "no_session"}
    return get_local_runner().start(active["id"])


@router.post("/api/scan/v3/local/stop")
async def local_scan_stop():
    return get_local_runner().stop()


@router.get("/api/scan/v3/local/status")
async def local_scan_status():
    return get_local_runner().get_status()


@router.websocket("/ws/scan-v3/local")
async def local_scan_ws(websocket: WebSocket):
    await websocket.accept()
    runner = get_local_runner()
    queue = runner.subscribe()

    try:
        status = runner.get_status()
        await websocket.send_json({
            "type": "stream_ready",
            "status": "ready",
            "mode": "local_direct",
            "message": "Local Direct event stream connected",
            "runner_state": status["runner_state"],
            "session_id": status["session_id"],
            "camera_status": status["camera_status"],
            "last_event": status["last_event"],
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
                })
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning(f"Local scan websocket error: {exc}")
    finally:
        runner.unsubscribe(queue)
