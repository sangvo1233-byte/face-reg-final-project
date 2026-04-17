"""
Live routes for MJPEG camera preview.
"""
import time

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(tags=["live"])

_NO_CAMERA_FRAME = None


def _blank_frame():
    global _NO_CAMERA_FRAME
    if _NO_CAMERA_FRAME is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "No camera", (205, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (80, 80, 80), 2)
        _, buf = cv2.imencode(".jpg", blank)
        _NO_CAMERA_FRAME = buf.tobytes()
    return _NO_CAMERA_FRAME


def _generate_mjpeg():
    from core.camera import get_camera

    cam = get_camera()
    cam.start()

    while True:
        frame = cam.get_latest_frame(copy=True)
        if frame is not None:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(1.0 / 24)
            continue

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _blank_frame() + b"\r\n"
        time.sleep(1.0)


@router.get("/api/live/stream")
async def live_stream():
    return StreamingResponse(
        _generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
