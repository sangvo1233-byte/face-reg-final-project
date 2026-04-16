"""
Live Routes — /api/live/*

Camera live stream + snapshot. Khong block server khi khong co camera.
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
        _, buf = cv2.imencode('.jpg', blank)
        _NO_CAMERA_FRAME = buf.tobytes()
    return _NO_CAMERA_FRAME


def _generate_mjpeg():
    """Generator MJPEG stream. Tra blank frame neu khong co camera."""
    from core.camera import get_camera
    cam = get_camera()

    while True:
        try:
            if not cam.is_opened():
                cam._get_camera()  # Thu mo 1 lan
        except Exception:
            pass

        if cam.is_opened():
            try:
                frame = cam.read_frame()
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
                time.sleep(1.0 / 24)
                continue
            except Exception:
                cam.release()

        # No camera — send blank, sleep longer
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + _blank_frame() + b'\r\n'
        time.sleep(2.0)


@router.get("/api/live/stream")
async def live_stream():
    return StreamingResponse(
        _generate_mjpeg(),
        media_type='multipart/x-mixed-replace; boundary=frame',
    )
