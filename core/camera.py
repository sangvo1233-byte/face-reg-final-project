"""
Camera Manager — Quản lý camera cho hệ thống chấm công.

Đọc frame, chọn frame sắc nét, quét enrollment 5 giây.
"""
import cv2
import time
import numpy as np
from loguru import logger

import config


class CameraManager:
    """Quản lý camera: open, read, capture best frame, enrollment scan."""

    def __init__(self):
        self._camera = None

    def _get_camera(self):
        if self._camera is not None and self._camera.isOpened():
            return self._camera
        logger.info(f"Opening camera: {config.CAMERA_SOURCE}")
        cap = cv2.VideoCapture(config.CAMERA_SOURCE)
        if isinstance(config.CAMERA_SOURCE, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        if not cap.isOpened():
            logger.warning(f"Cannot open camera: {config.CAMERA_SOURCE}")
            return None
        self._camera = cap
        return cap

    def read_frame(self) -> np.ndarray:
        """Đọc 1 frame từ camera."""
        ret, frame = self._get_camera().read()
        if not ret or frame is None:
            raise RuntimeError("Failed to read frame")
        return frame

    def capture_best_frame(self, count: int = 3) -> np.ndarray:
        """Chụp N frame, chọn sắc nét nhất (Laplacian variance)."""
        cap = self._get_camera()
        frames, scores = [], []
        for _ in range(count):
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
                time.sleep(0.1)
        if not frames:
            raise RuntimeError("No frames captured")
        return frames[int(np.argmax(scores))]

    def capture_enrollment_frames(self, duration: int = None) -> list[np.ndarray]:
        """Quét camera liên tục trong duration giây, trả danh sách frames.

        Dùng cho enrollment: thu thập nhiều frame ở nhiều góc.

        Args:
            duration: Thời gian quét (giây), mặc định theo config.ENROLL_SCAN_DURATION

        Returns:
            list[np.ndarray]: Danh sách frames (tối đa config.ENROLL_MAX_FRAMES)
        """
        duration = duration or config.ENROLL_SCAN_DURATION
        cap = self._get_camera()
        frames = []
        start = time.time()
        frame_interval = duration / config.ENROLL_MAX_FRAMES

        logger.info(f"Starting enrollment scan for {duration}s...")
        while time.time() - start < duration:
            if len(frames) >= config.ENROLL_MAX_FRAMES:
                break
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
            time.sleep(frame_interval)

        logger.info(f"Enrollment scan done: {len(frames)} frames in {time.time() - start:.1f}s")
        return frames

    def is_opened(self) -> bool:
        """Kiểm tra camera đang mở."""
        return self._camera is not None and self._camera.isOpened()

    def release(self):
        """Giải phóng camera."""
        if self._camera is not None:
            self._camera.release()
            self._camera = None


# ── Singleton ───────────────────────────────────────────────

_camera = None

def get_camera() -> CameraManager:
    global _camera
    if _camera is None:
        _camera = CameraManager()
    return _camera
