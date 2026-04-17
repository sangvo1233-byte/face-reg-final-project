"""
Shared camera producer.

One background thread owns cv2.VideoCapture and keeps the latest raw frame in
memory. Preview, enrollment, and local-direct scan all read that same buffer.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from loguru import logger

import config


class CameraService:
    def __init__(self):
        self._cap: cv2.VideoCapture | None = None
        self._latest_frame: np.ndarray | None = None
        self._last_frame_at = 0.0
        self._error: str | None = None
        self._source = config.CAMERA_SOURCE
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self):
        if self._running:
            return
        self._running = True
        self._error = None
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="camera-service")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._release()

    def get_latest_frame(self, copy: bool = True) -> np.ndarray | None:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy() if copy else self._latest_frame

    def get_status(self) -> dict:
        with self._lock:
            has_frame = self._latest_frame is not None
            last_frame_at = self._last_frame_at or None
        frame_age = time.time() - last_frame_at if last_frame_at else None
        return {
            "opened": self.is_opened(),
            "running": self._running,
            "has_frame": has_frame,
            "last_frame_at": last_frame_at,
            "frame_age_seconds": round(frame_age, 2) if frame_age is not None else None,
            "source": self._source,
            "error": self._error,
        }

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read_frame(self) -> np.ndarray:
        if not self._running:
            self.start()
        frame = self.get_latest_frame()
        if frame is None:
            raise RuntimeError("No frame available from camera")
        return frame

    def capture_best_frame(self, count: int = 3) -> np.ndarray:
        if not self._running:
            self.start()
        frames, scores = [], []
        for _ in range(count):
            frame = self.get_latest_frame()
            if frame is not None:
                frames.append(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
            time.sleep(0.1)
        if not frames:
            raise RuntimeError("No frames captured")
        return frames[int(np.argmax(scores))]

    def capture_enrollment_frames(self, duration: int = None) -> list[np.ndarray]:
        if not self._running:
            self.start()
        duration = duration or config.ENROLL_SCAN_DURATION
        frames = []
        start = time.time()
        frame_interval = duration / max(config.ENROLL_MAX_FRAMES, 1)
        while time.time() - start < duration and len(frames) < config.ENROLL_MAX_FRAMES:
            frame = self.get_latest_frame()
            if frame is not None:
                frames.append(frame)
            time.sleep(frame_interval)
        return frames

    def release(self):
        self.stop()

    def _capture_loop(self):
        interval = 1.0 / max(config.CAMERA_FPS, 1)
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._open_camera()
                if self._cap is None or not self._cap.isOpened():
                    time.sleep(2.0)
                    continue

            try:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    self._error = "Camera read failed"
                    self._release()
                    time.sleep(0.5)
                    continue
                with self._lock:
                    self._latest_frame = frame
                    self._last_frame_at = time.time()
                self._error = None
            except Exception as exc:
                self._error = str(exc)
                self._release()
                time.sleep(0.5)
                continue

            time.sleep(interval)

    def _open_camera(self):
        try:
            logger.info(f"Opening camera: {self._source}")
            cap = cv2.VideoCapture(self._source)
            if isinstance(self._source, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self._error = f"Cannot open camera: {self._source}"
                logger.warning(self._error)
                cap.release()
                self._cap = None
                return
            self._cap = cap
            self._error = None
        except Exception as exc:
            self._error = str(exc)
            self._cap = None

    def _release(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        with self._lock:
            self._latest_frame = None


CameraManager = CameraService

_camera = None


def get_camera() -> CameraService:
    global _camera
    if _camera is None:
        _camera = CameraService()
    return _camera
