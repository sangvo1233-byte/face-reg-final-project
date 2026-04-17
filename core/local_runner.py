"""
Local Direct Detect V3 runner.
"""
from __future__ import annotations

import asyncio
import threading
import time

from loguru import logger

from core.camera import get_camera
from core.runtime_v3 import DetectV3RuntimeSession


class EventHub:
    def __init__(self):
        self._subscribers: dict[asyncio.Queue, asyncio.AbstractEventLoop] = {}
        self._last_event: dict | None = None
        self._lock = threading.Lock()

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        loop = asyncio.get_running_loop()
        with self._lock:
            self._subscribers[queue] = loop
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        with self._lock:
            self._subscribers.pop(queue, None)

    def broadcast(self, event: dict):
        with self._lock:
            self._last_event = event
            subscribers = list(self._subscribers.items())
        for queue, loop in subscribers:
            try:
                loop.call_soon_threadsafe(self._push, queue, event)
            except RuntimeError:
                self.unsubscribe(queue)

    @staticmethod
    def _push(queue: asyncio.Queue, event: dict):
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    @property
    def last_event(self) -> dict | None:
        with self._lock:
            return self._last_event

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


class LocalDirectRunner:
    def __init__(self):
        self._camera = get_camera()
        self._runtime: DetectV3RuntimeSession | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._state = "idle"
        self._session_id: int | None = None
        self._error: str | None = None
        self._scan_fps = 0.0
        self._lock = threading.Lock()
        self._hub = EventHub()

    def start(self, session_id: int) -> dict:
        with self._lock:
            if self._running and self._session_id == session_id:
                return {"success": True, "message": "Already running", "state": self._state}
            if self._running:
                self._stop_internal()
            self._session_id = session_id
            self._error = None
            self._state = "starting"
            self._running = True

        self._camera.start()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="local-direct-scan")
        self._thread.start()
        return {"success": True, "message": f"Local direct scan started for session {session_id}", "state": "starting"}

    def stop(self) -> dict:
        with self._lock:
            self._stop_internal()
        return {"success": True, "message": "Local direct scan stopped", "state": "stopped"}

    def get_status(self) -> dict:
        state = self._state
        if not self._running and self._session_id is None and state in {"idle", "stopped"}:
            state = "no_session"
        return {
            "mode": "local_direct",
            "runner_state": state,
            "camera_status": self._camera.get_status(),
            "session_id": self._session_id,
            "last_event": self._hub.last_event,
            "error": self._error,
            "scan_fps": round(self._scan_fps, 1),
            "subscribers": self._hub.subscriber_count,
        }

    def subscribe(self) -> asyncio.Queue:
        return self._hub.subscribe()

    def unsubscribe(self, queue: asyncio.Queue):
        self._hub.unsubscribe(queue)

    @property
    def state(self) -> str:
        return self._state

    @property
    def session_id(self) -> int | None:
        return self._session_id

    def _stop_internal(self):
        self._running = False
        self._state = "stopped"
        self._session_id = None
        if self._runtime is not None:
            try:
                self._runtime.close()
            except Exception:
                pass
            self._runtime = None
        self._hub.broadcast({"type": "scan_state", "status": "stopped", "message": "Local direct scan stopped"})
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=3.0)
            self._thread = None

    def _loop(self):
        try:
            self._runtime = DetectV3RuntimeSession(self._session_id)
        except Exception as exc:
            self._error = f"Runtime init failed: {exc}"
            self._state = "stopped"
            self._running = False
            self._hub.broadcast({"type": "error", "status": "runtime_error", "message": self._error})
            return

        self._state = "running"
        self._hub.broadcast({
            "type": "stream_ready",
            "status": "ready",
            "mode": "local_direct",
            "message": "Local Direct scan active",
            "session_id": self._session_id,
            "camera_status": self._camera.get_status(),
        })

        frame_count = 0
        fps_start = time.time()
        while self._running:
            frame = self._camera.get_latest_frame(copy=True)
            if frame is None:
                self._state = "camera_error"
                self._error = self._camera.get_status().get("error") or "Camera frame unavailable"
                self._hub.broadcast({"type": "error", "status": "camera_error", "message": self._error})
                time.sleep(1.0)
                continue

            try:
                events = self._runtime.process_frame(frame)
            except Exception as exc:
                logger.exception(f"Local direct runtime error: {exc}")
                self._hub.broadcast({"type": "error", "status": "runtime_error", "message": str(exc)})
                time.sleep(0.1)
                continue

            for event in events:
                status = event.get("status", "")
                if status in {"challenge_required", "challenge_active"}:
                    self._state = "challenge"
                elif status == "cooldown":
                    self._state = "cooldown"
                elif self._state != "stopped":
                    self._state = "running"
                self._hub.broadcast(event)

            frame_count += 1
            now = time.time()
            if now - fps_start >= 1.0:
                self._scan_fps = frame_count / (now - fps_start)
                frame_count = 0
                fps_start = now
            time.sleep(0.01)

        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
        self._state = "stopped"


_local_runner: LocalDirectRunner | None = None


def get_local_runner() -> LocalDirectRunner:
    global _local_runner
    if _local_runner is None:
        _local_runner = LocalDirectRunner()
    return _local_runner
