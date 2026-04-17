"""
Backward-compatible browser stream wrapper for the shared Detect V3 runtime.
"""
from core.runtime_v3 import DetectV3RuntimeSession


class ScanV3StreamSession(DetectV3RuntimeSession):
    """Alias kept for existing imports/tests and the browser WebSocket route."""

    pass
