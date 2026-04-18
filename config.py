"""
Face Attendance System — Centralized configuration.
"""
import os
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_list(name: str, default: list[str] | None = None) -> list[str]:
    value = os.getenv(name)
    if not value:
        return list(default or [])
    return [item.strip() for item in value.split(",") if item.strip()]


APP_NAME = os.getenv("APP_NAME", "Face Attendance System")
APP_VERSION = os.getenv("APP_VERSION", "4.4.0")
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
DEBUG = _env_bool("DEBUG", APP_ENV == "development")
UVICORN_RELOAD = _env_bool("UVICORN_RELOAD", DEBUG)
API_DOCS_ENABLED = _env_bool("API_DOCS_ENABLED", DEBUG)

# ── Paths ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
LOGS_DIR = BASE_DIR / "logs"
FACE_CROPS_DIR = LOGS_DIR / "face_crops"
EVIDENCE_DIR = LOGS_DIR / "evidence"
WEB_DIR = BASE_DIR / "web"

for d in [MODELS_DIR, DATABASE_DIR, LOGS_DIR, FACE_CROPS_DIR, EVIDENCE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Database ────────────────────────────────────────────────
SQLITE_DB_PATH = DATABASE_DIR / "attendance.db"

# ── Camera ──────────────────────────────────────────────────
CAMERA_SOURCE = 0          # 0 = default webcam; use "rtsp://..." for IP cameras
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# ── Face Detection (InsightFace) ────────────────────────────
INSIGHTFACE_MODEL = "buffalo_l"
DET_SIZE = (480, 480)
MIN_FACE_SIZE = 30
DET_CONFIDENCE = 0.5

# ── Face Recognition (ArcFace) ──────────────────────────────
EMBEDDING_DIM = 512
COSINE_THRESHOLD = 0.45

# ── Enrollment ──────────────────────────────────────────────
ENROLL_SCAN_DURATION = 5       # Camera capture window during enrollment (seconds)
ENROLL_MIN_FRAMES = 3          # Minimum frames with detected face required to enroll
ENROLL_MAX_FRAMES = 15         # Maximum frames used to compute the average embedding

# ── Quality Gate ────────────────────────────────────────────
QUALITY_MIN_FACE_SIZE = 80          # pixels
QUALITY_MAX_BLUR = 100.0            # Laplacian variance below threshold indicates blur
QUALITY_MIN_BRIGHTNESS = 40
QUALITY_MAX_BRIGHTNESS = 220
QUALITY_MAX_YAW_ANGLE = 30

# ── Anti-Spoofing (Passive Liveness) ────────────────────────
LIVENESS_ENABLED = True
LIVENESS_SCORE_THRESHOLD = 0.5

# ── Enroll V2 (Multi-Angle) ─────────────────────────────
ENROLL_V2_MIN_FRAMES_PER_PHASE = 1      # tối thiểu frame hợp lệ mỗi phase
ENROLL_V2_RECOMMENDED_FRAMES   = 3      # khuyến nghị
ENROLL_V2_MAX_FRAMES_PER_PHASE = 8      # tối đa lấy
ENROLL_V2_BLUR_MIN             = 80.0   # Laplacian variance tối thiểu
ENROLL_V2_POSE_FRONT_MAX_DISP  = 0.12   # nose_x displacement tối đa cho "front"
ENROLL_V2_POSE_TURN_THRESHOLD  = 0.04   # nose_x shift tối thiểu cho left/right

# ── Detect V3 (Strict + Moiré) ─────────────────────────
DETECT_V3_COSINE_THRESHOLD       = 0.52   # ngưỡng chặt hơn default 0.45
DETECT_V3_MOIRE_SCREEN_THRESHOLD = 0.30   # dưới score này → is_screen
DETECT_V3_MOIRE_BLOCK_THRESHOLD = 0.25
DETECT_V3_MOIRE_CHALLENGE_THRESHOLD = 0.42
DETECT_V3_LIVENESS_BLOCK_THRESHOLD = 0.32
DETECT_V3_LIVENESS_CHALLENGE_THRESHOLD = 0.50
DETECT_V3_CHALLENGE_ENABLED = True
DETECT_V3_CHALLENGE_TTL_SECONDS = 10
DETECT_V3_CHALLENGE_MIN_FRAMES = 3
DETECT_V3_CHALLENGE_POSE_FRAMES = 2
DETECT_V3_CHALLENGE_MAX_FRAMES = 36
DETECT_V3_BLINK_EAR_CLOSED_THRESHOLD = 0.21
DETECT_V3_BLINK_EAR_OPEN_THRESHOLD = 0.24
DETECT_V3_BLINK_EAR_DELTA = 0.045
DETECT_V3_STREAM_BLINK_MIN_DROP = 0.05
DETECT_V3_STREAM_BLINK_MIN_OPEN_FRAMES = 2
DETECT_V3_STREAM_BLINK_MIN_CLOSED_FRAMES = 1
DETECT_V3_STREAM_BLINK_MAX_CLOSED_FRAMES = 10
DETECT_V3_STREAM_BLINK_BRIGHTNESS_JUMP_THRESHOLD = 24.0
DETECT_V3_STREAM_BLINK_LIGHTING_COOLDOWN_FRAMES = 2
DETECT_V3_STREAM_LIVE_MIN_FRAMES = 12
DETECT_V3_STREAM_ENABLED = True
DETECT_V3_STREAM_TARGET_FPS = 10
DETECT_V3_STREAM_DETECT_FPS = 10
DETECT_V3_STREAM_MOIRE_EVERY_N_DETECT = 3
DETECT_V3_STREAM_CLIENT_FRAME_WIDTH = 960
DETECT_V3_STREAM_CLIENT_JPEG_QUALITY = 0.82
DETECT_V3_STREAM_MIN_TRACK_SECONDS = 2.0
DETECT_V3_STREAM_MAX_CHECK_SECONDS = 6.0
DETECT_V3_STREAM_MOVEMENT_THRESHOLD = 2.0
DETECT_V3_STREAM_JPEG_MAX_BYTES = 450_000

# ── Model Paths ────────────────────────────────────────
FACE_LANDMARKER_MODEL = MODELS_DIR / "face_landmarker.task"

# ── Server ──────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = _env_int("PORT", 8000)
AUTO_PRELOAD_MODELS = _env_bool("AUTO_PRELOAD_MODELS", True)
AUTO_LOAD_EMBEDDING_CACHE = _env_bool("AUTO_LOAD_EMBEDDING_CACHE", True)
AUTO_START_CAMERA = _env_bool("AUTO_START_CAMERA", True)
CAMERA_REQUIRED = _env_bool("CAMERA_REQUIRED", False)
CORS_ALLOWED_ORIGINS = _env_list("CORS_ALLOWED_ORIGINS", [])
TRUSTED_HOSTS = _env_list("TRUSTED_HOSTS", [])
