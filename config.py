"""
Face Attendance System — Centralized configuration.
"""
from pathlib import Path

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

# ── Server ──────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000
