# Face Attendance System

Real-time face attendance system built with FastAPI, InsightFace ArcFace embeddings, MediaPipe-based stream liveness, SQLite, and a vanilla JavaScript dashboard.

The current default web runtime is Detect V4.4. It supports:

- `Auto` scan mode selection on the dashboard
- `Browser Stream` scanning over `ws://.../ws/scan-v4`
- `Local Direct` scanning from the server webcam over `ws://.../ws/scan-v4/local`
- multi-angle enrollment with `front`, `left`, and `right` capture
- layered anti-spoofing with moire detection, screen-context checks, phone-rectangle checks, passive liveness, stream liveness, and active challenge fallback

Detect V3 and legacy V1 routes are still present for compatibility and comparison, but the current UI is wired around V4.4.

---

## Current Runtime Summary

| Runtime | Main routes | Purpose | Status |
|---|---|---|---|
| Detect V4.4 | `/ws/scan-v4`, `/api/scan/v4`, `/ws/scan-v4/local` | Current production scan pipeline | Default |
| Detect V3 | `/ws/scan-v3`, `/api/scan/v3`, `/api/scan/v3/challenge` | Older strict + challenge path | Compatibility |
| Legacy V1 | `/api/scan`, `/api/enroll` | Baseline single-frame flow | Legacy |
| `dev/` scripts | `dev/detect-v*.py`, `dev/enroll-v*.py` | Research and iteration history | Experimental |

The dashboard scan-mode selector uses three frontend modes:

- `auto`: chooses `local_direct` when the page is opened on localhost/private LAN and the server camera is available; otherwise falls back to `browser_ws`
- `local_direct`: server webcam drives Detect V4.4 through `core/local_runner_v4.py`
- `browser_ws`: this browser camera sends JPEG frames to `/ws/scan-v4`

The `/phone` page and `/ws/phone-camera` transport are still available, but they are not the default Detect V4.4 dashboard path.

---

## Feature Highlights

- FastAPI backend with static dashboard served from `web/`
- SQLite student/session/attendance storage
- InsightFace face detection and 512-d ArcFace embeddings
- multi-angle V2 enrollment with per-angle validation
- Detect V4.4 backend-owned scan runtime shared by browser stream and local-direct modes
- active challenge fallback supporting `TURN_LEFT`, `TURN_RIGHT`, `LOOK_UP`, `LOOK_DOWN`, `OPEN_MOUTH`, and `CENTER_HOLD`
- attendance success overlay and in-camera challenge overlay
- debug "Tech Overlay" for runtime geometry and diagnostics
- student archive/restore flow with photo and evidence endpoints
- health, readiness, version, and capability endpoints for deployment checks

---

## Detect V4.4 Pipeline

```text
Browser camera or server webcam frame
  -> InsightFace face detect + embedding
  -> Streaming liveness tracker
  -> Rolling moire detector
  -> Screen-context detector
  -> Phone-rectangle detector
  -> Passive liveness check
  -> ArcFace match at V4 threshold
  -> If suspicious after identity match: start active challenge
  -> If challenge passes or no challenge needed: record attendance
```

### What V4.4 adds compared with V3

- Detect V4.4 is transport-agnostic: both browser-stream and local-direct modes feed the same backend runtime in `core/runtime_v4.py`
- moire decisions use the V4 detector in `core/detect_v4.py`
- screen-context analysis is enabled
- phone-rectangle analysis is enabled, including rolling decisions across frames
- challenge handling lives inside the runtime session instead of relying on a separate client-driven verification loop
- the frontend can render debug geometry from runtime diagnostics using the "Tech Overlay" toggle

### Active challenge behavior

V4.4 only starts a challenge after a matched identity looks suspicious. Current challenge types:

| Challenge | Description |
|---|---|
| `TURN_LEFT` | Turn face to the left |
| `TURN_RIGHT` | Turn face to the right |
| `LOOK_UP` | Tilt face upward |
| `LOOK_DOWN` | Tilt face downward |
| `OPEN_MOUTH` | Open mouth visibly |
| `CENTER_HOLD` | Hold face centered and still |

Current challenge state is in-memory and process-local.

---

## Project Structure

```text
face-reg-finnal-project/
|
|-- main.py
|-- config.py
|-- requirements.txt
|-- README.md
|-- REPORT_DETECT_V4_4_VI.md
|-- REPORT_DETECT_V4_4_VI.pdf
|-- start_tunnel.bat
|
|-- app/
|   |-- routes/
|       |-- attendance.py
|       |-- enrollment.py
|       |-- enrollment_v2.py
|       |-- live.py
|       |-- local_scan.py
|       |-- phone_camera.py
|       |-- scan_v3.py
|       |-- scan_v4.py
|       |-- system.py
|       |-- __init__.py
|
|-- core/
|   |-- anti_spoof.py
|   |-- camera.py
|   |-- challenge_v3.py
|   |-- database.py
|   |-- detect_v3.py
|   |-- detect_v4.py
|   |-- enrollment_v2.py
|   |-- face_engine.py
|   |-- liveness.py
|   |-- local_runner.py
|   |-- local_runner_v4.py
|   |-- moire.py
|   |-- runtime_v3.py
|   |-- runtime_v4.py
|   |-- schemas.py
|   |-- stream_scan_v3.py
|   |-- stream_scan_v4.py
|
|-- web/
|   |-- index.html
|   |-- phone.html
|   |-- style.css
|   |-- phone.js
|   |-- phone_style.css
|   |-- js/
|       |-- api.js
|       |-- enrollment.js
|       |-- history.js
|       |-- main.js
|       |-- scan.js
|       |-- session.js
|       |-- state.js
|       |-- students.js
|       |-- ui.js
|
|-- dev/
|   |-- README.md
|   |-- detect-v1.py
|   |-- detect-v2.py
|   |-- detect-v3.py
|   |-- detect-v4.0.py
|   |-- detect-v4.1.py
|   |-- detect-v4.2.py
|   |-- detect-v4.3.py
|   |-- detect-v4.4.py
|   |-- enroll-v1.py
|   |-- enroll-v2.py
|
|-- tests/
|   |-- run_test.py
|   |-- test_core.py
|   |-- test_detect_v4.py
|   |-- test_v2_v3.py
|
|-- models/
|-- database/
|-- logs/
```

## Repository Map

The table below clarifies which paths are tracked in Git and which are local-only runtime data.

| Path | Tracked in Git | Notes |
|---|---|---|
| `app/`, `core/`, `web/`, `tests/` | Yes | Application source code |
| `main.py`, `config.py`, `requirements.txt` | Yes | Entry point and config |
| `dev/` | Yes | Research scripts and iteration history |
| `.env.example` | Yes | Environment variable reference with no secrets |
| `pytest.ini`, `.gitignore`, `.gitattributes` | Yes | Repo tooling |
| `start_tunnel.bat` | Yes | Development convenience script |
| `models/` | No | Downloaded model files; too large for Git |
| `database/` | No | SQLite database and backups; local runtime data |
| `logs/` | No | Evidence images, face crops, and runtime logs |
| `.env` | No | Real environment variables; never commit |
| `tunnel_log*.txt` | No | Cloudflared output |

Git safety: `.gitignore` reduces the chance of accidentally committing local data, but it is not a security boundary. Sensitive values should be managed through environment variables or a secrets manager, not committed files.

---

Runtime data is stored in:

- `database/attendance.db`
- `database/backups/`
- `logs/evidence/`
- `logs/face_crops/`

---

## Requirements

- Python 3.10 or newer
- Conda or virtualenv recommended
- modern browser with camera permission
- optional NVIDIA GPU for faster ONNX inference

Python 3.10+ is required because the codebase uses modern type syntax such as `str | None`.

---

## Installation

### 1. Create an environment

```bash
conda create -n face-att python=3.11
conda activate face-att
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare models

InsightFace uses `models/` as its model root. The project also expects:

- `models/face_landmarker.task` for stream liveness

Optional download:

```bash
curl.exe -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

If you want GPU inference, install a matching `onnxruntime-gpu` build instead of CPU-only `onnxruntime`.

---

## Running The App

Start the FastAPI server:

```bash
python main.py
```

Open:

```text
http://localhost:8000/
```

Phone camera page:

```text
http://localhost:8000/phone
```

Useful health endpoints:

- `GET /health`
- `GET /ready`
- `GET /version`
- `GET /api/system/status`
- `GET /api/system/capabilities`

Interactive API docs are available at `/docs` when `API_DOCS_ENABLED=true`.

### Cloudflare tunnel

```bash
start_tunnel.bat
```

or:

```bash
cloudflared tunnel --url http://localhost:8000
```

This is useful when camera permissions require HTTPS on a remote device.

---

## API Summary

### Session endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/session/start` | Start a session |
| `POST` | `/api/session/end` | End the active session |
| `GET` | `/api/session/active` | Get the current active session |
| `GET` | `/api/session/{session_id}/result` | Get present/absent results for a session |
| `GET` | `/api/sessions` | List past sessions |
| `GET` | `/api/session/attendance` | Get current-session attendance list |

### Student endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/students?view=active|archived|all` | List students |
| `GET` | `/api/students/{student_id}` | Student details, embedding count, history |
| `PUT` | `/api/students/{student_id}` | Update name or class |
| `DELETE` | `/api/students/{student_id}` | Archive student |
| `POST` | `/api/students/{student_id}/restore` | Restore archived student |
| `GET` | `/api/students/{student_id}/photo` | Serve stored face crop |
| `GET` | `/api/evidence/{filename}` | Serve attendance evidence image |

### Enrollment endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/enroll` | Legacy single-image enrollment |
| `POST` | `/api/enroll/v2` | Multi-angle enrollment with front/left/right images |
| `POST` | `/api/enroll/v2/validate` | Validate one angle before saving |

### Detect V4.4 endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/scan/v4` | Single-frame V4 compatibility scan |
| `WS` | `/ws/scan-v4` | Default browser-camera V4 stream |
| `POST` | `/api/scan/v4/local/start` | Start server-webcam local-direct V4 runner |
| `POST` | `/api/scan/v4/local/stop` | Stop local-direct V4 runner |
| `GET` | `/api/scan/v4/local/status` | Get local-direct V4 status |
| `WS` | `/ws/scan-v4/local` | Subscribe to local-direct V4 events |

### Compatibility endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/scan` | Legacy V1 scan |
| `POST` | `/api/scan/v3` | Single-frame Detect V3 scan |
| `WS` | `/ws/scan-v3` | Detect V3 browser stream |
| `POST` | `/api/scan/v3/challenge` | Verify V3 challenge frames |
| `POST` | `/api/scan/v3/local/start` | Start V3 local-direct runner |
| `POST` | `/api/scan/v3/local/stop` | Stop V3 local-direct runner |
| `GET` | `/api/scan/v3/local/status` | Get V3 local-direct status |
| `WS` | `/ws/scan-v3/local` | Subscribe to V3 local-direct events |

### Camera and system endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/live/stream` | Server webcam MJPEG stream |
| `GET` | `/api/live/phone-stream` | MJPEG stream from phone uploader |
| `GET` | `/api/phone/status` | Phone transport status |
| `GET` | `/api/phone/latest` | Latest phone frame |
| `WS` | `/ws/phone-camera` | Phone frame upload transport |
| `GET` | `/api/system/status` | Runtime system summary |
| `GET` | `/api/system/capabilities` | API versions, modes, and thresholds |

---

## Configuration

Configuration is split across two places:

- `config.py` for app paths, server flags, camera options, enrollment rules, and shared V3 stream settings
- `core/detect_v4.py` for Detect V4.4 scoring thresholds and detector constants

### Common app settings in `config.py`

| Setting | Default | Purpose |
|---|---:|---|
| `APP_VERSION` | `4.4.0` | App version reported by `/version` |
| `HOST` | `0.0.0.0` | Uvicorn bind host |
| `PORT` | `8000` | Uvicorn bind port |
| `AUTO_PRELOAD_MODELS` | `True` | Warm up face engine during startup |
| `AUTO_LOAD_EMBEDDING_CACHE` | `True` | Load student embeddings during warmup |
| `AUTO_START_CAMERA` | `True` | Start server webcam on boot |
| `CAMERA_REQUIRED` | `False` | Mark camera failure as fatal when true |
| `UVICORN_RELOAD` | `DEBUG` | Enable hot reload in development |
| `API_DOCS_ENABLED` | `DEBUG` | Expose `/docs`, `/redoc`, and `/openapi.json` |
| `CAMERA_SOURCE` | `0` | Server webcam source |
| `DETECT_V3_STREAM_ENABLED` | `True` | Enable websocket streaming scan routes |
| `DETECT_V3_STREAM_TARGET_FPS` | `10` | Browser frame send target |
| `DETECT_V3_STREAM_DETECT_FPS` | `10` | Backend detect cadence |
| `ENROLL_V2_BLUR_MIN` | `80.0` | Minimum blur quality for enrollment |
| `ENROLL_V2_POSE_FRONT_MAX_DISP` | `0.12` | Max front nose displacement |
| `ENROLL_V2_POSE_TURN_THRESHOLD` | `0.04` | Min turn displacement for left/right |

### Detect V4.4 thresholds in `core/detect_v4.py`

| Setting | Default | Purpose |
|---|---:|---|
| `V4_COSINE_THRESHOLD` | `0.52` | V4 face-match threshold |
| `MOIRE_SCREEN_THRESHOLD` | `0.60` | Suspicious moire threshold |
| `MOIRE_BLOCK_THRESHOLD` | `0.45` | Hard screen block threshold |
| `SCREEN_CONTEXT_WEIGHT` | `0.35` | Weight for context scoring |
| `SCREEN_CONTEXT_STRONG_THRESHOLD` | `0.78` | Strong screen-context trigger |
| `PHONE_RECT_CONTEXT_SCALE` | `2.80` | ROI expansion around face |
| `PHONE_RECT_VERTICAL_RATIO` | `1.6` | Portrait-style ROI ratio |
| `PHONE_RECT_SUSPICIOUS_THRESHOLD` | `0.38` | Suspicious phone-rectangle threshold |
| `PHONE_RECT_STRONG_THRESHOLD` | `0.58` | Strong phone-rectangle threshold |

If you change V4 behavior, update both the code and this README so the documented thresholds stay honest.

---

## Testing

### Stable automated tests

These tests do not require a camera or files outside the repository:

```bash
python -m pytest tests/test_detect_v4.py tests/test_v2_v3.py -q
```

### Full repository test command

```bash
python -m pytest -q
```

This command also discovers `tests/test_core.py`.

`tests/test_core.py` is an integration test that depends on:
- an external video directory at `C:\Users\ADMIN\Desktop\Projects\face-attendance\test_video\`
- a live database and `logs/` directory, which means it writes real runtime data

Behavior of `tests/test_core.py`:
- if the external video directory does not exist, the test skips itself
- if the directory exists, the test runs and writes to local runtime data

Run that integration test explicitly when you want to validate the end-to-end flow:

```bash
python -m pytest tests/test_core.py -m integration -s
```

### Legacy smoke runner

Writes a result log to `tests/test_result.txt` (ignored by `.gitignore`):

```bash
python tests/run_test.py
```

### Optional frontend syntax check (requires Node.js)

```bash
node --check web/js/main.js
node --check web/js/scan.js
node --check web/js/enrollment.js
```

### Manual smoke checklist

- start a session from the dashboard
- enroll a student with the V2 front/left/right flow
- verify `Auto` picks the expected scan mode
- test `Browser Stream` on a remote browser
- test `Local Direct` on the machine that has the webcam attached
- verify attendance success overlay and challenge overlay both render
- toggle the `Tech Overlay` and confirm geometry/diagnostics appear
- end the session and inspect history details

---

## Known Limitations

- challenge state is in-memory and process-local, so multiple app workers would need shared state
- thresholds still need real-camera tuning for lighting, compression, and replay conditions
- `Local Direct` only makes sense when the camera is physically attached to the server machine
- V3 and V4 coexist in the codebase, so route coverage is broader than the default UI path
- runtime data under `database/`, `logs/`, and `models/` can grow quickly during testing

For implementation details behind Detect V4.4, see the research report in `dev/detect-v4.4.py` and the inline architecture comments in `core/detect_v4.py` and `core/runtime_v4.py`.

The `dev/` directory and its scripts are intentional artifacts documenting the V1 to V4.4 research progression. They are tracked in Git and are not junk files.
