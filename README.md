# Face Attendance System

Real-time face attendance system built with FastAPI, InsightFace ArcFace embeddings, SQLite, and a vanilla JavaScript web dashboard. The current production web flow uses multi-angle enrollment, Scan V3 recognition, WebSocket streaming liveness, passive anti-spoofing, and active challenge verification for suspicious scans.

---

## Overview

The project has two main usage paths:

| Path | Purpose | Status |
|---|---|---|
| Web dashboard | Production-oriented attendance UI and API workflow | Primary |
| `dev/` scripts | Standalone research and debugging tools | Reference / experimental |

The web dashboard is the current default path. It uses:

- `/api/enroll/v2` for camera/manual multi-angle enrollment.
- `/ws/scan-v3` for default browser-camera attendance scanning with continuous liveness tracking.
- `/api/scan/v3` for compatibility single-frame scanning.
- `/api/scan/v3/challenge` for suspicious-frame active verification.
- Modular frontend files under `web/js/`.

---

## Production Web Flow

```text
Browser camera frame stream over WebSocket
  -> MediaPipe FaceLandmarker Tasks API liveness on every frame
  -> InsightFace face detection at 10 FPS
  -> Match liveness track back to each detected bbox
  -> ArcFace embedding extraction
  -> Rolling moire score check every 3 detect cycles
  -> Passive liveness score check
  -> If no blink after timeout: block as spoof
  -> Cosine match against active embeddings
  -> If clean: mark attendance
  -> If suspicious: create active challenge
  -> If challenge passes: mark attendance
```

Challenge is only triggered for suspicious-but-not-hard-blocked scans. A hard spoof signal is blocked before matching. A clean real face should pass directly without challenge.

Current challenge actions:

| Action | Backend signal |
|---|---|
| `turn_left` | InsightFace 5-point landmark nose displacement |
| `turn_right` | InsightFace 5-point landmark nose displacement |
| `blink` | MediaPipe FaceMesh EAR across multiple browser frames |

Not implemented in production web yet: smile, nod, wink, multi-step challenge sequences.

---

## Web Features

### Dashboard

- Start and end attendance sessions.
- Camera source selection: browser camera for live Scan V3, plus server webcam, phone camera, and demo video utilities.
- Auto scan using continuous WebSocket Scan V3.
- Attendance log showing student names, IDs, and classes.
- In-camera success confirmation overlay with face evidence, student info, confidence, session, and countdown.
- In-camera challenge overlay with action cue for left/right/blink.

### Enrollment

- Browser-camera-first enrollment.
- Manual upload fallback.
- V2 multi-angle flow: front, left, right.
- Per-angle validation through `/api/enroll/v2/validate`.
- Multi-frame capture per angle.
- All-or-nothing save: all three angles must pass before replacing embeddings.
- Re-enrollment updates metadata and replaces embeddings transactionally.

### Students

- Active / Archived views.
- Student detail modal.
- Edit name and class.
- Re-enroll face data.
- Archive and restore students.
- Student photo endpoint with path traversal protection.

### History

- Session list.
- Session result detail from `/api/session/{id}/result`.
- Present/absent breakdown based on the session student snapshot.

---

## Anti-Spoofing

Production web anti-spoofing is implemented in layers:

| Layer | File | Purpose |
|---|---|---|
| Moire FFT | `core/moire.py` | Detect screen replay artifacts |
| Streaming liveness | `core/liveness.py` | Dev-style continuous EAR blink and movement verification |
| Passive liveness | `core/anti_spoof.py` | Texture, reflection, and color heuristics |
| Active challenge | `core/challenge_v3.py` | Require left/right/blink response when suspicious |
| Scan orchestration | `core/detect_v3.py` | Decide pass, block, challenge, or unknown |

Current decision bands are configured in `config.py`:

```python
DETECT_V3_MOIRE_BLOCK_THRESHOLD = 0.25
DETECT_V3_MOIRE_CHALLENGE_THRESHOLD = 0.42
DETECT_V3_LIVENESS_BLOCK_THRESHOLD = 0.32
DETECT_V3_LIVENESS_CHALLENGE_THRESHOLD = 0.50
```

The standalone research scripts in `dev/` contain additional challenge ideas such as smile, nod, and wink. Production web now uses the same core browser stream shape as `dev/detect-v3.py`: continuous FaceLandmarker blink tracking, bbox-aware liveness lookup, 10 FPS recognition cadence, and rolling moire. It still intentionally limits active challenge actions to blink, turn left, and turn right.

---

## Project Structure

```text
face-reg-finnal-project/
|
|-- main.py                       FastAPI entry point
|-- config.py                     Paths, thresholds, server config
|-- requirements.txt              Python dependencies
|-- start_tunnel.bat              Cloudflare tunnel helper
|
|-- app/
|   |-- routes/
|       |-- attendance.py         Sessions, legacy /api/scan
|       |-- enrollment.py         V1 enroll, students CRUD, photo/evidence
|       |-- enrollment_v2.py      V2 multi-angle enrollment endpoints
|       |-- live.py               Server webcam MJPEG stream
|       |-- phone_camera.py       Phone camera WebSocket + MJPEG stream
|       |-- scan_v3.py            Scan V3 and challenge endpoints
|       |-- system.py             System status and capabilities
|       |-- __init__.py           Route aggregation
|
|-- core/
|   |-- anti_spoof.py             Passive liveness heuristics
|   |-- camera.py                 Server webcam wrapper
|   |-- challenge_v3.py           Active challenge verification
|   |-- database.py               SQLite data access and transactions
|   |-- detect_v3.py              Scan V3 orchestration
|   |-- enrollment_v2.py          Multi-angle enrollment service
|   |-- face_engine.py            InsightFace detection, embeddings, matching
|   |-- liveness.py               MediaPipe blink/movement liveness trackers
|   |-- moire.py                  FFT moire detector
|   |-- schemas.py                Dataclass result models
|   |-- stream_scan_v3.py         WebSocket Scan V3 stream session
|
|-- web/
|   |-- index.html                Main dashboard
|   |-- style.css                 Dashboard styles
|   |-- phone.html                Phone camera client
|   |-- phone.js                  Phone WebSocket camera sender
|   |-- phone_style.css           Phone client styles
|   |-- js/
|       |-- api.js                Fetch wrapper
|       |-- enrollment.js         V2 camera/manual enrollment UI
|       |-- history.js            Session history UI
|       |-- main.js               Frontend module entry
|       |-- scan.js               Camera source, Scan V3, challenge, success overlay
|       |-- session.js            Start/end session UI
|       |-- state.js              Shared frontend state
|       |-- students.js           Students CRUD UI
|       |-- ui.js                 Theme, tabs, toast, modal helpers
|
|-- dev/
|   |-- README.md                 Research notes
|   |-- enroll.py                 Standalone V1 enrollment
|   |-- enroll-v2.py              Standalone V2 multi-angle enrollment
|   |-- detect.py                 Standalone V1 detection
|   |-- detect-v2.py              Standalone V2 detection research
|   |-- detect-v3.py              Standalone V3 detection research
|
|-- tests/
|   |-- run_test.py               Test runner
|   |-- test_core.py              Core smoke tests
|   |-- test_v2_v3.py             V2/V3 API and service tests
|
|-- models/                       InsightFace models are auto-downloaded here
|-- database/                     Runtime SQLite DB and backups
|-- logs/                         Runtime evidence images and face crops
```

`web/app.js` may still exist as legacy code, but the dashboard entry point is `web/js/main.js`.

---

## Installation

### Requirements

- Python 3.9 or higher.
- Conda environment recommended.
- CUDA-capable NVIDIA GPU recommended, but CPU can run for development.
- Modern browser with camera permissions for browser-camera enrollment/scan.

### Setup

```bash
conda create -n face-att python=3.10
conda activate face-att
pip install -r requirements.txt
```

For GPU acceleration, install an appropriate `onnxruntime-gpu` build instead of CPU-only `onnxruntime`.

### MediaPipe Notes

Production streaming liveness uses MediaPipe FaceLandmarker Tasks API from the `mediapipe` package.

`models/face_landmarker.task` is required for `/ws/scan-v3` streaming liveness, the optional `MultiFrameLiveness` path, and the research scripts in `dev/`.

Optional FaceLandmarker download:

```bash
curl -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

---

## Running

```bash
python main.py
```

Open:

```text
http://localhost:8000
```

Phone camera page:

```text
http://localhost:8000/phone
```

Cloudflare tunnel:

```bash
start_tunnel.bat
```

Or manually:

```bash
cloudflared tunnel --url http://localhost:8000
```

When using Cloudflare, browser camera permissions require HTTPS, so the tunnel URL is useful for camera testing.

---

## API Reference

### Sessions

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/session/start` | Start active session |
| `POST` | `/api/session/end` | End active session and compute summary |
| `GET` | `/api/session/active` | Get current active session |
| `GET` | `/api/session/{session_id}/result` | Session present/absent detail |
| `GET` | `/api/sessions` | List sessions |
| `GET` | `/api/session/attendance` | Current session attendance |

### Students

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/students?view=active|archived|all` | List students by status |
| `GET` | `/api/students/{student_id}` | Student detail, embedding count, history |
| `PUT` | `/api/students/{student_id}` | Update name/class metadata |
| `DELETE` | `/api/students/{student_id}` | Archive student, preserve embeddings |
| `POST` | `/api/students/{student_id}/restore` | Restore archived student |
| `GET` | `/api/students/{student_id}/photo` | Serve student face crop |
| `GET` | `/api/evidence/{filename}` | Serve attendance evidence image |

### Enrollment

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/enroll` | Legacy V1 single-image enrollment |
| `POST` | `/api/enroll/v2` | V2 front/left/right multi-angle enrollment |
| `POST` | `/api/enroll/v2/validate` | Validate one angle image before save |

### Scan

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/scan` | Legacy V1 scan |
| `POST` | `/api/scan/v3` | Compatibility single-frame Scan V3 API |
| `WS` | `/ws/scan-v3` | Default browser-camera Scan V3 stream with stateful liveness |
| `POST` | `/api/scan/v3/challenge` | Verify suspicious scan with frame batch |

### Camera / System

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/live/stream` | Server webcam MJPEG stream |
| `GET` | `/api/live/phone-stream` | Phone camera MJPEG stream |
| `GET` | `/api/phone/status` | Phone camera connection status |
| `GET` | `/api/phone/latest` | Latest phone frame if fresh |
| `WS` | `/ws/phone-camera` | Phone camera frame upload |
| `GET` | `/api/system/status` | Runtime status |
| `GET` | `/api/system/capabilities` | API versions and feature flags |

---

## Configuration

All main settings live in `config.py`.

| Parameter | Default | Purpose |
|---|---:|---|
| `COSINE_THRESHOLD` | `0.45` | Legacy/default match threshold |
| `DETECT_V3_COSINE_THRESHOLD` | `0.52` | Scan V3 strict match threshold |
| `DETECT_V3_MOIRE_SCREEN_THRESHOLD` | `0.30` | Legacy moire screen threshold |
| `DETECT_V3_MOIRE_BLOCK_THRESHOLD` | `0.25` | Hard block if moire score is lower |
| `DETECT_V3_MOIRE_CHALLENGE_THRESHOLD` | `0.42` | Challenge if moire score is below this band |
| `DETECT_V3_LIVENESS_BLOCK_THRESHOLD` | `0.32` | Hard block if passive liveness is lower |
| `DETECT_V3_LIVENESS_CHALLENGE_THRESHOLD` | `0.50` | Challenge if passive liveness is below this band |
| `DETECT_V3_CHALLENGE_TTL_SECONDS` | `10` | Challenge expiry window |
| `DETECT_V3_CHALLENGE_MIN_FRAMES` | `3` | Minimum challenge frames |
| `DETECT_V3_CHALLENGE_MAX_FRAMES` | `36` | Maximum challenge frames analyzed |
| `DETECT_V3_BLINK_EAR_CLOSED_THRESHOLD` | `0.21` | Blink closed-eye EAR threshold |
| `DETECT_V3_BLINK_EAR_OPEN_THRESHOLD` | `0.24` | Blink open-eye EAR threshold |
| `DETECT_V3_STREAM_TARGET_FPS` | `10` | Browser frame send target for `/ws/scan-v3` |
| `DETECT_V3_STREAM_DETECT_FPS` | `10` | Max recognition checks per second in the stream |
| `DETECT_V3_STREAM_MOIRE_EVERY_N_DETECT` | `3` | Run rolling moire analysis every N detect cycles |
| `DETECT_V3_STREAM_CLIENT_FRAME_WIDTH` | `960` | Browser frame max width for WebSocket scan |
| `DETECT_V3_STREAM_CLIENT_JPEG_QUALITY` | `0.82` | Browser JPEG quality target for WebSocket scan |
| `DETECT_V3_STREAM_MIN_TRACK_SECONDS` | `2.0` | Minimum stream observation time before requiring blink |
| `DETECT_V3_STREAM_MAX_CHECK_SECONDS` | `6.0` | Max time to wait for blink before spoof result |
| `ENROLL_V2_BLUR_MIN` | `80.0` | Minimum blur/quality score per angle |
| `ENROLL_V2_POSE_FRONT_MAX_DISP` | `0.12` | Max nose displacement for front |
| `ENROLL_V2_POSE_TURN_THRESHOLD` | `0.04` | Min displacement for left/right |
| `CAMERA_SOURCE` | `0` | Server webcam index or camera source |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

---

## Development / Research Tools

The `dev/` scripts are standalone tools and may contain research features not yet integrated into the web production path.

### Enrollment Scripts

| Script | Method | Notes |
|---|---|---|
| `python dev/enroll.py` | Single image | V1 baseline |
| `python dev/enroll-v2.py` | Front/left/right multi-angle | Research/reference for V2 flow |

### Detection Scripts

| Script | Anti-spoofing | Notes |
|---|---|---|
| `python dev/detect.py` | Basic liveness | V1 baseline |
| `python dev/detect-v2.py` | Moire + active challenge research | Experimental |
| `python dev/detect-v3.py` | Moire + challenge + stricter threshold | Research/reference |

Do not assume every `dev/` feature is active in the web API. The production web integration is defined by `core/detect_v3.py` and `core/challenge_v3.py`.

---

## Testing

Run JavaScript syntax checks:

```bash
node --check web/js/api.js
node --check web/js/state.js
node --check web/js/ui.js
node --check web/js/session.js
node --check web/js/students.js
node --check web/js/enrollment.js
node --check web/js/history.js
node --check web/js/scan.js
node --check web/js/main.js
```

Run focused V2/V3 tests:

```bash
python -m pytest tests/test_v2_v3.py -q
```

Run full tests:

```bash
python -m pytest tests -q
```

Manual browser smoke test:

- Hard refresh the dashboard with `Ctrl + F5`.
- Start a session.
- Enroll a student through browser camera.
- Scan through the web UI using `/ws/scan-v3`.
- Verify success overlay appears inside the camera frame.
- Verify default web scan asks for blink/liveness before marking attendance.
- Test suspicious conditions and verify challenge overlay for left/right/blink.
- End session and inspect history details.

---

## Known Limitations

- Challenge state is in-memory and process-local. Multiple FastAPI workers would require Redis or another shared store.
- Blink challenge uses regular 2D webcam frames. It improves replay resistance but is not equivalent to hardware depth sensing.
- Streaming/passive liveness thresholds require real camera tuning. Poor lighting, compression, or reflective glasses can still cause false rejects.
- Browser WebSocket scan depends on camera permission and network latency. Cloudflare tunnels can reduce effective FPS; the frontend uses backpressure to avoid flooding. Phone camera is not the default Scan V3 path.
- Some comments in older source files may contain mojibake from prior encoding issues; runtime behavior is unaffected.
