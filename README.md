# Face Attendance System

A real-time automated attendance system using face recognition with multi-layer anti-spoofing defense. Built on InsightFace ArcFace embeddings, served through a FastAPI backend and a Material Design 3 web dashboard.

---

## Recognition Pipeline

```
Camera Frame
    |
    v
[1] Detection        InsightFace RetinaFace       Bounding box + 5 facial landmarks
    |
    v
[2] Alignment        ArcFace alignment            Normalize face to 112x112 px
    |
    v
[3] Feature          ArcFace (buffalo_l)           512-dimensional embedding vector
    Extraction
    |
    v
[4] Anti-Spoof       Layer 1: Moire FFT           Screen replay detection (passive)
    Gate              Layer 2: Challenge-Response   Random action verification (active)
                      Layer 3: Multi-frame EAR      Blink + movement tracking
    |
    v
[5] Matching         Cosine similarity >= 0.52     Compare against stored embeddings -> mark attendance
```

---

## Anti-Spoofing Architecture (V2/V3)

The system implements a defense-in-depth strategy against presentation attacks:

| Layer | Method | Type | Detects |
|-------|--------|------|---------|
| 1 | Moire Pattern (FFT) | Passive | Video replays on LCD/OLED screens |
| 2 | Challenge-Response | Active | Pre-recorded video attacks |
| 3 | Multi-frame Liveness (EAR) | Passive | Printed photos, static images |

All layers operate using standard 2D webcams. No depth camera or infrared sensor required.

See [dev/README.md](dev/README.md) for full technical documentation of the anti-spoofing methodology.

---

## Project Structure

```
face-reg-final-project/
|
|-- main.py                     Entry point - FastAPI application server
|-- config.py                   System-wide configuration (thresholds, paths, ports)
|-- requirements.txt            Python dependencies
|-- start_tunnel.bat            Cloudflare tunnel launcher script
|
|-- app/                        API layer
|   |-- routes/
|       |-- attendance.py       Session management (/api/session/*) and scan endpoint
|       |-- enrollment.py       Student registration (/api/enroll, /api/students/*)
|       |-- live.py             MJPEG camera stream (/api/live/stream)
|       |-- phone_camera.py     WebSocket phone camera relay (/ws/phone-camera)
|       |-- system.py           System status endpoint (/api/system/status)
|
|-- core/                       Business logic layer
|   |-- face_engine.py          FaceEngine: detection, alignment, embedding, matching
|   |-- anti_spoof.py           Liveness detection (LBP texture + gradient analysis)
|   |-- liveness.py             Multi-frame liveness (EAR blink + head movement)
|   |-- database.py             SQLite data access layer (students, sessions, attendance)
|   |-- camera.py               Webcam streaming (MJPEG)
|   |-- schemas.py              Pydantic data models
|
|-- web/                        Frontend
|   |-- index.html              Main dashboard (Material Design 3 dark theme)
|   |-- app.js                  Dashboard logic (session control, scan, real-time log)
|   |-- style.css               Dashboard styles
|   |-- phone.html              Mobile camera client
|   |-- phone.js                WebSocket camera streaming logic
|   |-- phone_style.css         Mobile client styles
|
|-- dev/                        Standalone development tools
|   |-- README.md               Technical reference (anti-spoofing methodology)
|   |-- enroll.py               V1 enrollment: single-photo, 3s countdown
|   |-- enroll-v2.py            V2 enrollment: multi-angle (3 poses), multi-frame averaging
|   |-- detect.py               V1 detection: recognition + basic liveness
|   |-- detect-v2.py            V2 detection: + Moire FFT + Challenge-Response
|   |-- detect-v3.py            V3 detection: + stricter threshold + performance optimized
|
|-- tests/
|   |-- test_core.py            Unit tests for face engine and database layer
|   |-- run_test.py             Test runner
|
|-- models/                     Pre-trained model files (auto-downloaded)
|   |-- buffalo_l/              InsightFace RetinaFace + ArcFace
|   |-- face_landmarker.task    MediaPipe FaceLandmarker (download separately)
|
|-- database/                   SQLite database (auto-created at runtime)
|-- logs/                       Evidence images and face crops (auto-created at runtime)
```

---

## Installation

### Requirements

- Python 3.9 or higher
- CUDA-capable GPU recommended (NVIDIA RTX series tested)
- Conda environment recommended

### Setup

```bash
conda create -n face-att python=3.10
conda activate face-att

pip install -r requirements.txt
```

**GPU acceleration:** Replace `onnxruntime` with `onnxruntime-gpu` in `requirements.txt` before installing.

**MediaPipe model (required for V2/V3 anti-spoofing):**

```bash
# Download face_landmarker.task to models/ directory
curl -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### Running the Server

```bash
python main.py
```

Open the dashboard at: `http://localhost:8000`

### Remote Access / Mobile Camera

```bash
# Start Cloudflare tunnel (Windows)
start_tunnel.bat

# Or manually:
cloudflared tunnel --url http://localhost:8000
```

Use the generated `https://*.trycloudflare.com` URL on the mobile device, navigate to `/phone`, and select **Phone** as the camera source in the dashboard.

---

## Development Tools

Standalone scripts in `dev/` that run independently of the web server.

### Enrollment

| Script | Method | Embeddings | Accuracy |
|--------|--------|-----------|----------|
| `python dev/enroll.py` | Single photo, 3s countdown | 1 | Baseline |
| `python dev/enroll-v2.py` | 3 angles x multi-frame averaging | 3 | Higher |

**Recommended:** Use `enroll-v2.py` for production. Guides the user through frontal, left, and right poses, collects 3-8 high-quality frames per pose, averages them, and stores 3 robust embeddings per student.

### Detection

| Script | Anti-Spoof | Threshold | FPS (GPU) |
|--------|-----------|-----------|-----------|
| `python dev/detect.py` | EAR only | 0.45 | ~26 |
| `python dev/detect-v2.py` | Moire + Challenge + EAR | 0.45 | ~16 |
| `python dev/detect-v3.py` | Moire + Challenge + EAR (optimized) | 0.52 | ~22 |

**Recommended pairing:**
- `enroll-v2.py` + `detect-v3.py` for maximum accuracy and security
- `enroll.py` + `detect-v2.py` for backward compatibility

### Controls (all detect scripts)

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot to `dev/screenshots/` |
| `R` | Reload face embeddings cache |
| `L` | Toggle landmark visualization |
| `I` | Toggle information panel |
| `M` | Toggle moire overlay (V2/V3 only) |
| `E` | Print embedding statistics (V3 only) |

---

## Configuration

All configuration is defined in `config.py`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MATCH_THRESHOLD` | `0.45` | Minimum cosine similarity to accept a match |
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Server bind address |
| `ENROLL_SCAN_DURATION` | `5` | Camera capture duration during enrollment (seconds) |
| `ENROLL_MIN_FRAMES` | `3` | Minimum frames with detected face required to enroll |
| `QUALITY_MAX_BLUR` | `100.0` | Laplacian variance threshold for blur rejection |
| `CAMERA_SOURCE` | `0` | Webcam index, or RTSP URL for IP cameras |

Note: `detect-v3.py` overrides the cosine threshold to 0.52. This stricter threshold is designed for multi-angle enrolled databases (3+ embeddings per student).

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `insightface` | Face detection (RetinaFace) and embedding (ArcFace) |
| `onnxruntime` / `onnxruntime-gpu` | ONNX model inference |
| `mediapipe` | FaceLandmarker for liveness and pose verification |
| `fastapi` | Web API framework |
| `uvicorn[standard]` | ASGI server with WebSocket support |
| `opencv-python` | Camera capture and image processing |
| `numpy` | Numerical operations, FFT, vector arithmetic |
| `loguru` | Structured logging |

---

## How It Works

**Enrollment:** A face is captured and passed through the InsightFace pipeline to produce a 512-dimensional embedding vector. In V2 enrollment, this process is repeated at three head poses (frontal, left, right), with multiple frames averaged per pose to reduce noise. All embeddings are stored in the SQLite database.

**Attendance:** Each camera frame goes through the detection pipeline. Before matching, the anti-spoof gate checks for screen replay attacks (Moire FFT analysis), challenges suspicious inputs with random actions (blink, turn, smile), and verifies natural facial dynamics over time (multi-frame liveness). Only faces that pass all checks proceed to cosine similarity matching against stored embeddings.

**Similarity Threshold:** Setting a higher threshold requires a closer match and reduces false positives, but may cause legitimate users to fail recognition if their appearance varies. V3 uses 0.52 (stricter), which is viable because multi-angle enrollment produces more robust centroid embeddings that score higher for genuine matches.
