# Face Attendance System

A real-time automated attendance system using face recognition. Built on InsightFace ArcFace embeddings with anti-spoofing, served through a FastAPI backend and a Material Design 3 web dashboard.

---

## Recognition Pipeline

```
Camera Frame
    |
    v
[1] Detection        InsightFace (buffalo_l)      Bounding box + 5 facial landmarks
    |
    v
[2] Alignment        ArcFace alignment            Normalize face to 112x112 px (compensates for tilt/rotation)
    |
    v
[3] Feature          ArcFace backbone             512-dimensional embedding vector
    Extraction
    |
    v
[4] Liveness Check   Anti-spoof model             Distinguish real face (3D) from spoofed image (2D print/screen)
    |
    v
[5] Matching         Cosine similarity >= 0.45    Compare against stored embeddings in database -> mark attendance
```

---

## Project Structure

```
face-reg-final-project/
|
|-- main.py                     Entry point — FastAPI application server
|-- config.py                   System-wide configuration (thresholds, paths, ports)
|-- requirements.txt            Python dependencies
|-- start_tunnel.bat            Cloudflare tunnel launcher script
|
|-- app/                        API layer
|   |-- routes/
|       |-- attendance.py       Session management (/api/session/*) and scan endpoint (/api/scan)
|       |-- enrollment.py       Student registration (/api/enroll, /api/students/*)
|       |-- live.py             MJPEG camera stream (/api/live/stream)
|       |-- phone_camera.py     WebSocket phone camera relay (/ws/phone-camera)
|       |-- system.py           System status endpoint (/api/system/status)
|
|-- core/                       Business logic layer
|   |-- face_engine.py          FaceEngine: detection, alignment, embedding, matching
|   |-- anti_spoof.py           Liveness detection (LBP texture + gradient + skin tone analysis)
|   |-- database.py             SQLite data access layer (students, sessions, attendance)
|   |-- camera.py               Webcam streaming (MJPEG)
|   |-- schemas.py              Pydantic data models
|
|-- web/                        Frontend
|   |-- index.html              Main dashboard (Material Design 3 dark theme, sidebar layout)
|   |-- app.js                  Dashboard logic (session control, scan, real-time log)
|   |-- style.css               Dashboard styles
|   |-- phone.html              Mobile camera client
|   |-- phone.js                WebSocket camera streaming logic
|   |-- phone_style.css         Mobile client styles
|
|-- dev/                        Standalone development tools
|   |-- enroll.py               Register a face via webcam (3-second countdown, auto-capture)
|   |-- detect.py               Live detection viewer (bounding box, name, confidence, FPS)
|
|-- tests/
|   |-- test_core.py            Unit tests for face engine and database layer
|   |-- run_test.py             Test runner
|
|-- models/                     Pre-trained model files (InsightFace buffalo_l)
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

These scripts run independently of the web server using the core engine directly.

### Register a New Face

```bash
python dev/enroll.py
```

Prompts for student ID, name, and class. Opens the webcam with a 3-second countdown, captures the best frame with a detected face, and saves the embedding to the database.

### Live Detection View

```bash
python dev/detect.py
```

Opens the webcam with a real-time overlay showing bounding boxes, recognized name, student ID, cosine similarity score, and FPS counter.

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot to `dev/screenshots/` |
| `R` | Reload face embedding cache from database |

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

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `insightface` | Face detection and ArcFace feature extraction |
| `onnxruntime` / `onnxruntime-gpu` | ONNX model inference |
| `fastapi` | Web API framework |
| `uvicorn[standard]` | ASGI server with WebSocket support |
| `opencv-python` | Camera capture and image processing |
| `numpy` | Numerical operations and vector arithmetic |
| `loguru` | Structured logging |

---

## How It Works

**Enrollment:** A photo is passed through the InsightFace pipeline to produce a 512-dimensional embedding vector. This vector is stored in the SQLite database alongside the student record.

**Attendance:** Each camera frame goes through the same pipeline. The resulting embedding is compared against all stored embeddings using cosine similarity. If the highest similarity score exceeds the configured threshold (default 0.45), the student is marked present in the active session. A liveness check runs before matching to reject printed photos or screen-displayed images.

**Similarity Threshold:** Setting a higher threshold requires a closer match and reduces false positives, but may cause legitimate users to fail recognition if their appearance varies (lighting, angle). The default of 0.45 balances accuracy and usability.
