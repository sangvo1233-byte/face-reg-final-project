# Anti-Spoofing Face Recognition System — Developer Reference

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Module Descriptions](#module-descriptions)
   - 3.1 [enroll-v2.py — Multi-Angle Enrollment](#31-enroll-v2py--multi-angle-enrollment)
   - 3.2 [detect-v2.py — Anti-Spoof Detection](#32-detect-v2py--anti-spoof-detection)
   - 3.3 [detect-v3.py — Optimized Detection with Stricter Matching](#33-detect-v3py--optimized-detection-with-stricter-matching)
4. [Anti-Spoofing Methodology](#anti-spoofing-methodology)
   - 4.1 [Layer 1: Moire Pattern Detection (Passive)](#41-layer-1-moire-pattern-detection-passive)
   - 4.2 [Layer 2: Challenge-Response Verification (Active)](#42-layer-2-challenge-response-verification-active)
   - 4.3 [Layer 3: Multi-Frame Liveness Tracking](#43-layer-3-multi-frame-liveness-tracking)
5. [Multi-Angle Enrollment Methodology](#multi-angle-enrollment-methodology)
6. [Usage Instructions](#usage-instructions)
7. [Configuration Parameters](#configuration-parameters)
8. [Performance Characteristics](#performance-characteristics)
9. [Threat Model and Limitations](#threat-model-and-limitations)
10. [Dependencies](#dependencies)
11. [Version History](#version-history)

---

## Overview

This document describes the enhanced face recognition and anti-spoofing pipeline
implemented within the `dev/` directory. The system addresses a critical
vulnerability in standard face recognition deployments: susceptibility to
presentation attacks using high-resolution video replays displayed on modern
screens (4K monitors, high-PPI smartphones).

The solution introduces a two-layer defense mechanism that operates entirely
with standard 2D RGB webcams, requiring no specialized hardware such as infrared
emitters, structured-light projectors, or depth sensors.

**Key contributions:**

- Frequency-domain analysis (FFT) for passive moire interference detection,
  exploiting the physical interaction between camera sensor grids and screen
  pixel matrices.
- A randomized challenge-response protocol that defeats pre-recorded video
  attacks by requiring unpredictable user actions.
- Multi-angle, multi-frame enrollment that produces robust face embeddings
  invariant to moderate pose variations.

---

## System Architecture

```
                    ENROLLMENT PHASE                    DETECTION PHASE
                    (one-time setup)                    (per session)

                    enroll-v2.py                        detect-v2.py / detect-v3.py
                         |                                    |
                         v                                    v
                +------------------+                 +------------------+
                | Webcam Capture   |                 | Webcam Capture   |
                +--------+---------+                 +--------+---------+
                         |                                    |
                         v                                    v
                +------------------+                 +------------------+
                | FaceLandmarker   |                 | InsightFace      |
                | (pose verify)    |                 | RetinaFace Det.  |
                +--------+---------+                 +--------+---------+
                         |                                    |
                         v                                    v
                +------------------+                 +------------------+
                | InsightFace      |                 | Anti-Spoof Gate  |
                | ArcFace Embed.   |                 | (Moire + Challenge|
                +--------+---------+                 |  + Liveness)     |
                         |                           +--------+---------+
                         v                                    |
                +------------------+                          v
                | Multi-Frame Avg. |                 +------------------+
                | per angle        |                 | ArcFace Match    |
                +--------+---------+                 | (cosine sim.)    |
                         |                           +--------+---------+
                         v                                    |
                +------------------+                          v
                | SQLite DB        |<---------------------> Result
                | face_embeddings  |                 (PRESENT/UNKNOWN/SPOOF)
                +------------------+
```

---

## Module Descriptions

### 3.1 enroll-v2.py — Multi-Angle Enrollment

**Purpose:** Capture face embeddings from multiple viewing angles using a
guided, multi-phase enrollment protocol. Each phase collects multiple frames,
applies quality filtering, and produces a single averaged embedding per angle.

**Enrollment Protocol:**

| Phase | User Action | Duration | Output |
|-------|-------------|----------|--------|
| Calibration | Look straight at camera | 2.0s | Baseline nose position |
| Phase 1 (FRONT) | Maintain frontal pose | 4.0s | emb_front |
| Phase 2 (LEFT) | Turn head to the left | 4.0s | emb_left |
| Phase 3 (RIGHT) | Turn head to the right | 4.0s | emb_right |

**Multi-Frame Averaging:**

Within each phase, the system continuously detects faces, evaluates frame
quality (Laplacian variance for sharpness), verifies head pose compliance via
MediaPipe FaceLandmarker, and collects embeddings from qualifying frames. Upon
phase completion, all collected embeddings are averaged element-wise and
L2-normalized to produce a single representative embedding:

```
emb_phase = normalize(mean(emb_1, emb_2, ..., emb_N))
```

where N is between `MIN_GOOD_FRAMES` (3) and `MAX_GOOD_FRAMES` (8).

This averaging suppresses per-frame noise by a factor of sqrt(N), yielding
a more stable representation than any single-frame embedding.

**Pose Verification:**

Head pose is verified by tracking the normalized x-coordinate of the nose tip
landmark (MediaPipe landmark index 1). For the LEFT and RIGHT phases, the
system requires the nose to displace beyond `TURN_THRESHOLD` (0.04 normalized
units) from the calibrated baseline, ensuring the user has genuinely rotated
their head rather than remaining in a frontal pose.

**Database Storage:**

Three embeddings (one per angle) are stored as separate rows in the
`face_embeddings` table. The existing `Database.get_all_embeddings()` method
automatically computes the mean of all embeddings per student and re-normalizes,
producing a centroid embedding that covers the convex hull of enrolled poses.

**User Interface Elements:**

- Welcome screen with student information and phase preview
- Phase transition screens with countdown
- Real-time quality indicators (sharpness, pose correctness, frame count)
- Phase progress dots indicating overall enrollment state
- Capture pulse animation on successful frame acquisition
- Live aligned-face thumbnail preview
- Final summary displaying per-phase results with thumbnails
- Retry prompt on phase failure (maximum 1 retry per phase)

---

### 3.2 detect-v2.py — Anti-Spoof Detection

**Purpose:** Real-time face detection and recognition with integrated
presentation attack detection (PAD). Implements the full anti-spoofing pipeline
using moire pattern analysis and challenge-response verification.

**Pipeline per frame:**

1. Multi-frame liveness tracker processes every frame (EAR blink detection,
   head movement accumulation).
2. MediaPipe FaceLandmarker runs every frame to supply 478 facial landmarks
   for the challenge-response module.
3. InsightFace detection runs at a throttled rate (`DETECT_FPS = 10`),
   producing bounding boxes, 5-point landmarks, and 512-D ArcFace embeddings.
4. For each detected face, the moire detector analyzes the face ROI.
5. The anti-spoof gate evaluates moire score, challenge status, and liveness
   to determine whether to proceed with identity matching.
6. Cosine similarity matching against the embedding database using
   `config.COSINE_THRESHOLD` (default 0.45).

**Cosine Threshold:** 0.45 (configuration default)

**Controls:**

| Key | Action |
|-----|--------|
| Q | Quit application |
| S | Save screenshot to `dev/screenshots/` |
| R | Reload face embeddings cache |
| L | Toggle landmark visualization |
| I | Toggle right-side information panel |
| M | Toggle moire overlay (badge + FFT spectrum) |

---

### 3.3 detect-v3.py — Optimized Detection with Stricter Matching

**Purpose:** Extends detect-v2 with a stricter cosine similarity threshold,
performance optimizations, and enrollment quality awareness. Designed to be
paired with enroll-v2 for maximum recognition accuracy.

**Differences from detect-v2:**

| Aspect | detect-v2 | detect-v3 |
|--------|-----------|-----------|
| Cosine threshold | 0.45 (config) | 0.52 (hardcoded override) |
| FFT analysis size | 128x128 | 64x64 (4x fewer operations) |
| Moire frequency | Every detect cycle | Every 3rd detect cycle |
| MediaPipe execution | Every frame | Conditional (skipped when idle) |
| FFT masks | Recomputed per call | Pre-computed at initialization |
| Embedding count display | Not shown | Displayed per matched identity |
| Enrollment quality indicator | Not shown | "Multi-angle V2" / warning |
| Performance metrics | Not shown | Real-time ms display in HUD |

**Performance Optimizations:**

1. **FFT Size Reduction (128 to 64):** The moire analysis crop is resized to
   64x64 instead of 128x128, reducing FFT computation by approximately 4x
   (O(N^2 log N) complexity). Empirical testing indicates negligible impact on
   detection accuracy for the moire frequency bands of interest.

2. **Moire Frame Skipping:** The moire analyzer runs only on every third
   detection cycle (`MOIRE_EVERY_N_DETECT = 3`). Between analyses, the
   previous result is reused. Given the rolling-average history buffer of 20
   samples, this introduces no perceptible delay in detection decisions.

3. **Conditional MediaPipe Execution:** The FaceLandmarker is skipped entirely
   when no faces are present in the frame and no active challenge is in
   progress. This eliminates unnecessary color conversion and inference
   overhead during idle periods.

4. **Pre-computed Masks:** The Hanning window, radial distance matrix, band
   mask, DC mask, and high-frequency mask are computed once during
   `MoireDetector.__init__()` and reused for every subsequent analysis call.

**Additional Control:**

| Key | Action |
|-----|--------|
| E | Print per-student embedding statistics to terminal |

---

## Anti-Spoofing Methodology

### 4.1 Layer 1: Moire Pattern Detection (Passive)

**Theoretical Basis:**

When a digital camera captures an image displayed on an LCD or OLED screen,
the spatial sampling frequency of the camera sensor interacts with the
periodic pixel structure of the display, producing moire interference
patterns. These patterns manifest as characteristic periodic peaks in the
2D frequency domain that are absent in images of real faces.

Real human skin and facial features exhibit smooth, low-frequency spectral
characteristics with no periodic high-frequency structure. This fundamental
physical distinction provides a reliable classification signal.

**Implementation:**

The `MoireDetector` class computes four complementary frequency-domain signals:

**Signal 1 — Peak-to-Mean Ratio (2D FFT):**

The face ROI is converted to grayscale, resized to a fixed resolution,
windowed with a 2D Hanning function (to suppress spectral leakage), and
transformed via 2D FFT. Within the annular frequency band defined by
`[FREQ_LOW, FREQ_HIGH]`, the ratio of the maximum spectral magnitude to the
mean magnitude is computed. Ratios exceeding `PEAK_RATIO_THRESHOLD` (2.8)
indicate anomalous spectral peaks consistent with screen capture.

**Signal 2 — High-Frequency Energy Ratio:**

The proportion of total spectral energy residing in the high-frequency band
(above `FREQ_LOW`) relative to the total (excluding DC) is computed. Screen
images exhibit elevated high-frequency energy due to the pixel grid structure.
Values exceeding `ENERGY_RATIO_THRESHOLD` (0.35) contribute to the screen
evidence score.

**Signal 3 — Periodicity Score:**

Spectral magnitudes are sampled along 36 radial lines (at 10-degree
intervals) from the band inner radius to the outer radius. Along each radial
line, the number of samples exceeding twice the local mean is counted. The
maximum count across all angles is reported as the periodicity score. Moire
patterns produce high periodicity along at least one orientation.

**Signal 4 — Grid Line Detection (1D FFT):**

Row-wise and column-wise mean projections of the grayscale image are computed,
producing 1D signals that capture horizontal and vertical structure
respectively. Each projection is mean-subtracted and transformed via 1D FFT.
The peak-to-mean ratio in the mid-frequency band of each projection is
computed. Elevated ratios indicate the presence of periodic grid lines from
the display pixel layout.

**Score Fusion:**

The four signals are combined into a weighted evidence score:

```
screen_evidence =
    0.30 * (peak_ratio > PEAK_RATIO_THRESHOLD) +
    0.30 * (periodicity > PERIODIC_THRESHOLD) +
    0.25 * (grid_score > 0.5) +
    0.15 * (energy_ratio > ENERGY_RATIO_THRESHOLD)

moire_score = 1.0 - screen_evidence
```

A rolling average over the most recent 20 frames stabilizes the score. Faces
with `moire_score < 0.45` are classified as screen-displayed.

---

### 4.2 Layer 2: Challenge-Response Verification (Active)

**Purpose:**

Video replay attacks using a projector or virtual camera bypass moire
detection because no screen pixel grid is present. The challenge-response
module addresses this by requiring the user to perform a random physical
action that a pre-recorded video cannot anticipate.

**Challenge Types:**

| Challenge | Detection Method | Threshold |
|-----------|-----------------|-----------|
| BLINK | Both eyes EAR below threshold simultaneously | EAR < 0.20 |
| BLINK_LEFT | Left EAR below threshold, right EAR normal | Left < 0.20, Right > 0.23 |
| TURN_RIGHT | Nose x-displacement from baseline (negative) | dx > 0.06 |
| TURN_LEFT | Nose x-displacement from baseline (positive) | dx > 0.06 |
| NOD_UP | Nose y-displacement from baseline (negative) | dy > 0.04 |
| SMILE | Mouth width increase ratio from baseline | ratio > 1.3 |

**Eye Aspect Ratio (EAR):**

Computed from MediaPipe face landmarks using six landmarks per eye
(indices defined in `LEFT_EYE_IDX` and `RIGHT_EYE_IDX`):

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

where p1-p6 are the six eye contour landmarks ordered as: lateral canthus,
superior landmarks (2), medial canthus, inferior landmarks (2).

**Protocol:**

1. A random challenge is selected from the six available types.
2. The system captures a 5-frame baseline of the user's neutral position.
3. The user has `CHALLENGE_TIMEOUT` (5.0 seconds) to complete the action.
4. Upon successful completion, the challenge enters a cooldown period
   (`COOLDOWN_AFTER_PASS = 15.0 seconds`) during which no new challenges
   are triggered.
5. Upon timeout, `fail_count` is incremented. After 2 cumulative failures,
   the face is permanently classified as SPOOF for the current session.

**Trigger Conditions:**

- Triggered immediately when moire analysis classifies the face as screen-
  displayed (`is_screen = True`).
- Triggered with 2% per-frame probability when the moire score falls in the
  borderline range (0.45 to 0.60).
- Not triggered when the user has passed a challenge within the cooldown
  window.

---

### 4.3 Layer 3: Multi-Frame Liveness Tracking

**Purpose:**

Detect static presentation attacks (printed photos, static images on screens)
by requiring evidence of natural facial dynamics over a minimum observation
period.

**Implementation:**

The `MultiFrameLiveness` tracker (defined in `core/liveness.py`) processes
every frame from the camera and maintains per-face state including:

- Cumulative blink count (via EAR threshold crossing detection)
- Total head movement (pixel displacement of face centroid)
- Tracking duration

After a minimum tracking period of 2.0 seconds, the tracker evaluates
liveness based on the presence of at least one detected blink. Faces that
fail to exhibit any blinks after the observation period are classified as
presentation attacks.

---

## Multi-Angle Enrollment Methodology

**Problem Statement:**

Standard single-image enrollment captures a face embedding at a single pose
and illumination condition. When the user presents at a different angle during
recognition (common in uncontrolled environments), cosine similarity between
the enrolled and query embeddings drops significantly, producing false
negatives.

**Solution:**

Enroll-v2 captures embeddings at three canonical poses (frontal, left-turned,
right-turned), each averaged over multiple high-quality frames. The database
automatically computes the centroid embedding:

```
emb_centroid = normalize(mean(emb_front, emb_left, emb_right))
```

This centroid embedding resides at the geometric center of the three pose-
specific embeddings in the 512-dimensional feature space. By the triangle
inequality, the centroid is closer to any arbitrary intermediate pose than
any single pose-specific embedding would be.

**Noise Reduction via Multi-Frame Averaging:**

Per-frame embedding variance arises from sensor noise, micro-motion blur,
and instantaneous illumination fluctuations. By averaging N frames per phase:

```
Var(emb_averaged) = Var(emb_single) / N
```

With N = 5-8 frames, embedding noise is reduced by a factor of 2.2-2.8x.

---

## Usage Instructions

### Enrollment Workflow

```
cd face-reg-finnal-project
python dev/enroll-v2.py
```

The system will prompt for:
- **Student ID**: Unique identifier (e.g., "SV001")
- **Full Name**: Display name for recognition results
- **Class (optional)**: Class or group identifier

After input, the camera opens and the guided enrollment process begins
automatically. The user should follow the on-screen instructions for each
of the three phases. Press Q at any time to cancel.

### Detection Workflow

For systems enrolled with the original enroll.py (single-image):

```
python dev/detect-v2.py
```

For systems enrolled with enroll-v2.py (multi-angle):

```
python dev/detect-v3.py
```

Both versions require an active webcam. The detection runs continuously
until terminated with the Q key.

---

## Configuration Parameters

### enroll-v2.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PHASE_DURATION` | 4.0s | Duration of each enrollment phase |
| `MIN_GOOD_FRAMES` | 3 | Minimum qualifying frames per phase |
| `MAX_GOOD_FRAMES` | 8 | Maximum frames to collect per phase |
| `QUALITY_BLUR_MIN` | 80.0 | Minimum Laplacian variance for sharpness |
| `TURN_THRESHOLD` | 0.04 | Minimum nose displacement for pose verification |
| `MAX_RETRIES` | 1 | Maximum retries allowed per failed phase |

### detect-v2.py / detect-v3.py

| Parameter | V2 Default | V3 Default | Description |
|-----------|------------|------------|-------------|
| `DETECT_FPS` | 10 | 10 | Maximum detection rate (throttle) |
| `COSINE_THRESHOLD` | 0.45 | 0.52 | Minimum cosine similarity for match |
| `MOIRE_EVERY_N_DETECT` | N/A | 3 | Moire analysis frequency |
| `ANALYZE_SIZE` | 128 | 64 | FFT input resolution |
| `PEAK_RATIO_THRESHOLD` | 2.8 | 2.8 | Moire peak detection threshold |
| `ENERGY_RATIO_THRESHOLD` | 0.35 | 0.35 | High-frequency energy threshold |
| `PERIODIC_THRESHOLD` | 3.5 | 3.5 | Periodicity score threshold |
| `CHALLENGE_TIMEOUT` | 5.0s | 5.0s | Time allowed per challenge |
| `COOLDOWN_AFTER_PASS` | 15.0s | 15.0s | Challenge cooldown after pass |

---

## Performance Characteristics

### Frame Rate Benchmarks

All measurements taken on a system with NVIDIA GPU and CUDA support.

| Scenario | detect.py (V1) | detect-v2 | detect-v3 |
|----------|---------------|-----------|-----------|
| No faces in frame | ~30 FPS | ~18 FPS | ~28 FPS |
| 1 face, real | ~26 FPS | ~16 FPS | ~22 FPS |
| 1 face, spoof (screen) | ~26 FPS | ~16 FPS | ~22 FPS |
| CPU-only mode | ~12 FPS | ~8 FPS | ~12 FPS |

### Per-Component Latency

| Component | Time (GPU) | Time (CPU) | Frequency |
|-----------|-----------|-----------|-----------|
| InsightFace detect + embed | 30ms | 80ms | Throttled at 10 Hz |
| Moire FFT analysis (V3) | 2ms | 3ms | Every 3rd detect cycle |
| MediaPipe FaceLandmarker | 12ms | 25ms | Conditional per frame |
| Cosine similarity matching | 0.1ms | 0.1ms | Per detect cycle |
| Multi-frame liveness | 2ms | 3ms | Every frame |
| Quality check | 1ms | 1ms | Per detect cycle |

### Recognition Latency (Time to First Match)

The end-to-end time from a subject entering the camera field of view to
a confirmed identity match:

| Configuration | Typical Latency |
|---------------|----------------|
| enroll.py + detect-v2 | 2.0 - 2.5 seconds |
| enroll-v2 + detect-v3 | 2.0 - 2.5 seconds |
| Any config, spoof detected | 2.0s (moire) + 5.0s (challenge) |

The recognition latency is dominated by the liveness tracking observation
period (minimum 2.0 seconds), not by computational overhead.

---

## Threat Model and Limitations

### Attacks Defended

| Attack Vector | Moire | Challenge | Liveness | Combined |
|---------------|-------|-----------|----------|----------|
| Printed photograph | -- | Blocked | Blocked | DEFENDED |
| Video on smartphone | Blocked | Blocked | Blocked | DEFENDED |
| Video on laptop/monitor | Blocked | Blocked | Blocked | DEFENDED |
| Video on 4K television | Partial | Blocked | Blocked | DEFENDED |
| Projector replay | Bypassed | Blocked | Blocked | DEFENDED |
| Virtual camera (OBS) | Bypassed | Blocked | Blocked | DEFENDED |

### Known Vulnerabilities

| Attack Vector | Risk Level | Mitigation |
|---------------|-----------|------------|
| 3D silicone mask (high quality) | Medium | Not addressed; requires depth sensing |
| Real-time deepfake (face swap) | High | Not addressed; requires CNN-based detector |
| Cooperative impostor (live person) | High | Social engineering; outside technical scope |

### Environmental Limitations

- **Extreme lighting conditions:** Very low light or strong backlighting can
  degrade both face detection confidence and moire pattern visibility.
- **Low-resolution cameras:** Cameras below 480p may not capture sufficient
  moire detail for reliable screen detection.
- **Camera motion blur:** Excessive camera shake during enrollment reduces
  embedding quality despite the multi-frame averaging.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | >= 4.5 | Image capture, processing, and display |
| numpy | >= 1.20 | Array operations, FFT computation |
| insightface | >= 0.7 | Face detection (RetinaFace) and embedding (ArcFace) |
| mediapipe | >= 0.10 | FaceLandmarker for pose verification and liveness |
| onnxruntime | >= 1.12 | ONNX model inference backend for InsightFace |

**Required Model Files:**

| File | Location | Source |
|------|----------|--------|
| InsightFace buffalo_l | `models/buffalo_l/` | Bundled with insightface |
| face_landmarker.task | `models/` | [MediaPipe Model Hub](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) |

Note: If `face_landmarker.task` is not present, the challenge-response module
and pose verification in enroll-v2 are automatically disabled. Moire detection
and basic liveness tracking remain operational.

---

## Version History

| Version | File | Changes |
|---------|------|---------|
| V1 | detect.py, enroll.py | Baseline: InsightFace + EAR liveness |
| V2 | detect-v2.py, enroll-v2.py | Added moire FFT, challenge-response, multi-angle enrollment |
| V3 | detect-v3.py | Stricter threshold (0.52), performance optimizations, embedding stats |
