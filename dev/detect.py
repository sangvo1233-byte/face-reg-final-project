"""
dev/detect.py — Live Face Detection & Recognition
==================================================
Mở webcam, phát hiện khuôn mặt, vẽ bounding box + tên + accuracy theo thời gian thực.

Cách dùng:
    cd face-reg-finnal-project
    python dev/detect.py

Controls:
    Q  — Quit
    S  — Screenshot (saved to dev/screenshots/)
    R  — Reload embeddings cache
"""

import sys
import os
import time
import cv2
import numpy as np

# ── Path setup ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.face_engine import get_engine

# ── Config ──────────────────────────────────────────────────
CAM_INDEX   = 0
FRAME_W     = 1280
FRAME_H     = 720
FONT        = cv2.FONT_HERSHEY_SIMPLEX
DETECT_FPS  = 10          # max face-detect calls per second (throttle)
SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "screenshots")

# ── Colors (BGR) ────────────────────────────────────────────
C_PRESENT  = ( 80, 220,  80)   # green  — known face
C_UNKNOWN  = ( 40, 160, 255)   # amber  — face detected, no match
C_SPOOF    = ( 60,  60, 220)   # red    — spoof
C_WHITE    = (240, 240, 240)
C_BLACK    = ( 10,  10,  10)
C_GOLD     = ( 30, 200, 255)
C_DIM      = (120, 120, 120)

def draw_rounded_rect(img, pt1, pt2, color, thickness=2, r=12):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img,  (x1+r, y1),   (x2-r, y1),   color, thickness, cv2.LINE_AA)
    cv2.line(img,  (x1+r, y2),   (x2-r, y2),   color, thickness, cv2.LINE_AA)
    cv2.line(img,  (x1,   y1+r), (x1,   y2-r), color, thickness, cv2.LINE_AA)
    cv2.line(img,  (x2,   y1+r), (x2,   y2-r), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1+r, y1+r), (r, r), 180,  0, 90,  color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2-r, y1+r), (r, r), 270,  0, 90,  color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1+r, y2-r), (r, r),  90,  0, 90,  color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2-r, y2-r), (r, r),   0,  0, 90,  color, thickness, cv2.LINE_AA)

def draw_label_box(img, text, sub_text, x1, y1, color):
    """Draw a filled label box above the bounding box."""
    label_h = 48 if sub_text else 26
    lx1, ly1 = x1, max(0, y1 - label_h)
    lx2, ly2 = x1 + max(200, len(text) * 13 + 20), y1

    overlay = img.copy()
    cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), color, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    cv2.putText(img, text,     (lx1 + 8, ly1 + 18), FONT, 0.58, C_WHITE, 2, cv2.LINE_AA)
    if sub_text:
        cv2.putText(img, sub_text, (lx1 + 8, ly1 + 38), FONT, 0.46, C_WHITE, 1, cv2.LINE_AA)

def draw_hud(img, fps, face_count, total_students):
    """Top-left HUD: FPS, face count, DB size."""
    h, w = img.shape[:2]
    # Dark strip
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), C_BLACK, -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    items = [
        (f"FPS {fps:5.1f}", C_GOLD),
        (f"Faces: {face_count}", C_GREEN if face_count > 0 else C_DIM),
        (f"DB: {total_students} students", C_WHITE),
        ("Q=Quit  S=Screenshot  R=Reload", C_DIM),
    ]
    x = 14
    for txt, color in items:
        cv2.putText(img, txt, (x, 28), FONT, 0.52, color, 1, cv2.LINE_AA)
        x += len(txt) * 9 + 20

def conf_bar(img, x1, y2, x2, confidence, color):
    """Draw a small confidence bar below the bounding box."""
    bar_w = x2 - x1
    bar_h = 6
    filled = int(bar_w * min(1.0, confidence))
    cv2.rectangle(img, (x1, y2 + 4), (x2, y2 + 4 + bar_h), C_BLACK, -1)
    cv2.rectangle(img, (x1, y2 + 4), (x1 + filled, y2 + 4 + bar_h), color, -1)

def main():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    print("Loading face engine...")
    engine = get_engine()

    from core.database import get_db
    db = get_db()

    print("Opening camera...")
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera (index {CAM_INDEX})")
        sys.exit(1)

    print("Live detection running. Press Q to quit.\n")

    # State
    last_results = []        # cached detection results
    last_detect_time = 0.0
    detect_interval  = 1.0 / DETECT_FPS

    # FPS tracking
    fps_counter = 0
    fps_start   = time.time()
    display_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed — retrying...")
            time.sleep(0.05)
            continue

        now = time.time()

        # ── Face detection (throttled) ──────────────────────
        if now - last_detect_time >= detect_interval:
            last_detect_time = now
            try:
                faces = engine.detect(frame)
                last_results = []
                for face in faces:
                    x1, y1, x2, y2 = [int(v) for v in face.bbox]
                    entry = {
                        "bbox":  (x1, y1, x2, y2),
                        "name":  "Unknown",
                        "sid":   "",
                        "conf":   0.0,
                        "status": "unknown",
                        "label":  "No match",
                    }
                    if face.embedding is not None and len(face.embedding) > 0:
                        match = engine.match(face.embedding)
                        if match.matched:
                            entry["name"]   = match.name
                            entry["sid"]    = match.student_id
                            entry["conf"]   = float(match.score)
                            entry["status"] = "present"
                            entry["label"]  = f"ID: {match.student_id}  Conf: {match.score:.0%}"
                    last_results.append(entry)
            except Exception as e:
                print(f"Detect error: {e}")

        # ── Draw cached results ─────────────────────────────
        for r in last_results:
            x1, y1, x2, y2 = r["bbox"]
            status = r["status"]
            color  = C_PRESENT if status == "present" else (C_SPOOF if status == "spoof" else C_UNKNOWN)

            # Bounding box
            draw_rounded_rect(frame, (x1, y1), (x2, y2), color, thickness=2)

            # Label
            draw_label_box(frame, r["name"], r["label"], x1, y1, color)

            # Confidence bar
            conf_bar(frame, x1, y2, x2, r["conf"], color)

        # ── HUD ────────────────────────────────────────────
        total_students = len(db.get_all_students())
        draw_hud(frame, display_fps, len(last_results), total_students)

        # ── FPS calculation ─────────────────────────────────
        fps_counter += 1
        if now - fps_start >= 1.0:
            display_fps = fps_counter / (now - fps_start)
            fps_counter = 0
            fps_start   = now

        cv2.imshow("Live Face Detection — Face Attendance", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit.")
            break
        elif key == ord('s'):
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOT_DIR, f"detect_{ts}.jpg")
            cv2.imwrite(path, frame)
            print(f"Screenshot saved: {path}")
        elif key == ord('r'):
            print("Reloading face embeddings cache...")
            engine.reload_cache()
            print("Cache reloaded.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
