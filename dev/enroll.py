"""
dev/enroll.py — Standalone Face Enrollment Tool
================================================
Mở webcam, đếm ngược 3 giây, chụp ảnh khuôn mặt và lưu vào hệ thống.

Cách dùng:
    cd face-reg-finnal-project
    python dev/enroll.py

Nhập ID và tên khi được hỏi, sau đó nhìn thẳng vào camera.
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
COUNTDOWN   = 3        # seconds before capture
CAM_INDEX   = 0        # webcam index
FRAME_W     = 1280
FRAME_H     = 720
FONT        = cv2.FONT_HERSHEY_SIMPLEX

# ── Colors (BGR) ────────────────────────────────────────────
C_GREEN  = (80,  220,  80)
C_BLUE   = (255, 160,  40)
C_RED    = (60,   60, 220)
C_WHITE  = (240, 240, 240)
C_BLACK  = (10,   10,  10)
C_GOLD   = (30,  200, 255)

def draw_overlay(frame, text_lines, box_color=C_BLUE, box_alpha=0.45):
    """Draw a semi-transparent bottom HUD with text lines."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), C_BLACK, -1)
    cv2.addWeighted(overlay, box_alpha, frame, 1 - box_alpha, 0, frame)
    y = h - 65
    for i, (txt, scale, color) in enumerate(text_lines):
        cv2.putText(frame, txt, (20, y + i * 28), FONT, scale, color, 2, cv2.LINE_AA)

def draw_face_guide(frame):
    """Draw an ellipse face guide in the center."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.ellipse(frame, (cx, cy), (130, 170), 0, 0, 360, C_BLUE, 2, cv2.LINE_AA)
    # Corner ticks
    for angle, (dx, dy) in [(0, (0, -170)), (90, (130, 0)), (180, (0, 170)), (270, (-130, 0))]:
        px, py = cx + dx, cy + dy
        cv2.circle(frame, (px, py), 5, C_GOLD, -1, cv2.LINE_AA)

def capture_best_frame(cap, engine, duration=3.0):
    """Countdown overlay, capture the sharpest frame with a detected face."""
    best_frame = None
    best_score = -1
    start = time.time()
    cancelled = False

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        # Work on a display copy — raw_frame stays clean for enrollment
        frame = raw_frame.copy()

        elapsed = time.time() - start
        remaining = max(0.0, duration - elapsed)
        draw_face_guide(frame)

        # Try detect face for live preview bounding box
        try:
            faces = engine.detect(raw_frame)          # detect on clean frame
            for face in faces:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), C_GREEN, 2, cv2.LINE_AA)
                # Sharpness on clean ROI
                roi = raw_frame[max(0,y1):y2, max(0,x1):x2]
                if roi.size > 0:
                    lap = cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    if lap > best_score:
                        best_score = lap
                        best_frame = raw_frame.copy()  # save CLEAN frame
        except Exception as e:
            print(f"[WARN] Face detect error: {e}")

        # Countdown number
        if remaining > 0:
            count_txt = str(int(remaining) + 1)
            cv2.putText(frame, count_txt, (frame.shape[1]//2 - 30, 80),
                        FONT, 2.5, C_GOLD, 4, cv2.LINE_AA)

        draw_overlay(frame, [
            (f"ENROLLING — Look straight at camera", 0.55, C_WHITE),
            (f"Capturing in {remaining:.1f}s ...", 0.55, C_GOLD),
        ])

        cv2.imshow("Face Enrollment — Press Q to cancel", frame)

        if elapsed >= duration:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Cancelled by user.")
            cancelled = True
            break

    return best_frame, cancelled

def flash_result(cap, success, message, duration=2.0):
    """Show a full-screen result flash."""
    end = time.time() + duration
    color = C_GREEN if success else C_RED
    icon  = "✓  " if success else "✗  "
    while time.time() < end:
        ret, frame = cap.read()
        if not ret:
            break
        # Tint overlay
        tint = frame.copy()
        cv2.rectangle(tint, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
        cv2.addWeighted(tint, 0.25, frame, 0.75, 0, frame)
        draw_overlay(frame, [
            (f"{icon}{message}", 0.7, color),
        ])
        cv2.imshow("Face Enrollment — Press Q to cancel", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    print("=" * 50)
    print("  Face Enrollment Tool")
    print("=" * 50)
    student_id = input("Student ID   : ").strip()
    name       = input("Full Name    : ").strip()
    class_name = input("Class (opt)  : ").strip()

    if not student_id or not name:
        print("ERROR: ID and Name are required.")
        sys.exit(1)

    print(f"\nLoading face engine...")
    engine = get_engine()
    print("Engine ready. Opening camera...")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera (index {CAM_INDEX})")
        sys.exit(1)

    print(f"\nGet ready! Capturing in {COUNTDOWN} seconds...")
    print("Press Q to cancel.\n")

    frame, cancelled = capture_best_frame(cap, engine, duration=COUNTDOWN)

    if cancelled:
        cap.release()
        cv2.destroyAllWindows()
        print("Enroll cancelled.")
        sys.exit(0)

    if frame is None:
        cap.release()
        cv2.destroyAllWindows()
        print("ERROR: No face detected during capture window. Check camera and lighting.")
        sys.exit(1)

    print("Enrolling face...")
    result = engine.enroll_from_photo(student_id, name, frame, class_name)

    success = result.get("success", False)
    msg     = result.get("message", str(result))
    print(f"\nResult: {msg}")

    flash_result(cap, success, msg, duration=2.5)

    cap.release()
    cv2.destroyAllWindows()

    if success:
        print(f"\n✓ Enrolled: {name} ({student_id}) — {result.get('embeddings_saved', '?')} embedding(s) saved.")
    else:
        print(f"\n✗ Failed: {msg}")

if __name__ == "__main__":
    main()
