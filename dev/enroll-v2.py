"""
dev/enroll-v2.py — Multi-Angle + Multi-Frame Face Enrollment (V2)
==================================================================
Đăng ký khuôn mặt với 3 góc (chính diện, trái, phải), mỗi góc lấy
nhiều frame tốt nhất rồi average → embedding cực kỳ robust.

Cách dùng:
    cd face-reg-finnal-project
    python dev/enroll-v2.py

Flow:
  Phase 1: Nhìn thẳng (4s) → average top frames → emb_FRONT
  Phase 2: Nghiêng trái (4s) → average top frames → emb_LEFT
  Phase 3: Nghiêng phải (4s) → average top frames → emb_RIGHT
  → Lưu 3 embeddings vào DB → DB tự tính mean khi matching
"""

import sys
import os
import time
import cv2
import numpy as np

# ── Path setup ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.face_engine import get_engine

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python as mp_python
import config

# ── Config ──────────────────────────────────────────────────
CAM_INDEX       = 0
FRAME_W         = 1280
FRAME_H         = 720
FONT            = cv2.FONT_HERSHEY_SIMPLEX

# Enrollment parameters
PHASE_DURATION  = 4.0      # seconds per phase
MIN_GOOD_FRAMES = 3        # minimum quality frames needed per phase
MAX_GOOD_FRAMES = 8        # max frames to average per phase
QUALITY_BLUR_MIN = 80.0    # minimum Laplacian variance (sharpness)

# Head pose thresholds (normalized nose_x displacement)
TURN_THRESHOLD  = 0.04     # nose must shift at least this much for left/right phases

# ── Colors (BGR) ────────────────────────────────────────────
C_GREEN  = ( 80, 220,  80)
C_BLUE   = (255, 160,  40)
C_RED    = ( 60,  60, 220)
C_WHITE  = (240, 240, 240)
C_BLACK  = ( 10,  10,  10)
C_GOLD   = ( 30, 200, 255)
C_CYAN   = (220, 200,  40)
C_ORANGE = ( 40, 140, 255)
C_DIM    = (120, 120, 120)

# ── Phase definitions ──────────────────────────────────────
PHASES = [
    {
        "name": "FRONT",
        "instruction": "NHIN THANG VAO CAMERA",
        "icon": "[O]",
        "verify": "center",      # nose should be near center
        "color": C_GREEN,
    },
    {
        "name": "LEFT",
        "instruction": "NGHIENG DAU SANG TRAI",
        "icon": "[<-]",
        "verify": "left",        # nose should shift left
        "color": C_CYAN,
    },
    {
        "name": "RIGHT",
        "instruction": "NGHIENG DAU SANG PHAI",
        "icon": "[->]",
        "verify": "right",       # nose should shift right
        "color": C_ORANGE,
    },
]


# ═══════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════

def draw_overlay(frame, text_lines, box_alpha=0.5):
    """Draw a semi-transparent bottom HUD with text lines."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 100), (w, h), C_BLACK, -1)
    cv2.addWeighted(overlay, box_alpha, frame, 1 - box_alpha, 0, frame)
    y = h - 78
    for txt, scale, color in text_lines:
        cv2.putText(frame, txt, (20, y), FONT, scale, color, 2, cv2.LINE_AA)
        y += 26


def draw_face_guide(frame, phase_info):
    """Draw face guide ellipse with phase-specific indicators."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    color = phase_info["color"]

    # Main ellipse
    cv2.ellipse(frame, (cx, cy), (130, 170), 0, 0, 360, color, 2, cv2.LINE_AA)

    # Direction arrow based on phase
    if phase_info["verify"] == "left":
        # Arrow pointing left
        cv2.arrowedLine(frame, (cx - 50, cy), (cx - 180, cy), color, 3, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(frame, "<<<", (cx - 200, cy - 20), FONT, 0.8, color, 2, cv2.LINE_AA)
    elif phase_info["verify"] == "right":
        # Arrow pointing right
        cv2.arrowedLine(frame, (cx + 50, cy), (cx + 180, cy), color, 3, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(frame, ">>>", (cx + 140, cy - 20), FONT, 0.8, color, 2, cv2.LINE_AA)
    else:
        # Center crosshair
        cv2.line(frame, (cx - 15, cy), (cx + 15, cy), color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 15), (cx, cy + 15), color, 2, cv2.LINE_AA)

    # Corner dots
    for dx, dy in [(0, -170), (130, 0), (0, 170), (-130, 0)]:
        cv2.circle(frame, (cx + dx, cy + dy), 5, C_GOLD, -1, cv2.LINE_AA)


def draw_progress_bar(frame, progress, y_pos, color, label=""):
    """Draw a horizontal progress bar."""
    h, w = frame.shape[:2]
    bar_x = 20
    bar_w = w - 40
    bar_h = 12

    # Background
    cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_w, y_pos + bar_h), C_DIM, -1)
    # Filled
    filled = int(bar_w * min(1.0, progress))
    cv2.rectangle(frame, (bar_x, y_pos), (bar_x + filled, y_pos + bar_h), color, -1)
    # Border
    cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_w, y_pos + bar_h), C_WHITE, 1)

    if label:
        cv2.putText(frame, label, (bar_x, y_pos - 5), FONT, 0.4, C_WHITE, 1, cv2.LINE_AA)


def draw_phase_header(frame, phase_idx, phase_info, remaining):
    """Draw the large phase instruction at the top of the frame."""
    h, w = frame.shape[:2]

    # Dark header bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), C_BLACK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    color = phase_info["color"]
    phase_label = f"Phase {phase_idx + 1}/3: {phase_info['instruction']}"

    # Centered text
    text_size = cv2.getTextSize(phase_label, FONT, 0.75, 2)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(frame, phase_label, (text_x, 40), FONT, 0.75, color, 2, cv2.LINE_AA)

    # Timer
    timer_txt = f"[{remaining:.1f}s]"
    timer_size = cv2.getTextSize(timer_txt, FONT, 0.55, 1)[0]
    timer_x = (w - timer_size[0]) // 2
    cv2.putText(frame, timer_txt, (timer_x, 68), FONT, 0.55, C_GOLD, 1, cv2.LINE_AA)

    # Phase progress bar
    progress = max(0, 1.0 - remaining / PHASE_DURATION)
    draw_progress_bar(frame, progress, 78, color)


def draw_quality_indicators(frame, good_count, min_needed, max_target, blur_val, pose_ok):
    """Draw quality feedback indicators."""
    h, w = frame.shape[:2]
    y = 110

    # Frame count
    count_color = C_GREEN if good_count >= min_needed else C_ORANGE
    cv2.putText(frame, f"Good frames: {good_count}/{max_target} (min {min_needed})",
                (20, y), FONT, 0.45, count_color, 1, cv2.LINE_AA)
    y += 22

    # Sharpness
    blur_color = C_GREEN if blur_val > QUALITY_BLUR_MIN else C_RED
    blur_label = "SHARP" if blur_val > QUALITY_BLUR_MIN else "BLURRY"
    cv2.putText(frame, f"Sharpness: {blur_val:.0f} ({blur_label})",
                (20, y), FONT, 0.42, blur_color, 1, cv2.LINE_AA)
    y += 20

    # Pose
    pose_color = C_GREEN if pose_ok else C_RED
    pose_label = "CORRECT" if pose_ok else "WRONG POSE"
    cv2.putText(frame, f"Head pose: {pose_label}",
                (20, y), FONT, 0.42, pose_color, 1, cv2.LINE_AA)


def draw_phase_summary(frame, phase_results):
    """Draw summary of completed phases at the top-right."""
    h, w = frame.shape[:2]
    x = w - 250
    y = 110

    cv2.putText(frame, "Completed:", (x, y), FONT, 0.42, C_WHITE, 1, cv2.LINE_AA)
    y += 22

    for pr in phase_results:
        name = pr["name"]
        count = pr["frame_count"]
        color = C_GREEN if count >= MIN_GOOD_FRAMES else C_RED
        icon = "OK" if count >= MIN_GOOD_FRAMES else "!!"
        cv2.putText(frame, f"  [{icon}] {name}: {count} frames",
                    (x, y), FONT, 0.38, color, 1, cv2.LINE_AA)
        y += 18


def flash_result(cap, success, message, duration=2.5):
    """Show a full-screen result flash."""
    end = time.time() + duration
    color = C_GREEN if success else C_RED
    icon = "[OK] " if success else "[!!] "
    while time.time() < end:
        ret, frame = cap.read()
        if not ret:
            break
        tint = frame.copy()
        cv2.rectangle(tint, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
        cv2.addWeighted(tint, 0.25, frame, 0.75, 0, frame)
        draw_overlay(frame, [(f"{icon}{message}", 0.65, color)])
        cv2.imshow("Face Enrollment V2", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# ═══════════════════════════════════════════════════════════
#  ENROLLMENT LOGIC
# ═══════════════════════════════════════════════════════════

def get_nose_x(face_landmarks, img_w):
    """Get normalized nose X position from MediaPipe landmarks."""
    if face_landmarks and len(face_landmarks) > 1:
        return face_landmarks[1].x  # landmark 1 = nose tip, normalized 0-1
    return 0.5  # default center


def verify_pose(nose_x, baseline_nose_x, phase_info):
    """Check if the user's head pose matches the phase requirement."""
    verify_type = phase_info["verify"]

    if verify_type == "center":
        # Nose should be roughly centered (not turned too much)
        return abs(nose_x - 0.5) < 0.12

    elif verify_type == "left":
        # Nose shifts to the RIGHT in image (face turns left from user's perspective)
        # But in webcam (mirrored), turning left means nose goes left
        dx = baseline_nose_x - nose_x
        return dx > TURN_THRESHOLD

    elif verify_type == "right":
        dx = nose_x - baseline_nose_x
        return dx > TURN_THRESHOLD

    return True


def run_phase(cap, engine, mesh_landmarker, phase_idx, phase_info, baseline_nose_x):
    """Run one enrollment phase: capture multi-frame embeddings at a specific pose.

    Returns:
        dict with name, embedding (averaged), frame_count, or None if cancelled/failed
    """
    print(f"\n[Phase {phase_idx+1}] {phase_info['instruction']}")

    collected_embeddings = []
    collected_qualities = []
    mesh_ts = phase_idx * 10000  # offset per phase to keep timestamps increasing

    start = time.time()

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        frame = raw_frame.copy()
        now = time.time()
        elapsed = now - start
        remaining = max(0.0, PHASE_DURATION - elapsed)

        # ── MediaPipe landmarks for pose verification ───
        nose_x = 0.5
        current_mesh_lm = None
        if mesh_landmarker is not None:
            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mesh_ts += 33
            try:
                mesh_result = mesh_landmarker.detect_for_video(mp_image, mesh_ts)
                if mesh_result.face_landmarks:
                    current_mesh_lm = mesh_result.face_landmarks[0]
                    nose_x = get_nose_x(current_mesh_lm, FRAME_W)
            except Exception:
                pass

        # ── Verify head pose matches phase ──────────────
        pose_ok = verify_pose(nose_x, baseline_nose_x, phase_info)

        # ── Face detection + quality + embedding ────────
        blur_val = 0.0
        try:
            faces = engine.detect(raw_frame)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                x1, y1, x2, y2 = face.bbox[:4]

                # Sharpness
                roi = raw_frame[max(0,y1):y2, max(0,x1):x2]
                if roi.size > 0:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    blur_val = cv2.Laplacian(gray_roi, cv2.CV_64F).var()

                # Draw bounding box
                bb_color = C_GREEN if (pose_ok and blur_val > QUALITY_BLUR_MIN) else C_RED
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              bb_color, 2, cv2.LINE_AA)

                # Collect embedding if quality AND pose are good
                if (pose_ok
                    and blur_val > QUALITY_BLUR_MIN
                    and face.embedding is not None
                    and len(face.embedding) > 0
                    and len(collected_embeddings) < MAX_GOOD_FRAMES):

                    collected_embeddings.append(face.embedding.copy())
                    collected_qualities.append(blur_val)

                    # Flash green border briefly
                    cv2.rectangle(frame, (int(x1)-2, int(y1)-2), (int(x2)+2, int(y2)+2),
                                  C_GREEN, 4, cv2.LINE_AA)

        except Exception as e:
            print(f"[WARN] Detect error: {e}")

        # ── Draw UI ─────────────────────────────────────
        draw_face_guide(frame, phase_info)
        draw_phase_header(frame, phase_idx, phase_info, remaining)
        draw_quality_indicators(frame, len(collected_embeddings), MIN_GOOD_FRAMES,
                                MAX_GOOD_FRAMES, blur_val, pose_ok)

        draw_overlay(frame, [
            (f"{phase_info['icon']} {phase_info['instruction']}", 0.5, phase_info['color']),
            (f"Frames: {len(collected_embeddings)}/{MAX_GOOD_FRAMES}  |  Blur: {blur_val:.0f}", 0.42, C_GOLD),
            ("Press Q to cancel", 0.35, C_DIM),
        ])

        cv2.imshow("Face Enrollment V2", frame)

        # ── Check exit conditions ───────────────────────
        if elapsed >= PHASE_DURATION:
            break
        if len(collected_embeddings) >= MAX_GOOD_FRAMES:
            break  # got enough frames early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None  # cancelled

    # ── Phase result ────────────────────────────────────
    if len(collected_embeddings) < MIN_GOOD_FRAMES:
        print(f"[Phase {phase_idx+1}] FAILED — only {len(collected_embeddings)} good frames "
              f"(need {MIN_GOOD_FRAMES})")
        return {
            "name": phase_info["name"],
            "embedding": None,
            "frame_count": len(collected_embeddings),
            "success": False,
        }

    # Average the collected embeddings
    emb_stack = np.stack(collected_embeddings)
    mean_emb = np.mean(emb_stack, axis=0)
    # Re-normalize to unit vector
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm

    avg_quality = np.mean(collected_qualities)

    print(f"[Phase {phase_idx+1}] OK — {len(collected_embeddings)} frames averaged, "
          f"quality={avg_quality:.0f}")

    return {
        "name": phase_info["name"],
        "embedding": mean_emb,
        "frame_count": len(collected_embeddings),
        "quality": float(avg_quality),
        "success": True,
    }


def calibrate_baseline(cap, engine, mesh_landmarker, mesh_ts_start=0):
    """Capture baseline nose position (looking straight) for pose verification."""
    print("Calibrating baseline pose (look straight)...")
    nose_positions = []
    mesh_ts = mesh_ts_start
    start = time.time()

    while time.time() - start < 1.5:  # 1.5s calibration
        ret, frame = cap.read()
        if not ret:
            continue

        if mesh_landmarker is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mesh_ts += 33
            try:
                result = mesh_landmarker.detect_for_video(mp_image, mesh_ts)
                if result.face_landmarks:
                    nose_x = result.face_landmarks[0][1].x
                    nose_positions.append(nose_x)
            except Exception:
                pass

        # Show calibration screen
        display = frame.copy()
        h, w = display.shape[:2]
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), C_BLACK, -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        cv2.putText(display, "Calibrating... Look straight at camera",
                    (w // 2 - 250, 40), FONT, 0.65, C_GOLD, 2, cv2.LINE_AA)
        cv2.imshow("Face Enrollment V2", display)
        cv2.waitKey(1)

    if nose_positions:
        baseline = float(np.median(nose_positions))
        print(f"Baseline nose_x = {baseline:.3f}")
        return baseline, mesh_ts
    return 0.5, mesh_ts


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Face Enrollment V2 — Multi-Angle + Multi-Frame")
    print("=" * 60)

    student_id = input("Student ID   : ").strip()
    name       = input("Full Name    : ").strip()
    class_name = input("Class (opt)  : ").strip()

    if not student_id or not name:
        print("ERROR: ID and Name are required.")
        sys.exit(1)

    print(f"\nLoading face engine...")
    engine = get_engine()

    # ── MediaPipe FaceLandmarker for pose verification ──
    face_landmarker_model = str(config.MODELS_DIR / "face_landmarker.task")
    mesh_landmarker = None

    if os.path.exists(face_landmarker_model):
        base_options = mp_python.BaseOptions(model_asset_path=face_landmarker_model)
        mesh_options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        mesh_landmarker = vision.FaceLandmarker.create_from_options(mesh_options)
        print("FaceLandmarker loaded (pose verification ready)")
    else:
        print(f"[WARN] FaceLandmarker not found — pose verification disabled")
        print(f"[WARN] All phases will collect frames without pose check")

    print("\nOpening camera...")
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera (index {CAM_INDEX})")
        sys.exit(1)

    # ── Calibrate baseline ──────────────────────────────
    baseline_nose_x, mesh_ts = calibrate_baseline(cap, engine, mesh_landmarker)

    # ── Run 3 phases ────────────────────────────────────
    phase_results = []
    all_success = True

    for i, phase_info in enumerate(PHASES):
        # Brief transition screen
        trans_end = time.time() + 1.5
        while time.time() < trans_end:
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), C_BLACK, -1)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

                txt = f"Phase {i+1}/3: {phase_info['instruction']}"
                text_size = cv2.getTextSize(txt, FONT, 0.9, 2)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(frame, txt, (text_x, h // 2 - 20),
                            FONT, 0.9, phase_info["color"], 2, cv2.LINE_AA)
                cv2.putText(frame, "Get ready...", (w // 2 - 80, h // 2 + 30),
                            FONT, 0.6, C_GOLD, 1, cv2.LINE_AA)

                # Show completed phases
                draw_phase_summary(frame, phase_results)
                cv2.imshow("Face Enrollment V2", frame)
            cv2.waitKey(1)

        result = run_phase(cap, engine, mesh_landmarker, i, phase_info, baseline_nose_x)

        if result is None:
            # Cancelled
            cap.release()
            cv2.destroyAllWindows()
            print("\nEnrollment cancelled by user.")
            sys.exit(0)

        phase_results.append(result)
        if not result["success"]:
            all_success = False

    # ── Evaluate results ────────────────────────────────
    successful_phases = [r for r in phase_results if r["success"] and r["embedding"] is not None]

    if len(successful_phases) == 0:
        flash_result(cap, False, "FAILED — No valid embeddings captured", 3.0)
        cap.release()
        cv2.destroyAllWindows()
        print("\nEnrollment FAILED — no phases succeeded.")
        sys.exit(1)

    if len(successful_phases) < 3:
        print(f"\n[WARN] Only {len(successful_phases)}/3 phases succeeded.")
        print(f"[WARN] Enrollment will proceed but accuracy may be reduced.")

    # ── Save to database ────────────────────────────────
    from core.database import get_db
    db = get_db()

    # Add/update student
    db.add_student(student_id, name, class_name)

    # Delete old embeddings
    old_count = db.get_embedding_count(student_id)
    if old_count > 0:
        db.delete_embeddings(student_id)
        print(f"Deleted {old_count} old embedding(s)")

    # Save new embeddings (one per successful phase)
    saved = 0
    for pr in successful_phases:
        quality = pr.get("quality", 0.0)
        source = f"v2_{pr['name'].lower()}"
        db.save_embedding(student_id, pr["embedding"], quality, source)
        saved += 1
        print(f"  Saved: {source} ({pr['frame_count']} frames avg, quality={quality:.0f})")

    # Save best aligned face as photo
    try:
        engine._ensure_model()
        ret, frame = cap.read()
        if ret:
            face = engine.detect_largest(frame)
            if face is not None:
                from datetime import datetime
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                photo_path = str(config.FACE_CROPS_DIR / f"{student_id}_{ts}.jpg")
                cv2.imwrite(photo_path, face.aligned_face)
                db.update_student(student_id, photo_path=photo_path)
    except Exception:
        pass

    # Reload engine cache
    engine.reload_cache()

    # ── Show result ─────────────────────────────────────
    total_frames = sum(r["frame_count"] for r in successful_phases)
    msg = f"Enrolled: {name} — {saved} angles, {total_frames} frames"
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")

    flash_result(cap, True, msg, 3.0)

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nEnrollment V2 complete:")
    for pr in phase_results:
        status = "OK" if pr["success"] else "FAILED"
        print(f"  {pr['name']:>8}: {status} ({pr['frame_count']} frames)")
    print(f"  Total embeddings saved: {saved}")
    print(f"  DB will average {saved} embeddings for matching.")


if __name__ == "__main__":
    main()
