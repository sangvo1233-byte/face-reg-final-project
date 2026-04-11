"""
dev/detect.py — Live Face Detection & Recognition (Enhanced)
==============================================================
Mở webcam, phát hiện khuôn mặt, vẽ bounding box + tên + accuracy,
hiển thị landmarks, gender/age, aligned face, quality info.

Cách dùng:
    cd face-reg-finnal-project
    python dev/detect.py

Controls:
    Q  — Quit
    S  — Screenshot (saved to dev/screenshots/)
    R  — Reload embeddings cache
    L  — Toggle landmarks display
    I  — Toggle info panel
"""

import sys
import os
import time
import cv2
import numpy as np

# ── Path setup ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.face_engine import get_engine
from core.liveness import get_liveness

# ── Config ──────────────────────────────────────────────────
CAM_INDEX   = 0
FRAME_W     = 1280
FRAME_H     = 720
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL  = cv2.FONT_HERSHEY_PLAIN
DETECT_FPS  = 10          # max face-detect calls per second (throttle)
SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
INFO_PANEL_W = 320        # width of the right-side info panel

# ── Colors (BGR) ────────────────────────────────────────────
C_PRESENT  = ( 80, 220,  80)   # green  — known face
C_UNKNOWN  = ( 40, 160, 255)   # amber  — face detected, no match
C_SPOOF    = ( 60,  60, 220)   # red    — spoof
C_WHITE    = (240, 240, 240)
C_BLACK    = ( 10,  10,  10)
C_GOLD     = ( 30, 200, 255)
C_DIM      = (120, 120, 120)
C_CYAN     = (220, 200,  40)   # cyan — landmarks

C_BLUE     = (220, 140,  40)   # blue — info
C_PANEL_BG = ( 30,  30,  35)   # dark panel background

# Landmark labels for 5-point
LANDMARK_LABELS = ["L-Eye", "R-Eye", "Nose", "L-Mouth", "R-Mouth"]


def draw_rounded_rect(img, pt1, pt2, color, thickness=2, r=12):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Clamp radius
    r = min(r, abs(x2 - x1) // 2, abs(y2 - y1) // 2)
    if r < 1:
        cv2.rectangle(img, pt1, pt2, color, thickness, cv2.LINE_AA)
        return
    cv2.line(img,  (x1+r, y1),   (x2-r, y1),   color, thickness, cv2.LINE_AA)
    cv2.line(img,  (x1+r, y2),   (x2-r, y2),   color, thickness, cv2.LINE_AA)
    cv2.line(img,  (x1,   y1+r), (x1,   y2-r), color, thickness, cv2.LINE_AA)
    cv2.line(img,  (x2,   y1+r), (x2,   y2-r), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1+r, y1+r), (r, r), 180,  0, 90,  color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2-r, y1+r), (r, r), 270,  0, 90,  color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1+r, y2-r), (r, r),  90,  0, 90,  color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2-r, y2-r), (r, r),   0,  0, 90,  color, thickness, cv2.LINE_AA)


def draw_label_box(img, text, sub_text, extra_text, x1, y1, color):
    """Draw a filled label box above the bounding box with up to 3 lines."""
    line_count = 1 + (1 if sub_text else 0) + (1 if extra_text else 0)
    label_h = 20 * line_count + 8
    lx1, ly1 = x1, max(0, y1 - label_h)
    lx2, ly2 = x1 + max(220, len(text) * 13 + 20), y1

    overlay = img.copy()
    cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), color, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    y_off = ly1 + 16
    cv2.putText(img, text, (lx1 + 8, y_off), FONT, 0.55, C_WHITE, 2, cv2.LINE_AA)
    if sub_text:
        y_off += 18
        cv2.putText(img, sub_text, (lx1 + 8, y_off), FONT, 0.44, C_WHITE, 1, cv2.LINE_AA)
    if extra_text:
        y_off += 18
        cv2.putText(img, extra_text, (lx1 + 8, y_off), FONT, 0.44, C_GOLD, 1, cv2.LINE_AA)


def draw_landmarks(img, landmarks, color=C_CYAN):
    """Draw 5-point face landmarks with labels."""
    if landmarks is None or len(landmarks) < 5:
        return
    for i, (x, y) in enumerate(landmarks[:5]):
        px, py = int(x), int(y)
        # Outer circle
        cv2.circle(img, (px, py), 5, color, 2, cv2.LINE_AA)
        # Inner dot
        cv2.circle(img, (px, py), 2, C_WHITE, -1, cv2.LINE_AA)
        # Label
        label = LANDMARK_LABELS[i] if i < len(LANDMARK_LABELS) else str(i)
        cv2.putText(img, label, (px + 7, py - 4), FONT, 0.33, color, 1, cv2.LINE_AA)


def draw_hud(img, fps, face_count, total_students, show_landmarks, show_info):
    """Top-left HUD: FPS, face count, DB size, toggle states."""
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), C_BLACK, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    items = [
        (f"FPS {fps:5.1f}", C_GOLD),
        (f"Faces: {face_count}", C_PRESENT if face_count > 0 else C_DIM),
        (f"DB: {total_students}", C_WHITE),
        (f"[L]andmarks:{'ON' if show_landmarks else 'OFF'}", C_CYAN if show_landmarks else C_DIM),
        (f"[I]nfo:{'ON' if show_info else 'OFF'}", C_BLUE if show_info else C_DIM),
        ("Q=Quit S=Shot R=Reload", C_DIM),
    ]
    x = 14
    for txt, color in items:
        cv2.putText(img, txt, (x, 28), FONT, 0.44, color, 1, cv2.LINE_AA)
        x += len(txt) * 8 + 16


def conf_bar(img, x1, y2, x2, confidence, color):
    """Draw a small confidence bar below the bounding box."""
    bar_w = x2 - x1
    bar_h = 6
    filled = int(bar_w * min(1.0, confidence))
    cv2.rectangle(img, (x1, y2 + 4), (x2, y2 + 4 + bar_h), C_BLACK, -1)
    cv2.rectangle(img, (x1, y2 + 4), (x1 + filled, y2 + 4 + bar_h), color, -1)


def draw_info_panel(img, results, panel_w=INFO_PANEL_W):
    """Draw a right-side info panel showing detailed recognition data per face."""
    h, w = img.shape[:2]
    # Create dark panel
    panel = np.full((h, panel_w, 3), C_PANEL_BG, dtype=np.uint8)

    # Title
    cv2.putText(panel, "FACE RECOGNITION", (12, 30), FONT, 0.6, C_GOLD, 2, cv2.LINE_AA)
    cv2.line(panel, (12, 40), (panel_w - 12, 40), C_DIM, 1, cv2.LINE_AA)

    y = 60
    if not results:
        cv2.putText(panel, "No faces detected", (12, y), FONT, 0.45, C_DIM, 1, cv2.LINE_AA)
    else:
        for i, r in enumerate(results):
            if y > h - 40:
                break

            color = C_PRESENT if r["status"] == "present" else (C_SPOOF if r["status"] == "spoof" else C_UNKNOWN)

            # Face index header
            cv2.putText(panel, f"--- Face #{i+1} ---", (12, y), FONT, 0.45, color, 1, cv2.LINE_AA)
            y += 22

            # Aligned face thumbnail
            if r.get("aligned") is not None:
                thumb = cv2.resize(r["aligned"], (64, 64))
                # Border
                cv2.rectangle(panel, (12, y), (78, y + 66), color, 2)
                panel[y+1:y+65, 13:77] = thumb
                # Text next to thumbnail
                tx = 88
                cv2.putText(panel, r["name"], (tx, y + 18), FONT, 0.5, C_WHITE, 1, cv2.LINE_AA)
                status_txt = r["status"].upper()
                cv2.putText(panel, status_txt, (tx, y + 36), FONT, 0.42, color, 1, cv2.LINE_AA)
                if r["sid"]:
                    cv2.putText(panel, f"ID: {r['sid']}", (tx, y + 54), FONT, 0.4, C_DIM, 1, cv2.LINE_AA)
                y += 72
            else:
                cv2.putText(panel, f"Name: {r['name']}", (12, y), FONT, 0.42, C_WHITE, 1, cv2.LINE_AA)
                y += 18

            # Confidence
            conf_pct = r["conf"] * 100
            conf_color = C_PRESENT if conf_pct >= 50 else C_UNKNOWN
            cv2.putText(panel, f"Confidence: {conf_pct:.1f}%", (12, y), FONT, 0.42, conf_color, 1, cv2.LINE_AA)
            # Mini bar
            bar_x = 180
            bar_w = panel_w - bar_x - 16
            filled = int(bar_w * min(1.0, r["conf"]))
            cv2.rectangle(panel, (bar_x, y - 10), (bar_x + bar_w, y), C_BLACK, -1)
            cv2.rectangle(panel, (bar_x, y - 10), (bar_x + filled, y), conf_color, -1)
            y += 20

            # Detection confidence
            if r.get("det_conf") is not None:
                cv2.putText(panel, f"Det Score: {r['det_conf']:.2f}", (12, y), FONT, 0.42, C_DIM, 1, cv2.LINE_AA)
                y += 18



            # Quality
            if r.get("quality"):
                q = r["quality"]
                cv2.putText(panel, f"Blur: {q['blur']:.0f}  Bright: {q['bright']:.0f}", (12, y), FONT, 0.38, C_DIM, 1, cv2.LINE_AA)
                y += 16
                cv2.putText(panel, f"Yaw: {q['yaw']:.1f} deg  Size: {q['size']}px", (12, y), FONT, 0.38, C_DIM, 1, cv2.LINE_AA)
                y += 16
                q_status = "PASS" if q["passed"] else "FAIL"
                q_color = C_PRESENT if q["passed"] else C_SPOOF
                cv2.putText(panel, f"Quality: {q_status}", (12, y), FONT, 0.42, q_color, 1, cv2.LINE_AA)
                y += 20

            # Liveness info (multi-frame)
            if r.get("liveness"):
                lv = r["liveness"]
                if lv["is_live"] is None:
                    lv_color = C_GOLD
                    lv_status = "CHECKING"
                elif lv["is_live"]:
                    lv_color = C_PRESENT
                    lv_status = "LIVE"
                else:
                    lv_color = C_SPOOF
                    lv_status = "SPOOF"
                cv2.putText(panel, f"Liveness: {lv_status}", (12, y), FONT, 0.42, lv_color, 1, cv2.LINE_AA)
                y += 18
                cv2.putText(panel, f"  Blinks: {lv.get('blinks', 0)}  EAR: {lv.get('ear', 0):.2f}", (12, y), FONT, 0.36, C_DIM, 1, cv2.LINE_AA)
                y += 16
                cv2.putText(panel, f"  Movement: {lv.get('movement', 0):.1f}px", (12, y), FONT, 0.36, C_DIM, 1, cv2.LINE_AA)
                y += 16
                cv2.putText(panel, f"  Track: {lv.get('track_time', 0):.1f}s", (12, y), FONT, 0.36, C_DIM, 1, cv2.LINE_AA)
                y += 16
                if lv_status == "SPOOF" and lv.get("reason"):
                    cv2.putText(panel, f"  Reason: {lv['reason']}", (12, y), FONT, 0.36, C_SPOOF, 1, cv2.LINE_AA)
                    y += 16

            # Embedding info
            if r.get("emb_dim"):
                cv2.putText(panel, f"Embedding: {r['emb_dim']}D", (12, y), FONT, 0.38, C_DIM, 1, cv2.LINE_AA)
                y += 18

            # Separator
            cv2.line(panel, (12, y), (panel_w - 12, y), (50, 50, 55), 1)
            y += 14

    # Combine
    combined = np.hstack([img, panel])
    return combined


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
    print("Controls: L=Toggle Landmarks | I=Toggle Info Panel | S=Screenshot | R=Reload | Q=Quit")

    # Multi-frame liveness tracker
    liveness_tracker = get_liveness()
    print("Liveness tracker ready (EAR blink + head movement)")

    # State
    last_results = []           # cached detection results
    last_raw_faces = []         # raw InsightFace objects for extra data
    last_detect_time = 0.0
    detect_interval  = 1.0 / DETECT_FPS

    # Toggle states
    show_landmarks = True
    show_info_panel = True

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

        # ── Liveness tracking (EVERY frame) ─────────────────
        liveness_tracker.process_frame(frame)

        # ── Face detection (throttled) ──────────────────────
        if now - last_detect_time >= detect_interval:
            last_detect_time = now
            try:
                # Use raw InsightFace app for full attributes (gender, age)
                engine._ensure_model()
                raw_faces = engine._app.get(frame)
                last_results = []
                last_raw_faces = raw_faces

                for raw_face in raw_faces:
                    det_score = float(raw_face.det_score)
                    if det_score < 0.5:
                        continue

                    x1, y1, x2, y2 = [int(v) for v in raw_face.bbox]
                    w_face, h_face = x2 - x1, y2 - y1
                    if w_face < 30 or h_face < 30:
                        continue

                    # Landmarks (5 key points)
                    kps = raw_face.kps if hasattr(raw_face, 'kps') else None


                    # Aligned face
                    aligned = engine._align_face(frame, raw_face)

                    # Embedding
                    emb = (raw_face.normed_embedding
                           if hasattr(raw_face, 'normed_embedding')
                           and raw_face.normed_embedding is not None
                           else np.array([]))
                    emb_dim = len(emb) if len(emb) > 0 else 0

                    # Quality check
                    quality_info = None
                    try:
                        from core.schemas import DetectedFace
                        det_face = DetectedFace(
                            bbox=np.array([x1, y1, x2, y2]),
                            landmarks=kps,
                            confidence=det_score,
                            aligned_face=aligned,
                            embedding=emb
                        )
                        qr = engine.quality_check(frame, det_face)
                        quality_info = {
                            "passed": qr.passed,
                            "blur": qr.blur_score,
                            "bright": qr.brightness,
                            "yaw": qr.yaw_angle,
                            "size": qr.face_size,
                            "reasons": qr.reasons,
                        }
                    except Exception:
                        pass

                    # ── Multi-frame Liveness check ──────────
                    lv = liveness_tracker.get_liveness((x1, y1, x2, y2))
                    liveness_info = {
                        "is_live": lv.is_live,
                        "score": lv.score,
                        "reason": lv.reason,
                        "blinks": lv.blinks,
                        "ear": lv.ear,
                        "movement": lv.movement,
                        "track_time": lv.track_time,
                    }

                    entry = {
                        "bbox":     (x1, y1, x2, y2),
                        "name":     "Unknown",
                        "sid":      "",
                        "conf":     0.0,
                        "det_conf": det_score,
                        "status":   "unknown",
                        "label":    "No match",
                        "kps":      kps,
                        "aligned":  aligned,
                        "emb_dim":  emb_dim,
                        "quality":  quality_info,
                        "liveness": liveness_info,
                    }

                    # Liveness judgment
                    if lv.is_live is None:
                        # Still analyzing
                        entry["status"] = "unknown"
                        entry["name"]   = "Checking..."
                        entry["label"]  = f"Blinks: {lv.blinks} | EAR: {lv.ear:.2f}"
                        entry["extra_label"] = f"{lv.reason}"
                    elif not lv.is_live:
                        # SPOOF detected
                        entry["status"] = "spoof"
                        entry["name"]   = "SPOOF"
                        entry["label"]  = f"Blinks: {lv.blinks} | Move: {lv.movement}px"
                        entry["extra_label"] = f"{lv.reason} | {lv.track_time:.0f}s"
                    else:
                        # LIVE — proceed with recognition
                        if emb is not None and len(emb) > 0:
                            match = engine.match(emb)
                            if match.matched:
                                entry["name"]   = match.name
                                entry["sid"]    = match.student_id
                                entry["conf"]   = float(match.score)
                                entry["status"] = "present"
                                entry["label"]  = f"ID: {match.student_id}  Conf: {match.score:.0%}"

                        entry["extra_label"] = f"LIVE | Blinks: {lv.blinks} | {lv.track_time:.0f}s"

                    last_results.append(entry)

            except Exception as e:
                print(f"Detect error: {e}")
                import traceback
                traceback.print_exc()

        # ── Draw cached results ─────────────────────────────
        for r in last_results:
            x1, y1, x2, y2 = r["bbox"]
            status = r["status"]
            color  = C_PRESENT if status == "present" else (C_SPOOF if status == "spoof" else C_UNKNOWN)

            # Bounding box
            draw_rounded_rect(frame, (x1, y1), (x2, y2), color, thickness=2)

            # Landmarks
            if show_landmarks:
                draw_landmarks(frame, r.get("kps"), color=C_CYAN)

            # Label (with gender/age line)
            draw_label_box(frame, r["name"], r["label"], r.get("extra_label"), x1, y1, color)

            # Confidence bar
            conf_bar(frame, x1, y2, x2, r["conf"], color)

            # Quality indicator dot
            if r.get("quality"):
                q_color = C_PRESENT if r["quality"]["passed"] else C_SPOOF
                cv2.circle(frame, (x2 - 8, y1 + 8), 6, q_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (x2 - 8, y1 + 8), 6, C_WHITE, 1, cv2.LINE_AA)

        # ── HUD ────────────────────────────────────────────
        total_students = len(db.get_all_students())
        draw_hud(frame, display_fps, len(last_results), total_students, show_landmarks, show_info_panel)

        # ── FPS calculation ─────────────────────────────────
        fps_counter += 1
        if now - fps_start >= 1.0:
            display_fps = fps_counter / (now - fps_start)
            fps_counter = 0
            fps_start   = now

        # ── Compose final frame ─────────────────────────────
        if show_info_panel:
            display_frame = draw_info_panel(frame, last_results)
        else:
            display_frame = frame

        cv2.imshow("Live Face Detection & Recognition", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit.")
            break
        elif key == ord('s'):
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOT_DIR, f"detect_{ts}.jpg")
            cv2.imwrite(path, display_frame)
            print(f"Screenshot saved: {path}")
        elif key == ord('r'):
            print("Reloading face embeddings cache...")
            engine.reload_cache()
            print("Cache reloaded.")
        elif key == ord('l'):
            show_landmarks = not show_landmarks
            print(f"Landmarks: {'ON' if show_landmarks else 'OFF'}")
        elif key == ord('i'):
            show_info_panel = not show_info_panel
            print(f"Info panel: {'ON' if show_info_panel else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
