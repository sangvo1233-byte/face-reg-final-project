"""
dev/enroll-v2.py — Multi-Angle + Multi-Frame Face Enrollment (V2)
==================================================================
Đăng ký khuôn mặt với 3 góc (chính diện, trái, phải), mỗi góc lấy
nhiều frame tốt nhất rồi average → embedding cực kỳ robust.

Cách dùng:
    cd face-reg-finnal-project
    python dev/enroll-v2.py

Flow:
  Welcome screen → Calibrate → Phase 1 (Front) → Phase 2 (Left) →
  Phase 3 (Right) → Final Summary → Save to DB

Controls:
    Q  — Cancel enrollment
    R  — Retry current phase (if failed)
    SPACE — Skip to next phase
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
FONT            = cv2.FONT_HERSHEY_DUPLEX
FONT_S          = cv2.FONT_HERSHEY_SIMPLEX
WIN_NAME        = "Face Enrollment V2"

# Enrollment parameters
PHASE_DURATION  = 4.0      # seconds per phase
MIN_GOOD_FRAMES = 3        # minimum quality frames needed per phase
MAX_GOOD_FRAMES = 8        # max frames to average per phase
QUALITY_BLUR_MIN = 80.0    # minimum Laplacian variance (sharpness)

# Head pose thresholds (normalized nose_x displacement)
TURN_THRESHOLD  = 0.04     # nose must shift at least this much for left/right

# ── Colors (BGR) ────────────────────────────────────────────
C_GREEN  = ( 80, 220,  80)
C_BLUE   = (255, 160,  40)
C_RED    = ( 60,  60, 220)
C_WHITE  = (245, 245, 245)
C_BLACK  = ( 10,  10,  10)
C_GOLD   = ( 30, 200, 255)
C_CYAN   = (220, 200,  40)
C_ORANGE = ( 40, 140, 255)
C_DIM    = (170, 170, 175)
C_PANEL  = ( 28,  28,  32)
C_PULSE_GREEN = ( 60, 255, 60)

# ── Phase definitions ──────────────────────────────────────
PHASES = [
    {
        "name": "FRONT",
        "instruction": "NHIN THANG VAO CAMERA",
        "sub": "Giu mat that thang",
        "icon": "[O]",
        "verify": "center",
        "color": C_GREEN,
    },
    {
        "name": "LEFT",
        "instruction": "NGHIENG DAU SANG TRAI",
        "sub": "Xoay mat sang ben trai",
        "icon": "[<-]",
        "verify": "left",
        "color": C_CYAN,
    },
    {
        "name": "RIGHT",
        "instruction": "NGHIENG DAU SANG PHAI",
        "sub": "Xoay mat sang ben phai",
        "icon": "[->]",
        "verify": "right",
        "color": C_ORANGE,
    },
]


# ═══════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════

def draw_rounded_rect(img, pt1, pt2, color, thickness=2, r=10, fill=False):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(r, abs(x2-x1)//2, abs(y2-y1)//2)
    if r < 1:
        if fill:
            cv2.rectangle(img, pt1, pt2, color, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, pt1, pt2, color, thickness, cv2.LINE_AA)
        return
    if fill:
        # Fill the interior
        cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
        cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
        cv2.circle(img, (x1+r, y1+r), r, color, -1)
        cv2.circle(img, (x2-r, y1+r), r, color, -1)
        cv2.circle(img, (x1+r, y2-r), r, color, -1)
        cv2.circle(img, (x2-r, y2-r), r, color, -1)
    else:
        cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1+r, y1+r), (r,r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2-r, y1+r), (r,r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1+r, y2-r), (r,r),  90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2-r, y2-r), (r,r),   0, 0, 90, color, thickness, cv2.LINE_AA)


def draw_dark_overlay(frame, alpha=0.5):
    """Apply a dark overlay to the entire frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), C_BLACK, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_bottom_hud(frame, text_lines, box_alpha=0.55):
    """Draw a semi-transparent bottom HUD with text lines."""
    h, w = frame.shape[:2]
    hud_h = 26 * len(text_lines) + 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - hud_h), (w, h), C_BLACK, -1)
    cv2.addWeighted(overlay, box_alpha, frame, 1 - box_alpha, 0, frame)
    y = h - hud_h + 22
    for txt, scale, color in text_lines:
        cv2.putText(frame, txt, (20, y), FONT, scale, color, 2, cv2.LINE_AA)
        y += 26


def draw_face_guide(frame, phase_info):
    """Draw face guide ellipse with phase-specific indicators."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    color = phase_info["color"]

    # Main translucent ellipse
    overlay = frame.copy()
    cv2.ellipse(overlay, (cx, cy), (140, 180), 0, 0, 360, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.ellipse(frame, (cx, cy), (140, 180), 0, 0, 360, color, 2, cv2.LINE_AA)

    # Direction indicators
    if phase_info["verify"] == "left":
        cv2.arrowedLine(frame, (cx - 60, cy), (cx - 200, cy), color, 3, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(frame, "<<<", (cx - 220, cy - 25), FONT, 0.9, color, 2, cv2.LINE_AA)
        cv2.putText(frame, phase_info["sub"], (cx - 220, cy + 45), FONT, 0.45, color, 1, cv2.LINE_AA)
    elif phase_info["verify"] == "right":
        cv2.arrowedLine(frame, (cx + 60, cy), (cx + 200, cy), color, 3, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(frame, ">>>", (cx + 160, cy - 25), FONT, 0.9, color, 2, cv2.LINE_AA)
        cv2.putText(frame, phase_info["sub"], (cx + 100, cy + 45), FONT, 0.45, color, 1, cv2.LINE_AA)
    else:
        # Center crosshair
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), color, 2, cv2.LINE_AA)

    # Corner dots on ellipse
    for dx, dy in [(0, -180), (140, 0), (0, 180), (-140, 0)]:
        cv2.circle(frame, (cx + dx, cy + dy), 5, C_GOLD, -1, cv2.LINE_AA)


def draw_phase_dots(frame, current_phase, total_phases, phase_results):
    """Draw overall phase progress dots: ● ○ ○ always visible."""
    h, w = frame.shape[:2]
    dot_r = 12
    gap = 40
    total_w = total_phases * (dot_r * 2 + gap) - gap
    start_x = w - total_w - 30
    y = 30

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (start_x - 15, y - 20), (w - 10, y + 25), C_BLACK, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i in range(total_phases):
        cx = start_x + i * (dot_r * 2 + gap) + dot_r
        phase_color = PHASES[i]["color"]

        if i < len(phase_results):
            # Completed
            if phase_results[i]["success"]:
                cv2.circle(frame, (cx, y), dot_r, phase_color, -1, cv2.LINE_AA)
                cv2.putText(frame, "OK", (cx - 8, y + 4), FONT, 0.3, C_BLACK, 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (cx, y), dot_r, C_RED, 2, cv2.LINE_AA)
                cv2.putText(frame, "!!", (cx - 7, y + 4), FONT, 0.3, C_RED, 1, cv2.LINE_AA)
        elif i == current_phase:
            # Current — pulsing
            pulse = int(3 * abs(np.sin(time.time() * 4)))
            cv2.circle(frame, (cx, y), dot_r + pulse, phase_color, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, y), dot_r - 3, phase_color, -1, cv2.LINE_AA)
        else:
            # Future
            cv2.circle(frame, (cx, y), dot_r, C_DIM, 2, cv2.LINE_AA)

        # Phase label
        cv2.putText(frame, PHASES[i]["name"][:1], (cx - 4, y + 28), FONT, 0.3, C_WHITE, 1, cv2.LINE_AA)


def draw_progress_bar(frame, progress, y_pos, color, label=""):
    """Draw a horizontal progress bar."""
    h, w = frame.shape[:2]
    bar_x = 20
    bar_w = w - 40
    bar_h = 10

    cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_w, y_pos + bar_h), C_DIM, -1)
    filled = int(bar_w * min(1.0, progress))
    if filled > 0:
        cv2.rectangle(frame, (bar_x, y_pos), (bar_x + filled, y_pos + bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_w, y_pos + bar_h), C_WHITE, 1)

    if label:
        cv2.putText(frame, label, (bar_x, y_pos - 5), FONT, 0.35, C_WHITE, 1, cv2.LINE_AA)


def draw_phase_header(frame, phase_idx, phase_info, remaining):
    """Draw the phase instruction header at top."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), C_BLACK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    color = phase_info["color"]
    label = f"Phase {phase_idx + 1}/3: {phase_info['instruction']}"

    text_size = cv2.getTextSize(label, FONT, 0.72, 2)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(frame, label, (text_x, 35), FONT, 0.72, color, 2, cv2.LINE_AA)

    # Timer
    timer_txt = f"[{remaining:.1f}s]"
    timer_size = cv2.getTextSize(timer_txt, FONT, 0.5, 1)[0]
    timer_x = (w - timer_size[0]) // 2
    timer_color = C_GOLD if remaining > 1.0 else C_RED
    cv2.putText(frame, timer_txt, (timer_x, 58), FONT, 0.5, timer_color, 1, cv2.LINE_AA)

    # Countdown number (large, semi-transparent)
    count_num = str(int(remaining) + 1)
    num_size = cv2.getTextSize(count_num, FONT, 3.0, 5)[0]
    num_x = (w - num_size[0]) // 2
    num_y = h // 2 + num_size[1] // 2
    overlay2 = frame.copy()
    cv2.putText(overlay2, count_num, (num_x, num_y), FONT, 3.0, color, 5, cv2.LINE_AA)
    cv2.addWeighted(overlay2, 0.2, frame, 0.8, 0, frame)

    # Progress bar
    progress = max(0, 1.0 - remaining / PHASE_DURATION)
    draw_progress_bar(frame, progress, 70, color)


def draw_quality_panel(frame, good_count, min_needed, max_target, blur_val, pose_ok):
    """Draw quality feedback panel on the left side."""
    h, w = frame.shape[:2]
    px, py = 15, 100
    pw = 260

    # Panel background
    overlay = frame.copy()
    draw_rounded_rect(overlay, (px, py), (px + pw, py + 110), C_PANEL, fill=True)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    draw_rounded_rect(frame, (px, py), (px + pw, py + 110), C_DIM, thickness=1)

    # Title
    cv2.putText(frame, "QUALITY CHECK", (px + 8, py + 18), FONT, 0.4, C_GOLD, 1, cv2.LINE_AA)
    cv2.line(frame, (px + 8, py + 24), (px + pw - 8, py + 24), C_DIM, 1)

    y = py + 42

    # Frame count with mini bar
    count_color = C_GREEN if good_count >= min_needed else C_ORANGE
    cv2.putText(frame, f"Frames: {good_count}/{max_target}", (px + 8, y), FONT, 0.4, count_color, 1, cv2.LINE_AA)
    bar_x = px + 140
    bar_w = pw - 150
    bar_filled = int(bar_w * min(1.0, good_count / max_target))
    cv2.rectangle(frame, (bar_x, y - 10), (bar_x + bar_w, y), C_BLACK, -1)
    cv2.rectangle(frame, (bar_x, y - 10), (bar_x + bar_filled, y), count_color, -1)
    if good_count >= min_needed:
        cv2.putText(frame, "OK", (bar_x + bar_w + 4, y), FONT, 0.3, C_GREEN, 1, cv2.LINE_AA)
    y += 22

    # Sharpness
    blur_color = C_GREEN if blur_val > QUALITY_BLUR_MIN else C_RED
    blur_label = "SHARP" if blur_val > QUALITY_BLUR_MIN else "BLURRY"
    cv2.circle(frame, (px + 14, y - 4), 4, blur_color, -1, cv2.LINE_AA)
    cv2.putText(frame, f"Sharpness: {blur_val:.0f} ({blur_label})", (px + 24, y), FONT, 0.38, blur_color, 1, cv2.LINE_AA)
    y += 22

    # Head pose
    pose_color = C_GREEN if pose_ok else C_RED
    pose_label = "CORRECT" if pose_ok else "WRONG POSE"
    cv2.circle(frame, (px + 14, y - 4), 4, pose_color, -1, cv2.LINE_AA)
    cv2.putText(frame, f"Head Pose: {pose_label}", (px + 24, y), FONT, 0.38, pose_color, 1, cv2.LINE_AA)


def draw_face_thumbnail(frame, aligned_face, label="", color=C_GREEN):
    """Draw aligned face thumbnail in the bottom-right corner."""
    if aligned_face is None:
        return
    h, w = frame.shape[:2]
    thumb_size = 96
    thumb = cv2.resize(aligned_face, (thumb_size, thumb_size))

    tx = w - thumb_size - 20
    ty = h - thumb_size - 120

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (tx - 4, ty - 22), (tx + thumb_size + 4, ty + thumb_size + 4), C_PANEL, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Label
    cv2.putText(frame, label if label else "LIVE", (tx, ty - 6), FONT, 0.35, C_GOLD, 1, cv2.LINE_AA)

    # Thumbnail
    frame[ty:ty+thumb_size, tx:tx+thumb_size] = thumb
    cv2.rectangle(frame, (tx - 2, ty - 2), (tx + thumb_size + 2, ty + thumb_size + 2), color, 2, cv2.LINE_AA)


def draw_capture_pulse(frame, intensity):
    """Draw a green border pulse when a good frame is captured.
    intensity: 1.0 = full pulse, fades to 0.0.
    """
    if intensity <= 0.01:
        return
    h, w = frame.shape[:2]
    thickness = max(2, int(8 * intensity))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), C_PULSE_GREEN, thickness)
    cv2.addWeighted(overlay, intensity * 0.7, frame, 1.0, 0, frame)

    # Flash text
    if intensity > 0.5:
        txt = "CAPTURED!"
        txt_size = cv2.getTextSize(txt, FONT, 0.6, 2)[0]
        txt_x = w - txt_size[0] - 30
        cv2.putText(frame, txt, (txt_x, 68), FONT, 0.6, C_PULSE_GREEN, 2, cv2.LINE_AA)


def draw_student_info(frame, student_id, name, class_name):
    """Draw student info badge at top-left."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    draw_rounded_rect(overlay, (10, 48), (320, 100), C_PANEL, fill=True)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    draw_rounded_rect(frame, (10, 48), (320, 100), C_BLUE, thickness=1)

    cv2.putText(frame, f"ID: {student_id}", (20, 68), FONT, 0.4, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{name}", (20, 88), FONT, 0.45, C_GOLD, 1, cv2.LINE_AA)
    if class_name:
        cv2.putText(frame, f"Class: {class_name}", (180, 68), FONT, 0.35, C_DIM, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════
#  WELCOME & SUMMARY SCREENS
# ═══════════════════════════════════════════════════════════

def show_welcome_screen(cap, student_id, name, class_name, duration=3.0):
    """Show welcome screen with student info and countdown before starting."""
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        elapsed = time.time() - start
        remaining = max(0, duration - elapsed)

        draw_dark_overlay(frame, 0.4)
        h, w = frame.shape[:2]

        # Title
        title = "FACE ENROLLMENT V2"
        t_size = cv2.getTextSize(title, FONT, 1.0, 2)[0]
        cv2.putText(frame, title, ((w - t_size[0]) // 2, h // 2 - 100),
                    FONT, 1.0, C_GOLD, 2, cv2.LINE_AA)

        subtitle = "Multi-Angle + Multi-Frame"
        s_size = cv2.getTextSize(subtitle, FONT, 0.55, 1)[0]
        cv2.putText(frame, subtitle, ((w - s_size[0]) // 2, h // 2 - 70),
                    FONT, 0.55, C_DIM, 1, cv2.LINE_AA)

        # Student info card
        card_w, card_h = 400, 120
        card_x = (w - card_w) // 2
        card_y = h // 2 - 30
        overlay = frame.copy()
        draw_rounded_rect(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), C_PANEL, fill=True)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        draw_rounded_rect(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), C_BLUE, thickness=2)

        cv2.putText(frame, f"Student ID:  {student_id}", (card_x + 20, card_y + 30), FONT, 0.5, C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Name:        {name}", (card_x + 20, card_y + 58), FONT, 0.5, C_GOLD, 1, cv2.LINE_AA)
        if class_name:
            cv2.putText(frame, f"Class:       {class_name}", (card_x + 20, card_y + 86), FONT, 0.5, C_DIM, 1, cv2.LINE_AA)

        # Phase preview
        y_phases = card_y + card_h + 30
        phase_labels = ["1. Front", "2. Left", "3. Right"]
        total_label_w = sum(cv2.getTextSize(l, FONT, 0.45, 1)[0][0] + 40 for l in phase_labels)
        px = (w - total_label_w) // 2
        for i, pl in enumerate(phase_labels):
            color = PHASES[i]["color"]
            cv2.circle(frame, (px + 8, y_phases), 6, color, -1, cv2.LINE_AA)
            cv2.putText(frame, pl, (px + 20, y_phases + 5), FONT, 0.45, color, 1, cv2.LINE_AA)
            px += cv2.getTextSize(pl, FONT, 0.45, 1)[0][0] + 50

        # Countdown
        count_txt = f"Starting in {remaining:.0f}s..."
        c_size = cv2.getTextSize(count_txt, FONT, 0.6, 1)[0]
        cv2.putText(frame, count_txt, ((w - c_size[0]) // 2, h - 60),
                    FONT, 0.6, C_GOLD, 1, cv2.LINE_AA)

        cv2.putText(frame, "Press Q to cancel", ((w - 180) // 2, h - 30),
                    FONT, 0.4, C_DIM, 1, cv2.LINE_AA)

        cv2.imshow(WIN_NAME, frame)

        if elapsed >= duration:
            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
    return True


def show_final_summary(cap, phase_results, student_id, name, saved_count, duration=4.0):
    """Show final enrollment summary with thumbnails from each phase."""
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        elapsed = time.time() - start
        draw_dark_overlay(frame, 0.5)
        h, w = frame.shape[:2]

        success = saved_count >= 1

        # Title
        if success:
            title = "ENROLLMENT SUCCESSFUL"
            title_color = C_GREEN
        else:
            title = "ENROLLMENT FAILED"
            title_color = C_RED

        t_size = cv2.getTextSize(title, FONT, 0.9, 2)[0]
        cv2.putText(frame, title, ((w - t_size[0]) // 2, 60), FONT, 0.9, title_color, 2, cv2.LINE_AA)

        # Student info
        info_txt = f"{name}  (ID: {student_id})"
        i_size = cv2.getTextSize(info_txt, FONT, 0.55, 1)[0]
        cv2.putText(frame, info_txt, ((w - i_size[0]) // 2, 90), FONT, 0.55, C_GOLD, 1, cv2.LINE_AA)

        # Phase result cards
        card_w = 200
        card_h = 180
        gap = 30
        total_cards_w = 3 * card_w + 2 * gap
        start_x = (w - total_cards_w) // 2
        card_y = 120

        for i, pr in enumerate(phase_results):
            cx = start_x + i * (card_w + gap)
            phase_color = PHASES[i]["color"]

            # Card background
            overlay = frame.copy()
            draw_rounded_rect(overlay, (cx, card_y), (cx + card_w, card_y + card_h), C_PANEL, fill=True)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

            border_color = C_GREEN if pr["success"] else C_RED
            draw_rounded_rect(frame, (cx, card_y), (cx + card_w, card_y + card_h), border_color, thickness=2)

            # Phase name
            cv2.putText(frame, f"Phase {i+1}: {PHASES[i]['name']}", (cx + 10, card_y + 22),
                        FONT, 0.42, phase_color, 1, cv2.LINE_AA)

            # Thumbnail (if we have an aligned face stored)
            if pr.get("best_aligned") is not None:
                thumb = cv2.resize(pr["best_aligned"], (80, 80))
                tx = cx + (card_w - 80) // 2
                ty = card_y + 32
                frame[ty:ty+80, tx:tx+80] = thumb
                cv2.rectangle(frame, (tx-1, ty-1), (tx+81, ty+81), border_color, 1, cv2.LINE_AA)
            else:
                no_face_txt = "NO FACE"
                nf_size = cv2.getTextSize(no_face_txt, FONT, 0.4, 1)[0]
                nf_x = cx + (card_w - nf_size[0]) // 2
                cv2.putText(frame, no_face_txt, (nf_x, card_y + 80), FONT, 0.4, C_RED, 1, cv2.LINE_AA)

            # Stats
            status = "OK" if pr["success"] else "FAILED"
            s_color = C_GREEN if pr["success"] else C_RED
            cv2.putText(frame, f"Status: {status}", (cx + 10, card_y + 130), FONT, 0.38, s_color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Frames: {pr['frame_count']}", (cx + 10, card_y + 150), FONT, 0.35, C_DIM, 1, cv2.LINE_AA)
            if pr.get("quality"):
                cv2.putText(frame, f"Quality: {pr['quality']:.0f}", (cx + 10, card_y + 168), FONT, 0.35, C_DIM, 1, cv2.LINE_AA)

        # Summary line
        summary = f"Saved {saved_count}/3 angle embeddings to database"
        sm_size = cv2.getTextSize(summary, FONT, 0.5, 1)[0]
        cv2.putText(frame, summary, ((w - sm_size[0]) // 2, card_y + card_h + 35),
                    FONT, 0.5, C_WHITE, 1, cv2.LINE_AA)

        avg_txt = f"DB will compute mean embedding from {saved_count} angles for matching"
        a_size = cv2.getTextSize(avg_txt, FONT, 0.4, 1)[0]
        cv2.putText(frame, avg_txt, ((w - a_size[0]) // 2, card_y + card_h + 60),
                    FONT, 0.4, C_DIM, 1, cv2.LINE_AA)

        # Close timer
        remaining = max(0, duration - elapsed)
        cv2.putText(frame, f"Closing in {remaining:.0f}s... (Q to close now)",
                    ((w - 320) // 2, h - 30), FONT, 0.4, C_DIM, 1, cv2.LINE_AA)

        cv2.imshow(WIN_NAME, frame)

        if elapsed >= duration:
            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


def show_transition(cap, phase_idx, phase_info, phase_results, student_id, name, class_name, duration=2.0):
    """Transition screen between phases with countdown."""
    start = time.time()
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue

        remaining = max(0, duration - (time.time() - start))
        draw_dark_overlay(frame, 0.45)
        h, w = frame.shape[:2]

        # Phase announcement
        txt = f"Phase {phase_idx+1}/3"
        t_size = cv2.getTextSize(txt, FONT, 1.2, 3)[0]
        cv2.putText(frame, txt, ((w - t_size[0]) // 2, h // 2 - 40),
                    FONT, 1.2, phase_info["color"], 3, cv2.LINE_AA)

        instruction = phase_info["instruction"]
        i_size = cv2.getTextSize(instruction, FONT, 0.7, 2)[0]
        cv2.putText(frame, instruction, ((w - i_size[0]) // 2, h // 2 + 10),
                    FONT, 0.7, phase_info["color"], 2, cv2.LINE_AA)

        sub = phase_info["sub"]
        s_size = cv2.getTextSize(sub, FONT, 0.45, 1)[0]
        cv2.putText(frame, sub, ((w - s_size[0]) // 2, h // 2 + 40),
                    FONT, 0.45, C_DIM, 1, cv2.LINE_AA)

        # Ready countdown
        ready = f"Get ready... {remaining:.0f}"
        r_size = cv2.getTextSize(ready, FONT, 0.55, 1)[0]
        cv2.putText(frame, ready, ((w - r_size[0]) // 2, h // 2 + 80),
                    FONT, 0.55, C_GOLD, 1, cv2.LINE_AA)

        # Phase dots
        draw_phase_dots(frame, phase_idx, 3, phase_results)

        # Student info
        draw_student_info(frame, student_id, name, class_name)

        cv2.imshow(WIN_NAME, frame)
        cv2.waitKey(1)


def show_calibration(cap, mesh_landmarker, mesh_ts_start=0):
    """Calibrate baseline nose position with visual feedback."""
    print("Calibrating baseline pose (look straight)...")
    nose_positions = []
    mesh_ts = mesh_ts_start
    start = time.time()
    cal_duration = 2.0

    while time.time() - start < cal_duration:
        ret, frame = cap.read()
        if not ret:
            continue

        elapsed = time.time() - start
        progress = elapsed / cal_duration

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

        h, w = frame.shape[:2]

        # Face guide
        cx, cy = w // 2, h // 2
        cv2.ellipse(frame, (cx, cy), (140, 180), 0, 0, 360, C_BLUE, 2, cv2.LINE_AA)
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), C_BLUE, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), C_BLUE, 2, cv2.LINE_AA)

        # Header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), C_BLACK, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cal_text = "CALIBRATING — Look straight at camera"
        ct_size = cv2.getTextSize(cal_text, FONT, 0.6, 2)[0]
        cv2.putText(frame, cal_text, ((w - ct_size[0]) // 2, 35), FONT, 0.6, C_GOLD, 2, cv2.LINE_AA)

        # Progress bar
        draw_progress_bar(frame, progress, 50, C_BLUE, "Measuring baseline pose...")

        # Status dots
        detected = len(nose_positions)
        status = f"Samples: {detected}"
        cv2.putText(frame, status, (20, h - 30), FONT, 0.4, C_GREEN if detected > 5 else C_ORANGE, 1, cv2.LINE_AA)

        cv2.imshow(WIN_NAME, frame)
        cv2.waitKey(1)

    if nose_positions:
        baseline = float(np.median(nose_positions))
        print(f"Baseline nose_x = {baseline:.3f} ({len(nose_positions)} samples)")
        return baseline, mesh_ts
    return 0.5, mesh_ts


# ═══════════════════════════════════════════════════════════
#  ENROLLMENT LOGIC
# ═══════════════════════════════════════════════════════════

def get_nose_x(face_landmarks, img_w):
    """Get normalized nose X position from MediaPipe landmarks."""
    if face_landmarks and len(face_landmarks) > 1:
        return face_landmarks[1].x
    return 0.5


def verify_pose(nose_x, baseline_nose_x, phase_info):
    """Check if the user's head pose matches the phase requirement."""
    verify_type = phase_info["verify"]

    if verify_type == "center":
        return abs(nose_x - 0.5) < 0.12

    elif verify_type == "left":
        dx = baseline_nose_x - nose_x
        return dx > TURN_THRESHOLD

    elif verify_type == "right":
        dx = nose_x - baseline_nose_x
        return dx > TURN_THRESHOLD

    return True


def run_phase(cap, engine, mesh_landmarker, phase_idx, phase_info, baseline_nose_x, phase_results,
              student_id, name, class_name):
    """Run one enrollment phase: capture multi-frame embeddings at a specific pose.

    Returns:
        dict with name, embedding, frame_count, best_aligned, or None if cancelled
    """
    print(f"\n[Phase {phase_idx+1}] {phase_info['instruction']}")

    collected_embeddings = []
    collected_qualities = []
    best_aligned = None
    best_blur = 0
    current_aligned = None
    mesh_ts = phase_idx * 10000

    # Pulse animation state
    pulse_intensity = 0.0
    pulse_decay = 3.0  # decay speed

    start = time.time()

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        frame = raw_frame.copy()
        now = time.time()
        elapsed = now - start
        remaining = max(0.0, PHASE_DURATION - elapsed)
        dt = 0.033  # approximate frame delta

        # Decay pulse
        pulse_intensity = max(0.0, pulse_intensity - pulse_decay * dt)

        # ── MediaPipe landmarks for pose verification ───
        nose_x = 0.5
        if mesh_landmarker is not None:
            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mesh_ts += 33
            try:
                mesh_result = mesh_landmarker.detect_for_video(mp_image, mesh_ts)
                if mesh_result.face_landmarks:
                    nose_x = get_nose_x(mesh_result.face_landmarks[0], FRAME_W)
            except Exception:
                pass

        pose_ok = verify_pose(nose_x, baseline_nose_x, phase_info)

        # ── Face detection + quality + embedding ────────
        blur_val = 0.0
        try:
            faces = engine.detect(raw_frame)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                x1, y1, x2, y2 = face.bbox[:4]

                roi = raw_frame[max(0,y1):y2, max(0,x1):x2]
                if roi.size > 0:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    blur_val = cv2.Laplacian(gray_roi, cv2.CV_64F).var()

                # Update current aligned face for thumbnail
                if face.aligned_face is not None:
                    current_aligned = face.aligned_face.copy()

                # Bounding box
                is_good = pose_ok and blur_val > QUALITY_BLUR_MIN
                bb_color = C_GREEN if is_good else C_RED
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              bb_color, 2, cv2.LINE_AA)

                # Collect if quality + pose are good
                if (is_good
                    and face.embedding is not None
                    and len(face.embedding) > 0
                    and len(collected_embeddings) < MAX_GOOD_FRAMES):

                    collected_embeddings.append(face.embedding.copy())
                    collected_qualities.append(blur_val)

                    # Track best aligned face
                    if blur_val > best_blur and face.aligned_face is not None:
                        best_blur = blur_val
                        best_aligned = face.aligned_face.copy()

                    # Trigger capture pulse!
                    pulse_intensity = 1.0

                    # Green highlight
                    cv2.rectangle(frame, (int(x1)-3, int(y1)-3), (int(x2)+3, int(y2)+3),
                                  C_PULSE_GREEN, 4, cv2.LINE_AA)

        except Exception as e:
            print(f"[WARN] Detect error: {e}")

        # ── Draw UI layers ──────────────────────────────
        # 1. Face guide
        draw_face_guide(frame, phase_info)

        # 2. Phase header + countdown
        draw_phase_header(frame, phase_idx, phase_info, remaining)

        # 3. Phase dots (top-right)
        draw_phase_dots(frame, phase_idx, 3, phase_results)

        # 4. Quality panel (left)
        draw_quality_panel(frame, len(collected_embeddings), MIN_GOOD_FRAMES,
                           MAX_GOOD_FRAMES, blur_val, pose_ok)

        # 5. Student info (top-left, below header)
        draw_student_info(frame, student_id, name, class_name)

        # 6. Aligned face thumbnail (bottom-right)
        if current_aligned is not None:
            draw_face_thumbnail(frame, current_aligned, phase_info["name"], phase_info["color"])

        # 7. Capture pulse
        draw_capture_pulse(frame, pulse_intensity)

        # 8. Bottom HUD
        draw_bottom_hud(frame, [
            (f"{phase_info['icon']} {phase_info['instruction']}", 0.48, phase_info['color']),
            (f"Captured: {len(collected_embeddings)}/{MAX_GOOD_FRAMES}  |  Blur: {blur_val:.0f}  |  Q=Cancel  SPACE=Skip", 0.38, C_DIM),
        ])

        cv2.imshow(WIN_NAME, frame)

        # ── Exit conditions ─────────────────────────────
        if elapsed >= PHASE_DURATION:
            break
        if len(collected_embeddings) >= MAX_GOOD_FRAMES:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return None  # cancelled
        if key == ord(' '):
            break  # skip to next

    # ── Phase result ────────────────────────────────────
    if len(collected_embeddings) < MIN_GOOD_FRAMES:
        print(f"[Phase {phase_idx+1}] FAILED — only {len(collected_embeddings)} good frames "
              f"(need {MIN_GOOD_FRAMES})")
        return {
            "name": phase_info["name"],
            "embedding": None,
            "frame_count": len(collected_embeddings),
            "best_aligned": best_aligned,
            "quality": 0.0,
            "success": False,
        }

    # Average embeddings
    emb_stack = np.stack(collected_embeddings)
    mean_emb = np.mean(emb_stack, axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm

    avg_quality = float(np.mean(collected_qualities))
    print(f"[Phase {phase_idx+1}] OK — {len(collected_embeddings)} frames averaged, quality={avg_quality:.0f}")

    return {
        "name": phase_info["name"],
        "embedding": mean_emb,
        "frame_count": len(collected_embeddings),
        "best_aligned": best_aligned,
        "quality": avg_quality,
        "success": True,
    }


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

    # ── MediaPipe FaceLandmarker ────────────────────────
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

    print("\nOpening camera...")
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera (index {CAM_INDEX})")
        sys.exit(1)

    # ── Welcome screen ──────────────────────────────────
    if not show_welcome_screen(cap, student_id, name, class_name, duration=3.0):
        cap.release()
        cv2.destroyAllWindows()
        print("Cancelled by user.")
        sys.exit(0)

    # ── Calibrate baseline ──────────────────────────────
    baseline_nose_x, mesh_ts = show_calibration(cap, mesh_landmarker)

    # ── Run 3 phases (with retry) ───────────────────────
    phase_results = []
    MAX_RETRIES = 1

    for i, phase_info in enumerate(PHASES):
        retries = 0
        while True:
            # Transition screen
            show_transition(cap, i, phase_info, phase_results, student_id, name, class_name, duration=2.0)

            # Run phase
            result = run_phase(cap, engine, mesh_landmarker, i, phase_info, baseline_nose_x,
                               phase_results, student_id, name, class_name)

            if result is None:
                cap.release()
                cv2.destroyAllWindows()
                print("\nEnrollment cancelled by user.")
                sys.exit(0)

            if not result["success"] and retries < MAX_RETRIES:
                # Ask retry
                retry = show_retry_prompt(cap, i, phase_info, result)
                if retry:
                    retries += 1
                    continue  # retry this phase
                else:
                    phase_results.append(result)
                    break
            else:
                phase_results.append(result)
                break

    # ── Save to database ────────────────────────────────
    successful_phases = [r for r in phase_results if r["success"] and r["embedding"] is not None]

    if len(successful_phases) == 0:
        show_final_summary(cap, phase_results, student_id, name, 0, duration=4.0)
        cap.release()
        cv2.destroyAllWindows()
        print("\nEnrollment FAILED — no phases succeeded.")
        sys.exit(1)

    from core.database import get_db
    db = get_db()

    db.add_student(student_id, name, class_name)

    old_count = db.get_embedding_count(student_id)
    if old_count > 0:
        db.delete_embeddings(student_id)
        print(f"Deleted {old_count} old embedding(s)")

    saved = 0
    for pr in successful_phases:
        quality = pr.get("quality", 0.0)
        source = f"v2_{pr['name'].lower()}"
        db.save_embedding(student_id, pr["embedding"], quality, source)
        saved += 1
        print(f"  Saved: {source} ({pr['frame_count']} frames avg, quality={quality:.0f})")

    # Save photo
    try:
        engine._ensure_model()
        ret, frame = cap.read()
        if ret:
            face = engine.detect_largest(frame)
            if face is not None and face.aligned_face is not None:
                from datetime import datetime
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                photo_path = str(config.FACE_CROPS_DIR / f"{student_id}_{ts}.jpg")
                cv2.imwrite(photo_path, face.aligned_face)
                db.update_student(student_id, photo_path=photo_path)
    except Exception:
        pass

    engine.reload_cache()

    # ── Final summary screen ────────────────────────────
    show_final_summary(cap, phase_results, student_id, name, saved, duration=5.0)

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print(f"  Enrollment V2 complete: {name} ({student_id})")
    print(f"{'='*60}")
    for pr in phase_results:
        status = "OK" if pr["success"] else "FAILED"
        print(f"  {pr['name']:>8}: {status} ({pr['frame_count']} frames)")
    print(f"  Total embeddings saved: {saved}")
    print(f"  DB will average {saved} embeddings for matching.")


def show_retry_prompt(cap, phase_idx, phase_info, result):
    """Show retry prompt when a phase fails. Returns True if user wants retry."""
    start = time.time()
    duration = 5.0  # auto-skip after 5s

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        elapsed = time.time() - start
        remaining = max(0, duration - elapsed)
        draw_dark_overlay(frame, 0.5)
        h, w = frame.shape[:2]

        # Warning
        warn = f"Phase {phase_idx+1} FAILED"
        w_size = cv2.getTextSize(warn, FONT, 0.8, 2)[0]
        cv2.putText(frame, warn, ((w - w_size[0]) // 2, h // 2 - 50),
                    FONT, 0.8, C_RED, 2, cv2.LINE_AA)

        detail = f"Only {result['frame_count']} good frames (need {MIN_GOOD_FRAMES})"
        d_size = cv2.getTextSize(detail, FONT, 0.5, 1)[0]
        cv2.putText(frame, detail, ((w - d_size[0]) // 2, h // 2 - 15),
                    FONT, 0.5, C_ORANGE, 1, cv2.LINE_AA)

        # Options
        opt1 = "Press R to RETRY this phase"
        opt2 = "Press SPACE or wait to SKIP"
        o1_size = cv2.getTextSize(opt1, FONT, 0.5, 1)[0]
        o2_size = cv2.getTextSize(opt2, FONT, 0.5, 1)[0]
        cv2.putText(frame, opt1, ((w - o1_size[0]) // 2, h // 2 + 30),
                    FONT, 0.5, C_GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, opt2, ((w - o2_size[0]) // 2, h // 2 + 60),
                    FONT, 0.5, C_DIM, 1, cv2.LINE_AA)

        cv2.putText(frame, f"Auto-skip in {remaining:.0f}s...",
                    ((w - 200) // 2, h // 2 + 100), FONT, 0.4, C_DIM, 1, cv2.LINE_AA)

        cv2.imshow(WIN_NAME, frame)

        if elapsed >= duration:
            return False  # auto-skip

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            return True   # retry
        if key == ord(' ') or key == ord('q'):
            return False  # skip


if __name__ == "__main__":
    main()
