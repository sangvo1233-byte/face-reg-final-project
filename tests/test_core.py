"""
Integration test — Diem Danh Hoc Sinh (Full Pipeline)

Phu thuoc:
  - Video ngoai repo: C:\\Users\\ADMIN\\Desktop\\Projects\\face-attendance\\test_video\\
  - DB va logs runtime that su (ghi vao database/ va logs/)

Chay thu cong (khi co du file):
  pytest tests/test_core.py -m integration -s

Suite mac dinh (pytest -q) se tu dong skip neu khong tim thay thu muc video.
"""
import pytest

# Mark toan bo file nay la integration-only.
# De chay: pytest -m integration
pytestmark = pytest.mark.integration
import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))

from core.face_engine import get_engine
from core.database import get_db

# Path to test videos (from face-attendance project)
# This directory lives outside the repo and is NOT included in version control.
FACE_ATTENDANCE_DIR = Path(r"C:\Users\ADMIN\Desktop\Projects\face-attendance")
TEST_VIDEO_DIR = FACE_ATTENDANCE_DIR / "test_video"


def extract_frames(video_path: str, frame_indices: list[int]) -> list[np.ndarray]:
    """Trich xuat cac frame cu the tu video."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_indices:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def extract_best_frame(video_path: str, skip: int = 20) -> np.ndarray:
    """Doc video, skip N frame dau, tra frame ro nhat."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx >= skip:
            frames.append(frame)
            if len(frames) >= 10:
                break
        idx += 1
    cap.release()

    if not frames:
        return None

    # Chon frame sac nhat (Laplacian variance)
    scores = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    return frames[int(np.argmax(scores))]


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def test_full_pipeline():
    """Test toan bo pipeline: enroll -> session -> scan -> result.

    Integration test: chi chay khi TEST_VIDEO_DIR ton tai tren disk.
    """
    if not TEST_VIDEO_DIR.exists():
        pytest.skip(
            f"Integration test skipped: test video dir not found at {TEST_VIDEO_DIR}. "
            "Run with: pytest tests/test_core.py -m integration -s"
        )
    import builtins
    result_file = open(str(Path(__file__).resolve().parent / "test_result.txt"), "w", encoding="utf-8")
    _original_print = builtins.print
    def _print(*args, **kwargs):
        kwargs.setdefault('file', None)
        if kwargs['file'] is None:
            # Print to both stdout and file
            _original_print(*args, **kwargs)
            kwargs_copy = dict(kwargs)
            kwargs_copy['file'] = result_file
            _original_print(*args, **kwargs_copy)
        else:
            _original_print(*args, **kwargs)
    builtins.print = _print

    print("=" * 60)
    print("TEST: Diem Danh Hoc Sinh - Full Pipeline")
    print("=" * 60)

    engine = get_engine()
    db = get_db()

    # Verify test video files exist
    videos = sorted(TEST_VIDEO_DIR.glob("*.mp4"))
    print(f"\nFound {len(videos)} test videos in: {TEST_VIDEO_DIR}")
    for v in videos:
        print(f"  - {v.name}")

    if len(videos) < 2:
        print("FAIL: Can it nhat 2 test videos")
        return False

    # ── STEP 1: ENROLL ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: ENROLL HOC SINH (tu video 1)")
    print("=" * 60)

    # Dung video 1 de enroll: trich frame, detect, dang ky
    enroll_video = videos[0]
    print(f"\nDang enroll tu: {enroll_video.name}")

    frame = extract_best_frame(str(enroll_video))
    if frame is None:
        print("FAIL: Khong doc duoc frame tu video")
        return False

    print(f"  Frame size: {frame.shape}")

    # Detect tat ca khuon mat trong frame
    faces = engine.detect(frame)
    faces = sorted(faces, key=lambda f: f.bbox[0])  # Sap xep trai -> phai
    print(f"  Detected: {len(faces)} khuon mat")

    if len(faces) == 0:
        print("FAIL: Khong detect duoc khuon mat nao")
        return False

    # Dang ky cac hoc sinh
    names = ["Nguyen Van An", "Tran Thi Binh", "Le Hoang Cuong",
             "Pham Thu Dung", "Vu Minh Phu"]
    enrolled = 0
    for i, face in enumerate(faces):
        if i >= len(names):
            break
        if face.embedding is None or len(face.embedding) == 0:
            print(f"  SKIP: Face {i} khong co embedding")
            continue

        student_id = f"HS{i+1:03d}"
        name = names[i]

        db.add_student(student_id, name, "12A1")
        db.save_embedding(student_id, face.embedding, 100.0, 'test_video')

        # Save crop
        photo_path = str(Path("logs/face_crops") / f"{student_id}_test.jpg")
        cv2.imwrite(photo_path, face.aligned_face)
        db.update_student(student_id, photo_path=photo_path)

        bbox = face.bbox.astype(int)
        print(f"  [OK] {student_id}: {name} (bbox={bbox.tolist()}, conf={face.confidence:.3f})")
        enrolled += 1

    engine.reload_cache()
    print(f"\nEnrolled: {enrolled} hoc sinh")
    print(f"DB: {db.get_student_count()} students, {db.get_embedding_count()} embeddings")

    if enrolled == 0:
        print("FAIL: Khong enroll duoc hoc sinh nao")
        return False

    # ── STEP 2: TAO PHIEN DIEM DANH ────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: TAO PHIEN DIEM DANH")
    print("=" * 60)

    session_id = db.create_session("Buoi test tu dong", "12A1")
    print(f"  Session ID: {session_id}")

    # ── STEP 3: QUET DIEM DANH ──────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: QUET DIEM DANH (tu cac video)")
    print("=" * 60)

    total_scanned = 0
    total_recognized = 0
    total_new = 0

    for video in videos:
        print(f"\n--- Scanning: {video.name} ---")
        frame = extract_best_frame(str(video))
        if frame is None:
            print(f"  SKIP: Khong doc duoc")
            continue

        result = engine.scan_attendance(frame, session_id)
        total_scanned += result['faces_detected']
        total_recognized += result['recognized']

        for r in result['results']:
            status_label = {
                'present': '[CO MAT]',
                'already': '[DA DIEM DANH]',
                'unknown': '[KHONG NHAN DIEN]',
                'spoof':   '[GIA MAO]',
            }.get(r['status'], f'[{r["status"]}]')

            if r['status'] == 'present':
                total_new += 1

            conf_str = f"conf={r['confidence']:.3f}" if r['confidence'] > 0 else ""
            print(f"  {status_label} {r['name']} {r.get('student_id', '')} {conf_str}")

        if not result['results']:
            print(f"  Khong phat hien khuon mat nao")

    # ── STEP 4: KET QUA ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: KET QUA PHIEN DIEM DANH")
    print("=" * 60)

    session_result = db.end_session(session_id)
    full_result = db.get_session_result(session_id)

    print(f"\n  Tong khuon mat quet: {total_scanned}")
    print(f"  Nhan dien duoc: {total_recognized}")
    print(f"  Diem danh moi: {total_new}")
    print(f"\n  Co mat: {full_result['present_count']}/{full_result['total']}")
    print(f"  Vang:   {full_result['absent_count']}/{full_result['total']}")

    print("\n  Danh sach co mat:")
    for s in full_result['present']:
        print(f"    [V] {s['id']}: {s['name']}")

    print("\n  Danh sach vang:")
    for s in full_result['absent']:
        print(f"    [X] {s['id']}: {s['name']}")

    # ── SUMMARY ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Enrolled: {enrolled} hoc sinh")
    print(f"  Videos scanned: {len(videos)}")
    print(f"  Faces detected: {total_scanned}")
    print(f"  Recognized: {total_recognized}")
    print(f"  Attendance: {full_result['present_count']}/{full_result['total']}")
    success_rate = (full_result['present_count'] / full_result['total'] * 100) if full_result['total'] > 0 else 0
    print(f"  Success rate: {success_rate:.0f}%")
    print(f"  Result: {'PASS' if full_result['present_count'] > 0 else 'FAIL'}")
    print("=" * 60)

    builtins.print = _original_print
    result_file.close()
    return full_result['present_count'] > 0


if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
