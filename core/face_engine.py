"""
Face Engine — AI core for face recognition.

Pipeline: Detect (RetinaFace) -> Align -> Embed (ArcFace) -> Match.
Includes quality gate and passive liveness check.
"""
import cv2
import numpy as np
from loguru import logger

import config
from core.schemas import DetectedFace, MatchResult, QualityResult
from core.database import get_db
from core.anti_spoof import get_anti_spoof


class FaceEngine:
    """AI Engine: Detect + Recognize + Match + Enrollment."""

    def __init__(self):
        self._app = None
        self._embeddings: np.ndarray = None
        self._identities: list[dict] = []
        self._cache_loaded = False

    # ── Model ───────────────────────────────────────────────

    def _ensure_model(self):
        if self._app is not None:
            return
        logger.info(f"Loading InsightFace: {config.INSIGHTFACE_MODEL}")
        from insightface.app import FaceAnalysis
        self._app = FaceAnalysis(
            name=config.INSIGHTFACE_MODEL,
            root=str(config.MODELS_DIR),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self._app.prepare(ctx_id=0, det_size=config.DET_SIZE)
        try:
            import onnxruntime as _ort
            gpu = 'CUDAExecutionProvider' in _ort.get_available_providers()
        except Exception:
            gpu = False
        logger.info(f"Model loaded ({'GPU' if gpu else 'CPU'})")

    def _load_embeddings_cache(self):
        db = get_db()
        self._embeddings, self._identities = db.get_all_embeddings()
        self._cache_loaded = True
        logger.info(f"Loaded {len(self._identities)} identities from DB")

    def reload_cache(self):
        self._cache_loaded = False
        self._load_embeddings_cache()

    # ── Detection ───────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        self._ensure_model()
        faces = []
        for face in self._app.get(frame):
            if face.det_score < config.DET_CONFIDENCE:
                continue
            bbox = face.bbox.astype(int)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if w < config.MIN_FACE_SIZE or h < config.MIN_FACE_SIZE:
                continue
            aligned = self._align_face(frame, face)
            emb = (face.normed_embedding
                   if hasattr(face, 'normed_embedding')
                   and face.normed_embedding is not None
                   else np.array([]))
            faces.append(DetectedFace(
                bbox=bbox,
                landmarks=face.kps if hasattr(face, 'kps') else None,
                confidence=float(face.det_score),
                aligned_face=aligned,
                embedding=emb
            ))
        return faces

    def detect_largest(self, frame: np.ndarray) -> DetectedFace | None:
        faces = self.detect(frame)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def _align_face(self, frame, face) -> np.ndarray:
        if hasattr(face, 'kps') and face.kps is not None:
            dst = np.array([
                [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                [41.5493, 92.3655], [70.7299, 92.2041]
            ], dtype=np.float32)
            src = face.kps[:5].astype(np.float32)
            M = cv2.estimateAffinePartial2D(src, dst)[0]
            if M is None:
                M = cv2.getAffineTransform(src[:3], dst[:3])
            return cv2.warpAffine(frame, M, (112, 112))
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        return cv2.resize(frame[y1:y2, x1:x2], (112, 112))

    # ── Quality Gate ────────────────────────────────────────

    def quality_check(self, frame: np.ndarray, face: DetectedFace) -> QualityResult:
        bbox = face.bbox.astype(int) if isinstance(face.bbox, np.ndarray) else face.bbox
        x1, y1, x2, y2 = bbox[:4]
        face_size = min(x2 - x1, y2 - y1)

        h_img, w_img = frame.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w_img, x2), min(h_img, y2)
        face_crop = frame[y1c:y2c, x1c:x2c]

        if face_crop.size == 0:
            return QualityResult(passed=False, face_size=0, blur_score=0,
                                 brightness=0, yaw_angle=0, reasons=["empty_face_region"])

        gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
        brightness = float(np.mean(gray_crop))

        yaw_angle = 0.0
        if face.landmarks is not None and len(face.landmarks) >= 5:
            try:
                left_eye, right_eye, nose = face.landmarks[0], face.landmarks[1], face.landmarks[2]
                eye_center_x = (left_eye[0] + right_eye[0]) / 2
                eye_dist = abs(right_eye[0] - left_eye[0])
                yaw_angle = abs(nose[0] - eye_center_x) / eye_dist * 90 if eye_dist > 0 else 0
            except (IndexError, TypeError):
                pass

        reasons = []
        if face_size < config.QUALITY_MIN_FACE_SIZE:
            reasons.append(f"face_too_small ({face_size}px)")
        if blur_score < config.QUALITY_MAX_BLUR:
            reasons.append(f"too_blurry ({blur_score:.0f})")
        if brightness < config.QUALITY_MIN_BRIGHTNESS:
            reasons.append(f"too_dark ({brightness:.0f})")
        if brightness > config.QUALITY_MAX_BRIGHTNESS:
            reasons.append(f"too_bright ({brightness:.0f})")
        if yaw_angle > config.QUALITY_MAX_YAW_ANGLE:
            reasons.append(f"face_turned ({yaw_angle:.0f}°)")

        return QualityResult(passed=len(reasons) == 0, face_size=face_size,
                             blur_score=blur_score, brightness=brightness,
                             yaw_angle=yaw_angle, reasons=reasons)

    # ── Matching ────────────────────────────────────────────

    def match(self, embedding: np.ndarray) -> MatchResult:
        if not self._cache_loaded:
            self._load_embeddings_cache()
        if self._embeddings is None or len(self._embeddings) == 0:
            return MatchResult(False, "", "", 0.0, -1)
        sims = np.dot(self._embeddings, embedding)
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        if score >= config.COSINE_THRESHOLD and idx < len(self._identities):
            ident = self._identities[idx]
            return MatchResult(True, ident['name'], ident['student_id'], score, idx)
        return MatchResult(False, "", "", score, idx)

    def match_with_threshold(self, embedding: np.ndarray,
                             threshold: float) -> MatchResult:
        """Match using a caller-specified cosine threshold.

        Identical to match() but uses *threshold* instead of
        config.COSINE_THRESHOLD.  Used by DetectV3Service for the
        stricter 0.52 threshold.
        """
        if not self._cache_loaded:
            self._load_embeddings_cache()
        if self._embeddings is None or len(self._embeddings) == 0:
            return MatchResult(False, "", "", 0.0, -1)
        sims = np.dot(self._embeddings, embedding)
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        if score >= threshold and idx < len(self._identities):
            ident = self._identities[idx]
            return MatchResult(True, ident['name'], ident['student_id'], score, idx)
        return MatchResult(False, "", "", score, idx)

    # ── Face Metrics (V2 helper) ────────────────────────────

    def get_face_metrics(self, frame: np.ndarray,
                         face: DetectedFace) -> dict:
        """Return detailed face metrics including pose information.

        Extends quality_check with nose_x displacement for
        pose classification (front / left / right).
        """
        qr = self.quality_check(frame, face)

        # Compute nose_x displacement from eye center
        nose_x_disp = 0.0
        if face.landmarks is not None and len(face.landmarks) >= 5:
            try:
                left_eye = face.landmarks[0]
                right_eye = face.landmarks[1]
                nose = face.landmarks[2]
                eye_center_x = (left_eye[0] + right_eye[0]) / 2
                eye_dist = abs(right_eye[0] - left_eye[0])
                if eye_dist > 0:
                    nose_x_disp = (nose[0] - eye_center_x) / eye_dist
            except (IndexError, TypeError):
                pass

        return {
            'passed': qr.passed,
            'face_size': qr.face_size,
            'blur_score': qr.blur_score,
            'brightness': qr.brightness,
            'yaw_angle': qr.yaw_angle,
            'nose_x_disp': nose_x_disp,
            'reasons': qr.reasons,
        }

    # ── Enrollment ──────────────────────────────────────────

    def enroll_from_photo(self, student_id: str, name: str,
                          photo_frame: np.ndarray,
                          class_name: str = '') -> dict:
        """Enroll a student from a single photo frame."""
        self._ensure_model()
        face = self.detect_largest(photo_frame)
        if face is None:
            return {'success': False, 'message': 'No face detected in the image'}
        if face.embedding is None or len(face.embedding) == 0:
            return {'success': False, 'message': 'Failed to extract face embedding'}

        qr = self.quality_check(photo_frame, face)
        if not qr.passed:
            return {'success': False, 'message': f'Image quality check failed: {", ".join(qr.reasons)}'}

        db = get_db()
        db.add_student(student_id, name, class_name)
        db.delete_embeddings(student_id)
        db.save_embedding(student_id, face.embedding, qr.blur_score, 'photo')

        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_path = str(config.FACE_CROPS_DIR / f"{student_id}_{ts}.jpg")
        cv2.imwrite(photo_path, face.aligned_face)
        db.update_student(student_id, photo_path=photo_path)

        self.reload_cache()
        logger.info(f"Enrolled: {name} ({student_id})")
        return {'success': True, 'message': f'Enrolled: {name}', 'embeddings_saved': 1}

    # ── Scan (điểm danh) ────────────────────────────────────

    def scan_attendance(self, frame: np.ndarray, session_id: int) -> dict:
        """Scan a single frame, recognize faces, and record attendance.

        Returns:
            dict: {faces_detected, recognized, results: [{name, student_id, confidence, status}]}
        """
        self._ensure_model()
        from datetime import datetime

        faces = self.detect(frame)
        results = []

        for face in faces:
            if face.embedding is None or len(face.embedding) == 0:
                continue

            # Liveness check
            anti_spoof = get_anti_spoof()
            liveness = anti_spoof.check(frame, face.bbox)
            if not liveness.is_live:
                results.append({
                    'name': 'Unknown', 'student_id': '', 'confidence': 0,
                    'status': 'spoof', 'message': f'Liveness check failed ({liveness.reason})',
                    'bbox': face.bbox.tolist()
                })
                continue

            # Match
            match = self.match(face.embedding)
            if not match.matched:
                results.append({
                    'name': 'Unknown', 'student_id': '', 'confidence': match.score,
                    'status': 'unknown', 'message': 'No match found',
                    'bbox': face.bbox.tolist()
                })
                continue

            # Record attendance
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            evidence = str(config.EVIDENCE_DIR / f"{match.student_id}_{ts}.jpg")
            cv2.imwrite(evidence, frame)

            db = get_db()
            db_result = db.mark_attendance(session_id, match.student_id, match.score, evidence)

            results.append({
                'name': match.name, 'student_id': match.student_id,
                'confidence': match.score,
                'status': 'present' if db_result['success'] else 'already',
                'message': db_result['message'],
                'bbox': face.bbox.tolist()
            })

        return {
            'faces_detected': len(faces),
            'recognized': sum(1 for r in results if r['status'] in ('present', 'already')),
            'results': results
        }

    # ── Info ────────────────────────────────────────────────

    def get_identity_count(self) -> int:
        if not self._cache_loaded:
            self._load_embeddings_cache()
        return len(self._identities)


# ── Singleton ───────────────────────────────────────────────

_engine = None

def get_engine() -> FaceEngine:
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine
