"""
Enrollment V2 Service — Multi-Angle Face Enrollment.

Processes 3 pre-captured images (front, left, right) to create
a robust multi-angle embedding set for each student.

Usage:
    service = EnrollmentV2Service()
    result = service.enroll_multi_angle(
        student_id="HS001",
        name="Nguyen Van A",
        class_name="12A1",
        images={"front": img_front, "left": img_left, "right": img_right},
    )
"""
import cv2
import numpy as np
from datetime import datetime
from loguru import logger

import config
from core.face_engine import get_engine
from core.database import get_db


# Phase definitions matching dev/enroll-v2.py
PHASES = [
    {"name": "FRONT", "verify": "center"},
    {"name": "LEFT",  "verify": "left"},
    {"name": "RIGHT", "verify": "right"},
]


class EnrollmentV2Service:
    """Stateless multi-angle enrollment service.

    Accepts 3 images (front, left, right), validates each,
    extracts embeddings, and saves to the database.
    """

    def enroll_multi_angle(
        self,
        student_id: str,
        name: str,
        class_name: str,
        images: dict[str, np.ndarray],
    ) -> dict:
        """Enroll a student using multi-angle images.

        Args:
            student_id: Unique student identifier
            name: Full name
            class_name: Class name (optional, can be empty)
            images: Dict with keys "front", "left", "right" →
                    BGR numpy arrays

        Returns:
            dict with success, message, phase_results, total_saved
        """
        engine = get_engine()
        engine._ensure_model()

        required_keys = {"front", "left", "right"}
        provided_keys = set(images.keys())
        if not required_keys.issubset(provided_keys):
            missing = required_keys - provided_keys
            return {
                "success": False,
                "message": f"Missing images: {', '.join(missing)}",
                "phase_results": [],
                "total_saved": 0,
            }

        phase_results = []

        for phase in PHASES:
            angle = phase["name"].lower()   # "front", "left", "right"
            img = images[angle]

            result = self._process_angle(engine, img, phase)
            phase_results.append(result)

        # All 3 angles must pass
        successful = [r for r in phase_results if r["success"]]

        if len(successful) < len(PHASES):
            return {
                "success": False,
                "message": "All 3 angles must pass to enroll. Please retake failed angles.",
                "phase_results": [
                    {
                        "name": r["name"],
                        "success": r["success"],
                        "quality": r.get("quality", 0),
                        "reason": r.get("reason", ""),
                    }
                    for r in phase_results
                ],
                "total_saved": 0,
                "embeddings_saved": 0,
            }

        # ── Save to database ────────────────────────────
        db = get_db()
        
        # Update existing student metadata on re-enroll, or create new
        existing = db.get_student_any(student_id)
        if existing:
            db.update_student(student_id, name=name, class_name=class_name)
            if existing.get('is_active') == 0:
                db.restore_student(student_id)
        else:
            db.add_student(student_id, name, class_name)

        # Replace old embeddings (only happens after new set is fully validated)
        old_count = db.get_embedding_count(student_id)
        if old_count > 0:
            db.delete_embeddings(student_id)
            logger.info(f"Deleted {old_count} old embedding(s) for {student_id}")

        saved = 0
        best_aligned = None
        best_blur = 0.0

        for pr in successful:
            source = f"v2_{pr['name'].lower()}"
            quality = pr.get("quality", 0.0)
            db.save_embedding(student_id, pr["embedding"], quality, source)
            saved += 1
            logger.info(
                f"  Saved: {source} (quality={quality:.0f})"
            )

            # Track best aligned face for photo
            if pr.get("blur_score", 0) > best_blur and pr.get("best_aligned") is not None:
                best_blur = pr["blur_score"]
                best_aligned = pr["best_aligned"]

        # Save face crop photo
        if best_aligned is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            photo_path = str(config.FACE_CROPS_DIR / f"{student_id}_{ts}.jpg")
            cv2.imwrite(photo_path, best_aligned)
            db.update_student(student_id, photo_path=photo_path)

        engine.reload_cache()

        logger.info(
            f"Enrollment V2 complete: {name} ({student_id}) — "
            f"{saved}/{len(PHASES)} angles saved"
        )

        return {
            "success": True,
            "message": f"Enrolled: {name} ({saved}/{len(PHASES)} angles)",
            "phase_results": [
                {
                    "name": r["name"],
                    "success": r["success"],
                    "quality": r.get("quality", 0),
                    "reason": r.get("reason", ""),
                }
                for r in phase_results
            ],
            "total_saved": saved,
            "embeddings_saved": saved,
        }

    def _process_angle(self, engine, image: np.ndarray, phase: dict) -> dict:
        """Process a single angle image.

        Returns a result dict with embedding, quality, etc.
        """
        angle_name = phase["name"]
        verify_type = phase["verify"]

        # Detect largest face
        face = engine.detect_largest(image)
        if face is None:
            return {
                "name": angle_name,
                "success": False,
                "reason": "No face detected",
                "embedding": None,
                "quality": 0.0,
            }

        if face.embedding is None or len(face.embedding) == 0:
            return {
                "name": angle_name,
                "success": False,
                "reason": "Failed to extract embedding",
                "embedding": None,
                "quality": 0.0,
            }

        # Get metrics including pose info
        metrics = engine.get_face_metrics(image, face)
        blur_score = metrics["blur_score"]
        nose_x_disp = metrics["nose_x_disp"]

        # Quality check: blur
        if blur_score < config.ENROLL_V2_BLUR_MIN:
            return {
                "name": angle_name,
                "success": False,
                "reason": f"Image too blurry (blur={blur_score:.0f}, min={config.ENROLL_V2_BLUR_MIN})",
                "embedding": None,
                "quality": blur_score,
            }

        # Pose verification using nose_x displacement
        pose_ok, pose_reason = self._verify_pose(nose_x_disp, verify_type)
        if not pose_ok:
            return {
                "name": angle_name,
                "success": False,
                "reason": pose_reason,
                "embedding": None,
                "quality": blur_score,
            }

        # Normalize embedding
        emb = face.embedding.copy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return {
            "name": angle_name,
            "success": True,
            "reason": "",
            "embedding": emb,
            "quality": blur_score,
            "blur_score": blur_score,
            "best_aligned": face.aligned_face,
            "nose_x_disp": nose_x_disp,
        }

    def _verify_pose(self, nose_x_disp: float, verify_type: str) -> tuple[bool, str]:
        """Verify head pose matches the expected angle.

        Uses InsightFace 5-point landmarks (nose_x displacement
        relative to eye center) instead of MediaPipe.

        Args:
            nose_x_disp: Normalized nose X displacement from eye center
                         positive = nose shifted right, negative = shifted left
            verify_type: "center", "left", or "right"

        Returns:
            (passed, reason)
        """
        front_max = config.ENROLL_V2_POSE_FRONT_MAX_DISP  # 0.12
        turn_min = config.ENROLL_V2_POSE_TURN_THRESHOLD   # 0.04

        if verify_type == "center":
            if abs(nose_x_disp) > front_max:
                return False, (
                    f"Face not centered (disp={nose_x_disp:.3f}, "
                    f"max={front_max})"
                )
            return True, ""

        elif verify_type == "left":
            # Looking left: nose shifts left relative to eye center → negative disp
            if nose_x_disp > -turn_min:
                return False, (
                    f"Face not turned left enough (disp={nose_x_disp:.3f}, "
                    f"need < -{turn_min})"
                )
            return True, ""

        elif verify_type == "right":
            # Looking right: nose shifts right → positive disp
            if nose_x_disp < turn_min:
                return False, (
                    f"Face not turned right enough (disp={nose_x_disp:.3f}, "
                    f"need > {turn_min})"
                )
            return True, ""

        return True, ""


# ── Singleton ───────────────────────────────────────────────

_enrollment_v2_service = None


def get_enrollment_v2_service() -> EnrollmentV2Service:
    """Get or create the singleton EnrollmentV2Service."""
    global _enrollment_v2_service
    if _enrollment_v2_service is None:
        _enrollment_v2_service = EnrollmentV2Service()
    return _enrollment_v2_service
