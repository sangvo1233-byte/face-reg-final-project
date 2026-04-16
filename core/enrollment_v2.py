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
            frames = self._frames_for_angle(images[angle])

            result = self._process_angle_frames(engine, frames, phase)
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
                        "frame_count": r.get("frame_count", 0),
                        "frames_required": r.get("frames_required", 1),
                    }
                    for r in phase_results
                ],
                "total_saved": 0,
                "embeddings_saved": 0,
            }

        # ── Save to database ────────────────────────────
        best_aligned = None
        best_blur = 0.0
        photo_path = None
        embeddings_to_save = []

        for pr in successful:
            source = f"v2_{pr['name'].lower()}"
            quality = pr.get("quality", 0.0)
            embeddings_to_save.append({
                "embedding": pr["embedding"],
                "quality": quality,
                "source": source,
            })
            logger.info(
                f"  Prepared: {source} (quality={quality:.0f}, "
                f"frames={pr.get('frame_count', 1)})"
            )

            # Track best aligned face for photo
            if pr.get("blur_score", 0) > best_blur and pr.get("best_aligned") is not None:
                best_blur = pr["blur_score"]
                best_aligned = pr["best_aligned"]

        # Save face crop photo
        if best_aligned is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            photo_path = str(config.FACE_CROPS_DIR / f"{student_id}_{ts}.jpg")
            if not cv2.imwrite(photo_path, best_aligned):
                logger.warning(f"Failed to write enrollment photo for {student_id}")
                photo_path = None

        # Atomically update metadata and replace embeddings only after the
        # complete new embedding set has been validated and prepared.
        db = get_db()
        save_result = db.replace_student_embeddings(
            student_id=student_id,
            name=name,
            class_name=class_name,
            embeddings=embeddings_to_save,
            photo_path=photo_path,
        )
        saved = save_result["saved"]
        old_count = save_result["old_count"]
        if old_count > 0:
            logger.info(f"Replaced {old_count} old embedding(s) for {student_id}")

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
                    "frame_count": r.get("frame_count", 0),
                    "frames_required": r.get("frames_required", 1),
                }
                for r in phase_results
            ],
            "total_saved": saved,
            "embeddings_saved": saved,
        }

    def _frames_for_angle(self, value) -> list[np.ndarray]:
        """Normalize one image or a list of images into a frame list."""
        if isinstance(value, (list, tuple)):
            return [frame for frame in value if frame is not None]
        return [value] if value is not None else []

    def _process_angle_frames(self, engine, frames: list[np.ndarray], phase: dict) -> dict:
        """Validate multiple frames for one angle and average good embeddings."""
        angle_name = phase["name"]
        if not frames:
            return {
                "name": angle_name,
                "success": False,
                "reason": "No frames provided",
                "embedding": None,
                "quality": 0.0,
                "frame_count": 0,
                "frames_required": 1,
            }

        max_frames = max(1, config.ENROLL_V2_MAX_FRAMES_PER_PHASE)
        frames = frames[:max_frames]
        required = (
            min(config.ENROLL_V2_RECOMMENDED_FRAMES, len(frames))
            if len(frames) > 1 else config.ENROLL_V2_MIN_FRAMES_PER_PHASE
        )

        frame_results = [
            self._process_angle(engine, frame, phase)
            for frame in frames
        ]
        successful = [r for r in frame_results if r["success"]]

        if len(successful) < required:
            first_failure = next((r for r in frame_results if not r["success"]), None)
            reason = first_failure.get("reason", "Not enough valid frames") if first_failure else "Not enough valid frames"
            return {
                "name": angle_name,
                "success": False,
                "reason": f"{reason} ({len(successful)}/{required} good frames)",
                "embedding": None,
                "quality": 0.0,
                "frame_count": len(successful),
                "frames_required": required,
            }

        embeddings = np.stack([r["embedding"] for r in successful])
        emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        best = max(successful, key=lambda r: r.get("blur_score", 0))
        quality = float(np.mean([r.get("quality", 0.0) for r in successful]))

        return {
            "name": angle_name,
            "success": True,
            "reason": "",
            "embedding": emb,
            "quality": quality,
            "blur_score": best.get("blur_score", quality),
            "best_aligned": best.get("best_aligned"),
            "nose_x_disp": best.get("nose_x_disp", 0.0),
            "frame_count": len(successful),
            "frames_required": required,
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
            nose_x_disp: Normalized nose X displacement from eye center.
                         Positive maps to the user's LEFT turn in the
                         enrollment camera feed; negative maps to RIGHT.
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
            # User-facing LEFT in the enrollment camera feed.
            if nose_x_disp < turn_min:
                return False, (
                    f"Face not turned left enough (disp={nose_x_disp:.3f}, "
                    f"need > {turn_min})"
                )
            return True, ""

        elif verify_type == "right":
            # User-facing RIGHT in the enrollment camera feed.
            if nose_x_disp > -turn_min:
                return False, (
                    f"Face not turned right enough (disp={nose_x_disp:.3f}, "
                    f"need < -{turn_min})"
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
