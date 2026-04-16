"""
Detect V3 Service — Face Recognition with Moiré Anti-Spoof.

Combines:
  1. Moiré Pattern Detection (passive FFT anti-spoof)
  2. Stricter cosine threshold (0.52 vs default 0.45)
  3. Existing liveness checks

Fully compatible with Enroll V2 multi-angle embeddings.

Usage:
    service = DetectV3Service()
    result = service.scan_attendance(frame, session_id)
"""
import cv2
import numpy as np
from datetime import datetime
from loguru import logger

import config
from core.face_engine import get_engine
from core.database import get_db
from core.anti_spoof import get_anti_spoof
from core.moire import get_moire_detector


class DetectV3Service:
    """Stateless recognition service with moiré anti-spoof.

    Single-frame scan: detect → moiré check → liveness → match → record.
    """

    def scan_attendance(self, frame: np.ndarray, session_id: int) -> dict:
        """Scan a single frame with V3 pipeline.

        Returns:
            dict: {
                faces_detected, recognized,
                results: [{name, student_id, confidence, status,
                           message, bbox, moire_score, moire_is_screen}]
            }
        """
        engine = get_engine()
        engine._ensure_model()

        faces = engine.detect(frame)
        results = []
        moire_detector = get_moire_detector()

        for face in faces:
            if face.embedding is None or len(face.embedding) == 0:
                continue

            bbox = face.bbox
            x1, y1, x2, y2 = (
                int(bbox[0]), int(bbox[1]),
                int(bbox[2]), int(bbox[3]),
            )

            # ── Layer 1: Moiré check (passive anti-spoof) ──
            face_roi = frame[max(0, y1):y2, max(0, x1):x2]
            moire_result = moire_detector.analyze_single(face_roi)
            moire_score = moire_result.get("moire_score", 1.0)
            is_screen = moire_result.get("is_screen", False)

            if is_screen:
                results.append({
                    "name": "Unknown",
                    "student_id": "",
                    "confidence": 0,
                    "status": "spoof",
                    "message": f"Screen detected (moiré score: {moire_score:.0%})",
                    "bbox": [x1, y1, x2, y2],
                    "moire_score": moire_score,
                    "moire_is_screen": True,
                })
                continue

            # ── Layer 2: Existing liveness check ───────────
            anti_spoof = get_anti_spoof()
            liveness = anti_spoof.check(frame, face.bbox)
            if not liveness.is_live:
                results.append({
                    "name": "Unknown",
                    "student_id": "",
                    "confidence": 0,
                    "status": "spoof",
                    "message": f"Liveness check failed ({liveness.reason})",
                    "bbox": [x1, y1, x2, y2],
                    "moire_score": moire_score,
                    "moire_is_screen": False,
                })
                continue

            # ── Layer 3: Match with V3 strict threshold ────
            match = engine.match_with_threshold(
                face.embedding,
                config.DETECT_V3_COSINE_THRESHOLD,
            )

            if not match.matched:
                results.append({
                    "name": "Unknown",
                    "student_id": "",
                    "confidence": match.score,
                    "status": "unknown",
                    "message": (
                        f"No match (score={match.score:.3f}, "
                        f"threshold={config.DETECT_V3_COSINE_THRESHOLD})"
                    ),
                    "bbox": [x1, y1, x2, y2],
                    "moire_score": moire_score,
                    "moire_is_screen": False,
                })
                continue

            # ── Record attendance ──────────────────────────
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            evidence = str(config.EVIDENCE_DIR / f"{match.student_id}_{ts}.jpg")
            cv2.imwrite(evidence, frame)

            db = get_db()
            db_result = db.mark_attendance(
                session_id, match.student_id, match.score, evidence
            )

            # Embedding count info
            emb_count = db.get_embedding_count(match.student_id)

            results.append({
                "name": match.name,
                "student_id": match.student_id,
                "confidence": match.score,
                "status": "present" if db_result["success"] else "already",
                "message": db_result["message"],
                "bbox": [x1, y1, x2, y2],
                "moire_score": moire_score,
                "moire_is_screen": False,
                "embedding_count": emb_count,
                "enroll_type": "multi_angle_v2" if emb_count >= 3 else "single",
            })

        return {
            "faces_detected": len(faces),
            "recognized": sum(
                1 for r in results if r["status"] in ("present", "already")
            ),
            "results": results,
            "scan_version": "v3",
            "threshold": config.DETECT_V3_COSINE_THRESHOLD,
        }


# ── Singleton ───────────────────────────────────────────────

_detect_v3_service = None


def get_detect_v3_service() -> DetectV3Service:
    """Get or create the singleton DetectV3Service."""
    global _detect_v3_service
    if _detect_v3_service is None:
        _detect_v3_service = DetectV3Service()
    return _detect_v3_service
