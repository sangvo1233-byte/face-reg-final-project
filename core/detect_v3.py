"""
Detect V3 service: strict face recognition with moire/liveness protection.

Pipeline:
  detect face -> moire score -> passive liveness -> identity match
  -> suspicious frames require active challenge
  -> clean frames are recorded immediately
"""
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

import config
from core.anti_spoof import get_anti_spoof
from core.challenge_v3 import get_challenge_v3_service
from core.database import get_db
from core.face_engine import get_engine
from core.moire import get_moire_detector


class DetectV3Service:
    """Stateless single-frame scan service.

    The only stateful part is delegated to ChallengeV3Service when a scan is
    suspicious enough to require user action.
    """

    def scan_attendance(self, frame: np.ndarray, session_id: int) -> dict:
        engine = get_engine()
        engine._ensure_model()

        faces = engine.detect(frame)
        results = []
        moire_detector = get_moire_detector()
        db = get_db()
        session = db.get_session(session_id)

        for face in faces:
            if face.embedding is None or len(face.embedding) == 0:
                continue

            bbox = self._bbox_list(face.bbox)
            x1, y1, x2, y2 = bbox
            face_roi = frame[max(0, y1):y2, max(0, x1):x2]

            moire_result = moire_detector.analyze_single(face_roi)
            moire_score = float(moire_result.get("moire_score", 1.0))
            moire_decision = self._moire_decision(moire_score, moire_result)
            if moire_decision["action"] == "block":
                logger.warning(
                    "Detect V3 moire blocked frame: "
                    f"score={moire_score:.3f}, "
                    f"block={config.DETECT_V3_MOIRE_BLOCK_THRESHOLD:.3f}, "
                    f"challenge={config.DETECT_V3_MOIRE_CHALLENGE_THRESHOLD:.3f}, "
                    f"peak_ratio={moire_result.get('peak_ratio')}, "
                    f"periodicity={moire_result.get('periodicity')}, "
                    f"grid_score={moire_result.get('grid_score')}"
                )
                results.append(self._spoof_result(
                    bbox=bbox,
                    message=f"Screen detected (moire score: {moire_score:.0%})",
                    moire_score=moire_score,
                    moire_is_screen=True,
                    liveness_score=None,
                ))
                continue

            anti_spoof = get_anti_spoof()
            liveness = anti_spoof.check(frame, face.bbox)
            liveness_score = float(liveness.score)
            liveness_decision = self._liveness_decision(liveness_score, liveness.reason)
            if liveness_decision["action"] == "block":
                results.append(self._spoof_result(
                    bbox=bbox,
                    message=f"Liveness check failed ({liveness.reason})",
                    moire_score=moire_score,
                    moire_is_screen=False,
                    liveness_score=liveness_score,
                ))
                continue

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
                    "bbox": bbox,
                    "moire_score": moire_score,
                    "moire_is_screen": False,
                    "liveness_score": liveness_score,
                    "challenge_required": False,
                })
                continue

            student = db.get_student(match.student_id) or {}
            emb_count = db.get_embedding_count(match.student_id)
            candidate = {
                "name": match.name,
                "student_id": match.student_id,
                "class_name": student.get("class_name", ""),
                "confidence": match.score,
                "bbox": bbox,
                "embedding_count": emb_count,
                "enroll_type": "multi_angle_v2" if emb_count >= 3 else "single",
            }

            suspicion = self._challenge_reason(moire_decision, liveness_decision)
            if suspicion and config.DETECT_V3_CHALLENGE_ENABLED:
                results.append(self._challenge_required_result(
                    session_id=session_id,
                    session=session,
                    candidate=candidate,
                    bbox=bbox,
                    reason=suspicion,
                    moire_score=moire_score,
                    moire_result=moire_result,
                    liveness_score=liveness_score,
                    liveness_reason=liveness.reason,
                ))
                continue

            results.append(self._record_attendance(
                frame=frame,
                session_id=session_id,
                session=session,
                match=match,
                student=student,
                emb_count=emb_count,
                bbox=bbox,
                moire_score=moire_score,
                liveness_score=liveness_score,
            ))

        return {
            "faces_detected": len(faces),
            "recognized": sum(
                1 for result in results
                if result["status"] in ("present", "already")
            ),
            "results": results,
            "scan_version": "v3",
            "threshold": config.DETECT_V3_COSINE_THRESHOLD,
        }

    def _record_attendance(
        self,
        *,
        frame: np.ndarray,
        session_id: int,
        session: dict | None,
        match: Any,
        student: dict,
        emb_count: int,
        bbox: list[int],
        moire_score: float,
        liveness_score: float,
    ) -> dict[str, Any]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        evidence = str(config.EVIDENCE_DIR / f"{match.student_id}_{ts}.jpg")
        cv2.imwrite(evidence, frame)

        db = get_db()
        db_result = db.mark_attendance(
            session_id, match.student_id, match.score, evidence
        )
        evidence_filename = Path(evidence).name

        return {
            "name": match.name,
            "student_id": match.student_id,
            "class_name": student.get("class_name", ""),
            "session_id": session_id,
            "session_name": (
                session.get("name", f"Session #{session_id}")
                if session else f"Session #{session_id}"
            ),
            "confidence": match.score,
            "status": "present" if db_result["success"] else "already",
            "message": db_result["message"],
            "bbox": bbox,
            "evidence_url": f"/api/evidence/{evidence_filename}",
            "moire_score": moire_score,
            "moire_is_screen": False,
            "liveness_score": liveness_score,
            "embedding_count": emb_count,
            "enroll_type": "multi_angle_v2" if emb_count >= 3 else "single",
            "challenge_required": False,
        }

    def record_attendance_result(
        self,
        *,
        frame: np.ndarray,
        session_id: int,
        session: dict | None,
        match: Any,
        student: dict,
        emb_count: int,
        bbox: list[int],
        moire_score: float,
        liveness_score: float,
    ) -> dict[str, Any]:
        """Public wrapper used by the WebSocket stream pipeline."""
        return self._record_attendance(
            frame=frame,
            session_id=session_id,
            session=session,
            match=match,
            student=student,
            emb_count=emb_count,
            bbox=bbox,
            moire_score=moire_score,
            liveness_score=liveness_score,
        )

    def challenge_required_result(
        self,
        *,
        session_id: int,
        session: dict | None,
        candidate: dict[str, Any],
        bbox: list[int],
        reason: str,
        moire_score: float,
        moire_result: dict,
        liveness_score: float,
        liveness_reason: str,
    ) -> dict[str, Any]:
        """Public wrapper used by the WebSocket stream pipeline."""
        return self._challenge_required_result(
            session_id=session_id,
            session=session,
            candidate=candidate,
            bbox=bbox,
            reason=reason,
            moire_score=moire_score,
            moire_result=moire_result,
            liveness_score=liveness_score,
            liveness_reason=liveness_reason,
        )

    def spoof_result(
        self,
        *,
        bbox: list[int],
        message: str,
        moire_score: float,
        moire_is_screen: bool,
        liveness_score: float | None,
    ) -> dict[str, Any]:
        """Public wrapper used by the WebSocket stream pipeline."""
        return self._spoof_result(
            bbox=bbox,
            message=message,
            moire_score=moire_score,
            moire_is_screen=moire_is_screen,
            liveness_score=liveness_score,
        )

    def unknown_result(
        self,
        *,
        bbox: list[int],
        match: Any,
        moire_score: float,
        liveness_score: float,
    ) -> dict[str, Any]:
        return {
            "name": "Unknown",
            "student_id": "",
            "confidence": match.score,
            "status": "unknown",
            "message": (
                f"No match (score={match.score:.3f}, "
                f"threshold={config.DETECT_V3_COSINE_THRESHOLD})"
            ),
            "bbox": bbox,
            "moire_score": moire_score,
            "moire_is_screen": False,
            "liveness_score": liveness_score,
            "challenge_required": False,
        }

    def candidate_from_match(
        self,
        *,
        match: Any,
        student: dict,
        emb_count: int,
        bbox: list[int],
    ) -> dict[str, Any]:
        return {
            "name": match.name,
            "student_id": match.student_id,
            "class_name": student.get("class_name", ""),
            "confidence": match.score,
            "bbox": bbox,
            "embedding_count": emb_count,
            "enroll_type": "multi_angle_v2" if emb_count >= 3 else "single",
        }

    def moire_decision(self, score: float, result: dict) -> dict[str, str]:
        return self._moire_decision(score, result)

    def liveness_decision(self, score: float, reason: str) -> dict[str, str]:
        return self._liveness_decision(score, reason)

    def challenge_reason(self, *decisions: dict[str, str]) -> str:
        return self._challenge_reason(*decisions)

    def bbox_list(self, bbox: Any) -> list[int]:
        return self._bbox_list(bbox)

    def _challenge_required_result(
        self,
        *,
        session_id: int,
        session: dict | None,
        candidate: dict[str, Any],
        bbox: list[int],
        reason: str,
        moire_score: float,
        moire_result: dict,
        liveness_score: float,
        liveness_reason: str,
    ) -> dict[str, Any]:
        challenge = get_challenge_v3_service().create_challenge(
            session_id=session_id,
            candidate=candidate,
            reason=reason,
            moire_score=moire_score,
            liveness_score=liveness_score,
            diagnostics={
                "moire": moire_result,
                "liveness_reason": liveness_reason,
            },
        )

        logger.info(
            "Detect V3 challenge required: "
            f"student={candidate['student_id']}, reason={reason}, "
            f"moire={moire_score:.3f}, liveness={liveness_score:.3f}, "
            f"type={challenge['type']}"
        )

        return {
            "name": candidate["name"],
            "student_id": candidate["student_id"],
            "class_name": candidate.get("class_name", ""),
            "session_id": session_id,
            "session_name": (
                session.get("name", f"Session #{session_id}")
                if session else f"Session #{session_id}"
            ),
            "confidence": candidate["confidence"],
            "status": "challenge_required",
            "message": f"Verification required: {reason}",
            "bbox": bbox,
            "moire_score": moire_score,
            "moire_is_screen": False,
            "liveness_score": liveness_score,
            "liveness_reason": liveness_reason,
            "embedding_count": candidate.get("embedding_count", 0),
            "enroll_type": candidate.get("enroll_type", "single"),
            "challenge_required": True,
            "challenge": challenge,
        }

    def _spoof_result(
        self,
        *,
        bbox: list[int],
        message: str,
        moire_score: float,
        moire_is_screen: bool,
        liveness_score: float | None,
    ) -> dict[str, Any]:
        return {
            "name": "Unknown",
            "student_id": "",
            "confidence": 0,
            "status": "spoof",
            "message": message,
            "bbox": bbox,
            "moire_score": moire_score,
            "moire_is_screen": moire_is_screen,
            "liveness_score": liveness_score,
            "challenge_required": False,
        }

    def _moire_decision(self, score: float, result: dict) -> dict[str, str]:
        if score < config.DETECT_V3_MOIRE_BLOCK_THRESHOLD:
            return {"action": "block", "reason": "screen detected"}
        if score < config.DETECT_V3_MOIRE_CHALLENGE_THRESHOLD:
            return {"action": "challenge", "reason": "moire pattern suspicious"}
        if result.get("is_screen") is True:
            return {"action": "challenge", "reason": "moire detector suspicious"}
        return {"action": "pass", "reason": ""}

    def _liveness_decision(self, score: float, reason: str) -> dict[str, str]:
        if score < config.DETECT_V3_LIVENESS_BLOCK_THRESHOLD:
            return {"action": "block", "reason": reason or "liveness failed"}
        if score < config.DETECT_V3_LIVENESS_CHALLENGE_THRESHOLD:
            return {"action": "challenge", "reason": reason or "liveness suspicious"}
        return {"action": "pass", "reason": ""}

    def _challenge_reason(self, *decisions: dict[str, str]) -> str:
        reasons = [
            decision["reason"]
            for decision in decisions
            if decision.get("action") == "challenge"
        ]
        return "; ".join(reasons)

    def _bbox_list(self, bbox: Any) -> list[int]:
        return [int(value) for value in bbox[:4]]


_detect_v3_service: DetectV3Service | None = None


def get_detect_v3_service() -> DetectV3Service:
    global _detect_v3_service
    if _detect_v3_service is None:
        _detect_v3_service = DetectV3Service()
    return _detect_v3_service
