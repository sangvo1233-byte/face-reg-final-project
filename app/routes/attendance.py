"""
Attendance Routes — /api/session/*, /api/scan

Manages attendance sessions and face scan endpoint.
"""
import cv2
import numpy as np
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from loguru import logger

from core.face_engine import get_engine
from core.database import get_db

router = APIRouter(tags=["attendance"])


class SessionCreate(BaseModel):
    name: str
    class_name: str = ''


# ── Sessions ────────────────────────────────────────────────

@router.post("/api/session/start")
async def start_session(data: SessionCreate):
    """Start a new attendance session."""
    db = get_db()
    # Reject if a session is already active
    active = db.get_active_session()
    if active:
        return {'success': False, 'message': f'Session #{active["id"]} is already open'}
    session_id = db.create_session(data.name, data.class_name)
    return {'success': True, 'session_id': session_id, 'message': f'Session "{data.name}" created'}


@router.post("/api/session/end")
async def end_session():
    """Kết thúc phiên điểm danh hiện tại."""
    db = get_db()
    active = db.get_active_session()
    if not active:
        return {'success': False, 'message': 'Không có phiên nào đang mở'}
    result = db.end_session(active['id'])
    return {
        'success': True,
        'session_id': active['id'],
        'message': f'Kết thúc phiên: {result["present"]}/{result["total"]} có mặt',
        **result
    }


@router.get("/api/session/active")
async def get_active_session():
    """Lấy phiên đang active."""
    db = get_db()
    session = db.get_active_session()
    if not session:
        return {'active': False}
    return {'active': True, 'session': session}


@router.get("/api/session/{session_id}/result")
async def get_session_result(session_id: int):
    """Kết quả điểm danh của 1 phiên."""
    db = get_db()
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(404, "Phiên không tồn tại")
    result = db.get_session_result(session_id)
    attendance = db.get_session_attendance(session_id)
    return {'session': session, 'result': result, 'attendance': attendance}


@router.get("/api/sessions")
async def list_sessions(limit: int = Query(50)):
    """Danh sách phiên điểm danh."""
    db = get_db()
    sessions = db.get_all_sessions(limit)
    return {'sessions': sessions}


# ── Scanning ────────────────────────────────────────────────

@router.post("/api/scan")
async def scan_face(image: UploadFile = File(...)):
    """Quét ảnh → nhận diện → ghi điểm danh vào phiên active."""
    db = get_db()
    active = db.get_active_session()
    if not active:
        return {'success': False, 'error': 'Chưa bắt đầu phiên điểm danh'}

    contents = await image.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Ảnh không hợp lệ")

    engine = get_engine()
    result = engine.scan_attendance(frame, active['id'])
    return result


@router.get("/api/session/attendance")
async def get_current_attendance():
    """Danh sách đã điểm danh trong phiên hiện tại."""
    db = get_db()
    active = db.get_active_session()
    if not active:
        return {'active': False, 'attendance': []}
    attendance = db.get_session_attendance(active['id'])
    result = db.get_session_result(active['id'])
    return {'active': True, 'session': active, 'attendance': attendance, 'result': result}
