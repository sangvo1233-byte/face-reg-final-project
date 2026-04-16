"""
Enrollment Routes — /api/enroll/*, /api/students/*

Student registration and list management.
"""
import cv2
import numpy as np

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from loguru import logger

from core.face_engine import get_engine
from core.database import get_db

router = APIRouter(tags=["enrollment"])


@router.post("/api/enroll")
async def enroll_student(
    name: str = Form(...),
    student_id: str = Form(...),
    class_name: str = Form(''),
    image: UploadFile = File(...),
):
    """Register a student using one photo."""
    contents = await image.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Invalid image")

    engine = get_engine()
    return engine.enroll_from_photo(student_id, name, frame, class_name)


from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import config

class StudentUpdate(BaseModel):
    name: str | None = None
    class_name: str | None = None

@router.get("/api/students")
async def list_students(view: str = "active"):
    """List students with optional view filter (active, archived, all)."""
    db = get_db()
    if view == "archived":
        students = db.get_all_students(active_only=False, archived_only=True)
    elif view == "all":
        students = db.get_all_students(active_only=False, archived_only=False)
    else:
        students = db.get_all_students(active_only=True, archived_only=False)
    return {'students': students, 'total': len(students)}

@router.get("/api/students/{student_id}")
async def get_student(student_id: str):
    """Get details of a single student (works for active and archived)."""
    db = get_db()
    s = db.get_student_any(student_id)
    if not s:
        raise HTTPException(404, "Student not found")
    emb_count = db.get_embedding_count(student_id)
    history = db.get_student_history(student_id)
    return {'student': s, 'embedding_count': emb_count, 'history': history}

@router.put("/api/students/{student_id}")
async def update_student(student_id: str, payload: StudentUpdate):
    """Update student metadata (name, class_name)."""
    db = get_db()
    s = db.get_student_any(student_id)
    if not s:
        raise HTTPException(404, "Student not found")
    
    updates = {}
    if payload.name is not None:
        updates['name'] = payload.name
    if payload.class_name is not None:
        updates['class_name'] = payload.class_name
        
    if updates:
        db.update_student(student_id, **updates)
        engine = get_engine()
        engine.reload_cache()
    return {'success': True, 'message': 'Student updated'}

@router.post("/api/students/{student_id}/restore")
async def restore_student(student_id: str):
    """Restore a soft-deleted student."""
    db = get_db()
    if db.restore_student(student_id):
        engine = get_engine()
        engine.reload_cache()
        return {'success': True, 'message': f'Student {student_id} restored'}
    raise HTTPException(404, "Student not found or already active")

@router.get("/api/students/{student_id}/photo")
async def get_student_photo(student_id: str):
    """Serve the student's face crop photo."""
    db = get_db()
    s = db.get_student_any(student_id)
    if not s or not s.get('photo_path'):
        raise HTTPException(404, "Photo not found")
    
    path = str(s['photo_path'])
    if not os.path.exists(path):
        raise HTTPException(404, "Photo file missing")
    return FileResponse(path)

@router.delete("/api/students/{student_id}")
async def delete_student(student_id: str):
    """Archive a student (soft delete). Preserves embeddings for restore."""
    db = get_db()
    engine = get_engine()
    success = db.delete_student(student_id)
    if not success:
        raise HTTPException(404, "Student not found")
    # Do NOT delete embeddings — archive must be reversible via restore
    engine.reload_cache()
    return {'success': True, 'message': f'Student {student_id} archived'}
