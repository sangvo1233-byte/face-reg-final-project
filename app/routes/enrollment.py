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


@router.get("/api/students")
async def list_students():
    """List all students."""
    db = get_db()
    students = db.get_all_students()
    return {'students': students, 'total': len(students)}


@router.get("/api/students/{student_id}")
async def get_student(student_id: str):
    """Get details of a single student."""
    db = get_db()
    s = db.get_student(student_id)
    if not s:
        raise HTTPException(404, "Student not found")
    emb_count = db.get_embedding_count(student_id)
    history = db.get_student_history(student_id)
    return {'student': s, 'embedding_count': emb_count, 'history': history}


@router.delete("/api/students/{student_id}")
async def delete_student(student_id: str):
    """Delete a student (soft delete)."""
    db = get_db()
    engine = get_engine()
    success = db.delete_student(student_id)
    if not success:
        raise HTTPException(404, "Student not found")
    db.delete_embeddings(student_id)
    engine.reload_cache()
    return {'success': True, 'message': f'Student {student_id} removed'}
