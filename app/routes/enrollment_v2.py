"""
Enrollment V2 Routes — POST /api/enroll/v2

Multi-angle face enrollment: front + left + right.
"""
import cv2
import numpy as np

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from loguru import logger

from core.enrollment_v2 import get_enrollment_v2_service

router = APIRouter(tags=["enrollment_v2"])


@router.post("/api/enroll/v2")
async def enroll_v2(
    student_id: str = Form(...),
    name: str = Form(...),
    class_name: str = Form(""),
    image_front: UploadFile = File(...),
    image_left: UploadFile = File(...),
    image_right: UploadFile = File(...),
):
    """Enroll a student using 3-angle images (front, left, right).

    Multi-angle enrollment improves recognition accuracy by
    averaging embeddings from different head poses.
    """
    # Decode all three images
    images = {}
    for key, upload in [
        ("front", image_front),
        ("left", image_left),
        ("right", image_right),
    ]:
        contents = await upload.read()
        frame = cv2.imdecode(
            np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR
        )
        if frame is None:
            raise HTTPException(400, f"Invalid image for angle: {key}")
        images[key] = frame

    service = get_enrollment_v2_service()
    result = service.enroll_multi_angle(student_id, name, class_name, images)

    if not result["success"]:
        # Return 200 with success=False (not a server error, just enrollment failure)
        return result

    return result

@router.post("/api/enroll/v2/validate")
async def validate_v2_angle(
    angle: str = Form(...),
    image: UploadFile = File(...)
):
    """Validate a single angle image (front, left, right) for instant feedback."""
    contents = await image.read()
    frame = cv2.imdecode(
        np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR
    )
    if frame is None:
        raise HTTPException(400, "Invalid image")
        
    service = get_enrollment_v2_service()
    
    # We create a dummy phase dict for the _process_angle internal function
    verify_mapping = {"front": "center", "left": "left", "right": "right"}
    if angle.lower() not in verify_mapping:
        raise HTTPException(400, "Invalid angle (must be front, left, or right)")
        
    phase = {
        "name": angle.upper(),
        "verify": verify_mapping[angle.lower()]
    }
    
    # Needs access to face engine
    from core.face_engine import get_engine
    engine = get_engine()
    engine._ensure_model()
    
    result = service._process_angle(engine, frame, phase)
    
    # Scrub embedding from response
    if "embedding" in result:
        del result["embedding"]
    if "best_aligned" in result:
        del result["best_aligned"]
        
    return result
