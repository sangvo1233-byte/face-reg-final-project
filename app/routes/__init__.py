"""
Routes Package — Aggregate router cho hệ thống chấm công.
"""
from fastapi import APIRouter

from app.routes.attendance import router as attendance_router
from app.routes.enrollment import router as enrollment_router
from app.routes.system import router as system_router
from app.routes.live import router as live_router
from app.routes.phone_camera import router as phone_camera_router
from app.routes.enrollment_v2 import router as enrollment_v2_router
from app.routes.scan_v3 import router as scan_v3_router
from app.routes.scan_v4 import router as scan_v4_router
from app.routes.local_scan import router as local_scan_router

router = APIRouter()

router.include_router(attendance_router)
router.include_router(enrollment_router)
router.include_router(system_router)
router.include_router(live_router)
router.include_router(phone_camera_router)
router.include_router(enrollment_v2_router)
router.include_router(scan_v3_router)
router.include_router(scan_v4_router)
router.include_router(local_scan_router)
