"""
Face Attendance System — Điểm Danh Học Sinh — Entry Point
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger
from contextlib import asynccontextmanager

import config
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Preloading models...")
    try:
        from core.face_engine import get_engine
        get_engine()._ensure_model()
        logger.info("Models preloaded successfully")
    except Exception as e:
        logger.warning(f"Model preload failed: {e}")

    try:
        from core.camera import get_camera
        get_camera().start()
        logger.info("Camera service started")
    except Exception as e:
        logger.warning(f"Camera service start failed: {e}")

    yield

    try:
        from core.local_runner import get_local_runner
        get_local_runner().stop()
    except Exception:
        pass
    try:
        from core.local_runner_v4 import get_local_runner_v4
        get_local_runner_v4().stop()
    except Exception:
        pass
    try:
        from core.camera import get_camera
        get_camera().stop()
    except Exception:
        pass


app = FastAPI(title="Điểm Danh Học Sinh", version="1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(config.WEB_DIR)), name="static")
app.include_router(router)


@app.get("/")
async def root():
    return FileResponse(str(config.WEB_DIR / "index.html"))


@app.get("/phone")
async def phone_page():
    return FileResponse(str(config.WEB_DIR / "phone.html"))


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Face Attendance — Điểm Danh Học Sinh")
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True,
                reload_dirs=[str(config.BASE_DIR)])
