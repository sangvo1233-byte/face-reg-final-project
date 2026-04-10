"""
Face Attendance System — Điểm Danh Học Sinh — Entry Point
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

import config
from app.routes import router

app = FastAPI(title="Điểm Danh Học Sinh", version="1.0")
app.mount("/static", StaticFiles(directory=str(config.WEB_DIR)), name="static")
app.include_router(router)


@app.get("/")
async def root():
    return FileResponse(str(config.WEB_DIR / "index.html"))


@app.get("/phone")
async def phone_page():
    return FileResponse(str(config.WEB_DIR / "phone.html"))


@app.on_event("startup")
async def startup():
    logger.info("Preloading models...")
    try:
        from core.face_engine import get_engine
        get_engine()._ensure_model()
        logger.info("Models preloaded successfully")
    except Exception as e:
        logger.warning(f"Model preload failed: {e}")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Face Attendance — Điểm Danh Học Sinh")
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True,
                reload_dirs=[str(config.BASE_DIR)])
