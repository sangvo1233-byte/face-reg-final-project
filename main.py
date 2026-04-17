"""
Face Attendance System - product entrypoint.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

import config
from app.routes import router


STARTED_AT = time.time()


def _status(status: str, message: str = "") -> dict[str, Any]:
    return {"status": status, "message": message}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.startup_status = {
        "model": _status("pending"),
        "camera": _status("pending"),
    }

    if config.AUTO_PRELOAD_MODELS:
        logger.info("Preloading face engine...")
        try:
            from core.face_engine import get_engine

            get_engine().warmup(load_embeddings=config.AUTO_LOAD_EMBEDDING_CACHE)
            app.state.startup_status["model"] = _status("ready", "Face engine preloaded")
            logger.info("Face engine preloaded successfully")
        except Exception as exc:
            app.state.startup_status["model"] = _status("error", str(exc))
            logger.warning(f"Face engine preload failed: {exc}")
    else:
        app.state.startup_status["model"] = _status("skipped", "AUTO_PRELOAD_MODELS is disabled")
        logger.info("Face engine preload skipped")

    if config.AUTO_START_CAMERA:
        logger.info("Starting camera service...")
        try:
            from core.camera import get_camera

            get_camera().start()
            app.state.startup_status["camera"] = _status("ready", "Camera service started")
            logger.info("Camera service started")
        except Exception as exc:
            status = "error" if config.CAMERA_REQUIRED else "degraded"
            app.state.startup_status["camera"] = _status(status, str(exc))
            logger.warning(f"Camera service start failed: {exc}")
    else:
        app.state.startup_status["camera"] = _status("skipped", "AUTO_START_CAMERA is disabled")
        logger.info("Camera service startup skipped")

    yield

    for name, factory in (
        ("local runner V3", "core.local_runner:get_local_runner"),
        ("local runner V4", "core.local_runner_v4:get_local_runner_v4"),
    ):
        try:
            module_name, func_name = factory.split(":")
            module = __import__(module_name, fromlist=[func_name])
            getattr(module, func_name)().stop()
            logger.info(f"Stopped {name}")
        except Exception as exc:
            logger.debug(f"Stop {name} skipped/failed: {exc}")

    try:
        from core.camera import get_camera

        get_camera().stop()
        logger.info("Camera service stopped")
    except Exception as exc:
        logger.debug(f"Camera service stop skipped/failed: {exc}")


app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if config.API_DOCS_ENABLED else None,
    redoc_url="/redoc" if config.API_DOCS_ENABLED else None,
    openapi_url="/openapi.json" if config.API_DOCS_ENABLED else None,
)

if config.TRUSTED_HOSTS:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.TRUSTED_HOSTS)

if config.CORS_ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.mount("/static", StaticFiles(directory=str(config.WEB_DIR)), name="static")
app.include_router(router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "app": config.APP_NAME,
        "version": config.APP_VERSION,
        "environment": config.APP_ENV,
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
    }


@app.get("/ready")
async def ready():
    checks = dict(getattr(app.state, "startup_status", {}))

    try:
        from core.database import get_db

        checks["database"] = {
            "status": "ready",
            "message": "SQLite database reachable",
            "active_students": get_db().get_student_count(),
            "active_session": bool(get_db().get_active_session()),
        }
    except Exception as exc:
        checks["database"] = _status("error", str(exc))

    ready_states = {"ready", "skipped", "degraded"}
    is_ready = all(check.get("status") in ready_states for check in checks.values())
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={
            "status": "ready" if is_ready else "not_ready",
            "checks": checks,
        },
    )


@app.get("/version")
async def version():
    return {
        "app": config.APP_NAME,
        "version": config.APP_VERSION,
        "environment": config.APP_ENV,
        "detect_runtime": "v4.4",
        "scan_versions": ["v3", "v4.4"],
        "features": {
            "browser_stream_v4": True,
            "local_direct_v4": True,
            "tech_overlay": True,
            "auto_preload_models": config.AUTO_PRELOAD_MODELS,
            "auto_start_camera": config.AUTO_START_CAMERA,
            "camera_required": config.CAMERA_REQUIRED,
        },
    }


@app.get("/")
async def root():
    return FileResponse(str(config.WEB_DIR / "index.html"))


@app.get("/phone")
async def phone_page():
    return FileResponse(str(config.WEB_DIR / "phone.html"))


if __name__ == "__main__":
    import uvicorn

    logger.info(
        f"Starting {config.APP_NAME} {config.APP_VERSION} "
        f"env={config.APP_ENV} reload={config.UVICORN_RELOAD}"
    )
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.UVICORN_RELOAD,
        reload_dirs=[str(config.BASE_DIR)] if config.UVICORN_RELOAD else None,
    )
