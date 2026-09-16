"""FastAPI app factory (Section 11): thin bridge; all AI logic lives in mlpipeline."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root and ml/ are on sys.path for direct uvicorn execution
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ML_DIR = _REPO_ROOT / "ml"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.core.config import get_settings
from backend.app.core.errors import APIError, VALIDATION_CODE, api_error_handler, error_envelope
from backend.app.core.logging import request_id_middleware, setup_logging
from backend.app.services.inference_service import get_inference_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings)
    settings.ensure_dirs()
    service = get_inference_service(settings)
    service.startup()  # degraded-mode aware (Section 11.4)
    logger = logging.getLogger("api")
    logger.info(
        "startup complete; model=%s",
        service.state.model_version or "unavailable (degraded mode)",
    )
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Marine Debris Sonar AI",
        version=settings.app_version,
        lifespan=lifespan,
    )

    # CORS: explicit allowlist + Vercel and preview origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_origin_regex=r"^https://.*(\.vercel\.app|\.trycloudflare\.com)$",
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["x-request-id"],
    )

    @app.middleware("http")
    async def _request_id(request: Request, call_next):
        return await request_id_middleware(request, call_next)

    @app.middleware("http")
    async def _security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    app.add_exception_handler(APIError, api_error_handler)

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", "")
        details = {"errors": [{"loc": [str(l) for l in e["loc"]], "msg": e["msg"]} for e in exc.errors()]}
        return JSONResponse(
            status_code=422,
            content=error_envelope(VALIDATION_CODE, "request validation failed", request_id, details),
        )

    from backend.app.api.v1.router import api_router
    from backend.app.api.v1.health import health as health_endpoint

    app.include_router(api_router, prefix="/api/v1")
    app.get("/health", tags=["health"])(health_endpoint)
    return app


app = create_app()
