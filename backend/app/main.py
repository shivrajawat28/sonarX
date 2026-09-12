"""FastAPI app factory (Section 11): thin bridge; all AI logic lives in mlpipeline."""
from __future__ import annotations

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

    # CORS: explicit allowlist, never a wildcard (Section 20)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["x-request-id"],
    )

    @app.middleware("http")
    async def _request_id(request: Request, call_next):
        return await request_id_middleware(request, call_next)

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

    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
