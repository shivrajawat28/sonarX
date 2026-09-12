"""Structured logging + request-ID middleware (Section 16)."""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

from backend.app.core.config import Settings


class JSONFormatter(logging.Formatter):
    """Minimal JSON log formatter: ts, level, logger, message, extras."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "model_version", "image_id", "duration_ms", "status_code", "path"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(settings: Settings) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())
    logging.getLogger("uvicorn.access").handlers = [handler]


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


async def request_id_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Bind a request_id to every request; echo it in responses and logs."""
    request_id = request.headers.get("x-request-id") or new_request_id()
    request.state.request_id = request_id
    logger = logging.getLogger("api.access")
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            "%s %s",
            request.method,
            request.url.path,
            extra={
                "request_id": request_id,
                "duration_ms": duration_ms,
                "path": str(request.url.path),
                "status_code": getattr(locals().get("response", None), "status_code", 500),
            },
        )
    response.headers["x-request-id"] = request_id
    return response
