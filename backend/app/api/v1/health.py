"""GET /api/v1/health (Section 12)."""
from __future__ import annotations

from fastapi import APIRouter, Request

from backend.app.core.config import get_settings
from backend.app.services.inference_service import get_inference_service

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict:
    settings = get_settings()
    service = get_inference_service(settings)
    model_loaded = service.state.loaded
    storage_ok = settings.data_root.exists()
    return {
        "status": "ok" if (model_loaded or service.state.error) and storage_ok else "degraded",
        "model": {"version": service.state.model_version, "loaded": model_loaded},
        "model_note": service.state.error,
        "storage_ok": storage_ok,
        "app_version": settings.app_version,
    }
