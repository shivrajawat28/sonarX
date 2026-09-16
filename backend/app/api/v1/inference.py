"""POST /detections/run (Section 12): sync single-image inference."""
from __future__ import annotations

import math
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, field_validator

from backend.app.core.config import get_settings
from backend.app.services.inference_service import get_inference_service

router = APIRouter(prefix="/detections", tags=["inference"])


class DetectionOverrides(BaseModel):
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("confidence_threshold", mode="after")
    @classmethod
    def validate_finite(cls, v: float | None) -> float | None:
        if v is not None and (math.isnan(v) or math.isinf(v)):
            raise ValueError("confidence_threshold must be a finite float between 0.0 and 1.0")
        return v


class RunRequest(BaseModel):
    image_id: str
    model_version: str | None = None
    overrides: DetectionOverrides | None = Field(default=None, description="{confidence_threshold?: float}")
    save: bool = False


@router.post("/run", status_code=201)
def run_detection(request: Request, body: RunRequest) -> dict:
    settings = get_settings()
    service = get_inference_service(settings)
    service.check_model_available()  # degraded mode fast-fail (Section 11.4)
    override = body.overrides.confidence_threshold if body.overrides else None
    payload = service.run_for_image(
        body.image_id,
        model_version=body.model_version,
        confidence_override=override,
        save=body.save,
    )
    payload["request_id"] = getattr(request.state, "request_id", "")
    return payload
