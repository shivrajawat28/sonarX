"""Detection history/detail/override endpoints (Section 12)."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from backend.app.core.config import get_settings
from backend.app.core.errors import NotFoundError
from backend.app.services.inference_service import get_inference_service

router = APIRouter(prefix="/detections", tags=["detections"])


class AnalystOverride(BaseModel):
    status: str  # accepted | flagged | rejected
    note: str | None = None


@router.get("/runs/{run_id}")
def get_run(request: Request, run_id: str) -> dict:
    settings = get_settings()
    service = get_inference_service(settings)
    run = service.repo.get("detection_runs", run_id)
    if run is None:
        raise NotFoundError(f"run '{run_id}' not found")
    detections, _ = service.repo.list("detections", page=1, size=200, filters={"run_id": run_id})
    run["detections"] = detections
    return run


@router.get("")
def list_detections(
    request: Request,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
    survey_id: str | None = None,
    image_id: str | None = None,
    class_name: str | None = Query(None, alias="class"),
    status: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict:
    settings = get_settings()
    service = get_inference_service(settings)
    filters = {"survey_id": survey_id, "image_id": image_id,
               "class_name": class_name, "filtering_status": status}
    items, total = service.repo.list("detections", page=page, size=size, filters=filters)
    if date_from:
        items = [d for d in items if (d.get("created_at") or "") >= date_from]
    if date_to:
        items = [d for d in items if (d.get("created_at") or "") <= date_to]
    return {"items": items, "total": total, "page": page}


@router.get("/{detection_id}")
def get_detection(request: Request, detection_id: str) -> dict:
    settings = get_settings()
    service = get_inference_service(settings)
    det = service.repo.get("detections", detection_id)
    if det is None:
        raise NotFoundError(f"detection '{detection_id}' not found")
    return det


@router.patch("/{detection_id}")
def override_status(request: Request, detection_id: str, body: AnalystOverride) -> dict:
    settings = get_settings()
    service = get_inference_service(settings)
    if body.status not in ("accepted", "flagged", "rejected"):
        from backend.app.core.errors import APIError

        raise APIError("status must be accepted|flagged|rejected", details={"got": body.status})
    updated = service.repo.update(
        "detections", detection_id,
        {
            "filtering_status": body.status,
            "analyst_note": body.note,
            "analyst_overridden": True,
        },
    )
    if updated is None:
        raise NotFoundError(f"detection '{detection_id}' not found")
    return updated
