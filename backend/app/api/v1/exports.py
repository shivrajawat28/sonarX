"""Export endpoints (Section 12): JSON + CSV with a stable column contract."""
from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, Query, Request
from fastapi.responses import Response

from backend.app.core.config import get_settings
from backend.app.services.inference_service import get_inference_service

router = APIRouter(prefix="/exports", tags=["exports"])

CSV_COLUMNS = [
    "detection_id", "run_id", "image_id", "class_name",
    "model_confidence", "final_confidence", "filtering_status", "filter_reasons",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "latitude", "longitude", "geo_status",
    "model_version", "preprocess_config_hash", "created_at",
]


def _fetch_filtered(request: Request, **params) -> list[dict]:
    """Every matching detection — exports must never truncate.

    (bugfix: this used `repo.list(size=200)`, and the repository caps page size
    at 200, so exports of runs with more detections silently dropped rows.)
    """
    settings = get_settings()
    service = get_inference_service(settings)
    filters = {
        "survey_id": params.get("survey_id"),
        "image_id": params.get("image_id"),
        "class_name": params.get("class_name"),
        "filtering_status": params.get("status"),
    }
    items, _ = service.repo.list_all("detections", filters=filters)
    return items


@router.get("/detections.json")
def export_json(
    request: Request,
    survey_id: str | None = None,
    image_id: str | None = None,
    class_name: str | None = Query(None, alias="class"),
    status: str | None = None,
) -> Response:
    items = _fetch_filtered(request, survey_id=survey_id, image_id=image_id,
                            class_name=class_name, status=status)
    payload = {
        "exported_at": _now(),
        "count": len(items),
        "notice": "coordinates are null where navigation metadata was unavailable (never fabricated)",
        "detections": items,
    }
    return Response(
        content=json.dumps(payload, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=detections.json"},
    )


@router.get("/detections.csv")
def export_csv(
    request: Request,
    survey_id: str | None = None,
    image_id: str | None = None,
    class_name: str | None = Query(None, alias="class"),
    status: str | None = None,
) -> Response:
    items = _fetch_filtered(request, survey_id=survey_id, image_id=image_id,
                            class_name=class_name, status=status)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for d in items:
        bbox = d.get("bbox_source_coords") or {}
        reasons = "; ".join(d.get("filter_reasons") or [])
        writer.writerow({
            "detection_id": d.get("detection_id"),
            "run_id": d.get("run_id"),
            "image_id": d.get("image_id"),
            "class_name": d.get("class_name"),
            "model_confidence": d.get("model_confidence"),
            "final_confidence": d.get("final_confidence"),
            "filtering_status": d.get("filtering_status"),
            "filter_reasons": reasons,
            "bbox_x": bbox.get("x"), "bbox_y": bbox.get("y"),
            "bbox_w": bbox.get("w"), "bbox_h": bbox.get("h"),
            "latitude": d.get("latitude"),
            "longitude": d.get("longitude"),
            "geo_status": d.get("geo_status"),
            "model_version": d.get("model_version"),
            "preprocess_config_hash": d.get("preprocess_config_hash"),
            "created_at": d.get("created_at"),
        })
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=detections.csv"},
    )


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")
