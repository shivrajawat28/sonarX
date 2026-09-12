"""Survey endpoints + async batch inference trigger (Section 12 / ADR-005)."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Query, Request
from pydantic import BaseModel

from backend.app.core.config import get_settings
from backend.app.core.errors import ModelUnavailableError, NotFoundError
from backend.app.services.inference_service import get_inference_service

router = APIRouter(prefix="/surveys", tags=["surveys"])


class SurveyRunRequest(BaseModel):
    model_version: str | None = None
    save: bool = True


@router.get("")
def list_surveys(request: Request, page: int = Query(1, ge=1), size: int = Query(20, ge=1, le=200)) -> dict:
    settings = get_settings()
    service = get_inference_service(settings)
    items, total = service.repo.list("surveys", page=page, size=size)
    return {"items": items, "total": total, "page": page}


@router.get("/{survey_id}")
def get_survey(request: Request, survey_id: str) -> dict:
    settings = get_settings()
    service = get_inference_service(settings)
    survey = service.repo.get("surveys", survey_id)
    if survey is None:
        raise NotFoundError(f"survey '{survey_id}' not found")
    images, total = service.repo.list("sonar_images", page=1, size=200, filters={"survey_id": survey_id})
    survey["image_records"] = images
    return survey


@router.post("/{survey_id}/run", status_code=202)
def run_survey(request: Request, survey_id: str, body: SurveyRunRequest, bg: BackgroundTasks) -> dict:
    """Async batch: creates a job; the job loops the SAME engine per image (ADR-005)."""
    settings = get_settings()
    service = get_inference_service(settings)

    if not service.state.loaded:
        raise ModelUnavailableError(service.state.error or "no model loaded")

    survey = service.repo.get("surveys", survey_id)
    if survey is None:
        raise NotFoundError(f"survey '{survey_id}' not found")
    if not survey.get("images"):
        raise NotFoundError(f"survey '{survey_id}' has no images")

    from backend.app.services.job_service import job_service_with_bg

    jobs = job_service_with_bg(settings, bg)
    job = jobs.create(
        job_type="survey_batch",
        payload={"survey_id": survey_id, "save": body.save,
                 "model_version": body.model_version or service.state.model_version},
    )
    jobs.submit_batch(job["job_id"], service)
    return {"job_id": job["job_id"], "survey_id": survey_id, "status": job["status"]}
