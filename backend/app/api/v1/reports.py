"""Report endpoints (Section 12): async generation job + PDF download."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import Response
from pydantic import BaseModel

from backend.app.core.config import get_settings
from backend.app.core.errors import NotFoundError, ReportGenerationFailedError
from backend.app.services.inference_service import get_inference_service

router = APIRouter(prefix="/reports", tags=["reports"])


class ReportRequest(BaseModel):
    survey_id: str | None = None
    run_id: str | None = None
    format: str = "pdf"


@router.post("", status_code=202)
def create_report(request: Request, body: ReportRequest, bg: BackgroundTasks) -> dict:
    settings = get_settings()
    service = get_inference_service(settings)
    if not body.survey_id and not body.run_id:
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail="survey_id or run_id required")
    subject = body.survey_id or body.run_id
    if body.survey_id and service.repo.get("surveys", body.survey_id) is None:
        raise NotFoundError(f"survey '{body.survey_id}' not found")
    if body.run_id and service.repo.get("detection_runs", body.run_id) is None:
        raise NotFoundError(f"run '{body.run_id}' not found")

    from backend.app.services.job_service import job_service_with_bg

    jobs = job_service_with_bg(settings, bg)
    job = jobs.create(
        job_type="report",
        payload={"survey_id": body.survey_id, "run_id": body.run_id, "format": body.format},
    )
    jobs.submit_report(job["job_id"], service)
    return {"job_id": job["job_id"], "status": job["status"]}


@router.get("/{report_id}")
def download_report(request: Request, report_id: str, format: str = "html") -> Response:
    settings = get_settings()
    service = get_inference_service(settings)
    report = service.repo.get("reports", report_id)
    if report is None:
        raise NotFoundError(f"report '{report_id}' not found")
    fmt = (format or "html").lower()
    key = "artifact_path_pdf" if fmt == "pdf" else "artifact_path"
    artifact = report.get(key)
    if fmt == "pdf" and not artifact:
        raise NotFoundError(
            "no PDF artifact for this report (generated before dual-format support) — regenerate the report"
        )
    if not artifact:
        raise ReportGenerationFailedError("report record exists but artifact missing")
    from backend.app.core.security import resolve_within

    path = resolve_within(settings.data_root, artifact)
    if not path.is_file():
        raise NotFoundError("report artifact file missing")
    if fmt == "pdf":
        media_type = "application/pdf"
    else:
        media_type = "text/html"
    return Response(
        content=path.read_bytes(),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={path.name}"},
    )
