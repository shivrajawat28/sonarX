"""GET /jobs/{job_id} (Section 12): poll persisted job status/progress."""
from __future__ import annotations

from fastapi import APIRouter, Request

from backend.app.core.config import get_settings
from backend.app.core.errors import NotFoundError
from backend.app.services.job_service import get_job_service

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}")
def get_job(request: Request, job_id: str) -> dict:
    settings = get_settings()
    job = get_job_service(settings).get(job_id)
    if job is None:
        raise NotFoundError(f"job '{job_id}' not found")
    return job
