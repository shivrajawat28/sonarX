"""Aggregates all v1 routers (Section 12)."""
from fastapi import APIRouter

from backend.app.api.v1 import (
    detections,
    exports,
    health,
    images,
    inference,
    models,
    preprocessing,
    reports,
    surveys,
    uploads,
    jobs,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(uploads.router)
api_router.include_router(images.router)
api_router.include_router(preprocessing.router)
api_router.include_router(inference.router)
api_router.include_router(surveys.router)
api_router.include_router(detections.router)
api_router.include_router(exports.router)
api_router.include_router(reports.router)
api_router.include_router(jobs.router)
api_router.include_router(models.router)
