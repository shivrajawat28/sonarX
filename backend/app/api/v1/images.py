"""Image file endpoints (Section 12): original + processed artifacts."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

from backend.app.core.config import get_settings
from backend.app.core.errors import NotFoundError
from backend.app.services.inference_service import get_inference_service

router = APIRouter(prefix="/images", tags=["images"])


@router.get("/{image_id}")
def get_original_image(request: Request, image_id: str) -> Response:
    settings = get_settings()
    service = get_inference_service(settings)
    img = service.repo.get("sonar_images", image_id)
    if img is None:
        raise NotFoundError(f"image '{image_id}' not found")
    from backend.app.core.security import resolve_within

    path = resolve_within(settings.data_root, img["source_path"])
    if not path.is_file():
        raise NotFoundError(f"image file missing for '{image_id}'")
    media = "image/png" if img["format"] == "png" else "image/jpeg" if img["format"] in ("jpg", "jpeg") else "image/tiff"
    return Response(content=path.read_bytes(), media_type=media)


@router.get("/{image_id}/processed")
def get_processed_image(request: Request, image_id: str, run_id: str | None = None) -> Response:
    """Serve the latest preprocessed preview artifact for an image."""
    settings = get_settings()
    service = get_inference_service(settings)
    if service.repo.get("sonar_images", image_id) is None:
        raise NotFoundError(f"image '{image_id}' not found")
    from backend.app.core.security import resolve_within

    candidates = sorted(
        (settings.artifacts_dir / "preprocessed").glob(f"*{image_id}*.png")
        if (settings.artifacts_dir / "preprocessed").is_dir() else []
    )
    if not candidates:
        raise NotFoundError(f"no processed artifact for '{image_id}' (run a preview or detection first)")
    path = resolve_within(settings.artifacts_dir, "preprocessed", candidates[-1].name)
    return Response(content=path.read_bytes(), media_type="image/png")
