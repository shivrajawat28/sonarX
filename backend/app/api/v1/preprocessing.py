"""POST /previews/preprocess (Section 12): preprocess preview without detection."""
from __future__ import annotations

from fastapi import APIRouter, Request
import numpy as np
import cv2
from pydantic import BaseModel, Field

from backend.app.core.config import get_settings
from backend.app.core.errors import NotFoundError, ml_exception_to_api_error
from backend.app.services.inference_service import get_inference_service

router = APIRouter(prefix="/previews", tags=["preprocessing"])


class PreprocessPreviewRequest(BaseModel):
    image_id: str
    preprocess_config: str | None = Field(
        default=None, description="named preset under ml/configs/preprocessing (default: model's config)"
    )


@router.post("/preprocess")
def preprocess_preview(request: Request, body: PreprocessPreviewRequest) -> dict:
    settings = get_settings()
    service = get_inference_service(settings)
    engine = service.require_engine()

    img_doc = service.repo.get("sonar_images", body.image_id)
    if img_doc is None:
        raise NotFoundError(f"image '{body.image_id}' not found")

    from backend.app.core.security import resolve_within

    src = resolve_within(settings.data_root, img_doc["source_path"])
    pixels = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if pixels is None:
        raise NotFoundError(f"stored image file missing for {body.image_id}")

    try:
        if body.preprocess_config:
            from pathlib import Path
            from mlpipeline.config.schemas import PreprocessConfig as PC

            cfg_path = Path("ml/configs/preprocessing") / f"{body.preprocess_config}.yaml"
            cfg, h = load_cfg_file(cfg_path, PC)
        else:
            cfg, h = engine.preprocess_config, engine.preprocess_hash
        from mlpipeline.preprocessing.pipeline import PreprocessingEngine

        eng = PreprocessingEngine()
        meta, arr = eng.run_with_array(
            _gray(pixels), cfg, h, body.image_id
        )
        artifact = service.storage.save_processed_image(
            (np.clip(arr, 0, 1) * 255).round().astype(np.uint8), f"preview_{body.image_id}"
        )
    except Exception as e:  # noqa: BLE001
        raise ml_exception_to_api_error(e) from e

    return {
        "image_id": body.image_id,
        "preview_url": f"/api/v1/images/{body.image_id}/processed",
        "artifact_path": str(artifact),
        "applied_ops": [op.model_dump() for op in meta.applied_ops],
        "config_hash": meta.config_hash,
        "config_name": meta.config_name,
        "scale_factors": meta.scale_factors.model_dump(),
        "warnings": meta.warnings,
        "timings_ms": {op.op: op.duration_ms for op in meta.applied_ops},
    }


def load_cfg_file(path, cls):
    from mlpipeline.config import load_config_file

    return load_config_file(path, cls)


def _gray(pixels) -> np.ndarray:
    arr = np.asarray(pixels)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return np.clip(arr.astype(np.float32), 0.0, 1.0)
