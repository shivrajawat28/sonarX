"""Inference service: the backend's ONLY pipeline entry point (Section 8.3).

Owns engine lifecycle (degraded-mode aware, Section 11.4) and the
single-image sync path. Batch jobs loop THIS engine via mlpipeline.batch.
"""
from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np

from backend.app.core.config import Settings
from backend.app.core.errors import ModelUnavailableError, NotFoundError
from backend.app.core.security import resolve_within
from backend.app.persistence.file_repository import FileRepository, get_file_repository
from backend.app.services.storage_service import StorageService, get_storage_service
from mlpipeline.config import load_config_file
from mlpipeline.config.schemas import DetectionConfig, PreprocessConfig
from mlpipeline.inference import SonarInferenceEngine
from mlpipeline.inference.engine import ModelNotLoadedError
from mlpipeline.registry import get_registry
from mlpipeline.registry.models import resolve_repo_path

logger = logging.getLogger("api.inference")


@dataclass
class EngineState:
    engine: SonarInferenceEngine | None = None
    model_version: str | None = None
    error: str | None = None

    @property
    def loaded(self) -> bool:
        return self.engine is not None and self.engine.loaded_version is not None


class InferenceService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.storage: StorageService = get_storage_service(settings)
        self.repo: FileRepository = get_file_repository(settings)
        self.state = EngineState()

    # -- lifecycle (called from app lifespan) ---------------------------------
    def startup(self) -> None:
        """Load the active model; on failure enter DEGRADED mode (Section 11.4)."""
        registry = get_registry_at(self.settings.models_dir / "registry.json")
        entry = (
            registry.get(self.settings.active_model_version)
            if self.settings.active_model_version
            else registry.get_active()
        )
        if entry is None:
            self.state = EngineState(error="no active model registered (stub-only degraded mode)")
            logger.warning("model unavailable at startup: %s", self.state.error)
            return
        try:
            pp_cfg, pp_hash = load_config_file(
                resolve_repo_path(entry.preprocess_config_ref.path), PreprocessConfig
            )
            if pp_hash != entry.preprocess_config_ref.sha256:
                raise ValueError("preprocessing config hash drift vs registry record")
            engine = SonarInferenceEngine(
                detector_kind=entry.architecture_family,
                registry=registry,
                preprocess_config=pp_cfg,
                preprocess_hash=pp_hash,
                detection_config=DetectionConfig(
                    confidence_threshold=self.settings.confidence_threshold,
                    iou_threshold=self.settings.iou_threshold,
                ),
            )
            engine.load(entry.model_version)
            self.state = EngineState(engine=engine, model_version=entry.model_version)
            logger.info("model loaded: %s", entry.model_version, extra={"model_version": entry.model_version})
        except Exception as e:  # noqa: BLE001 — degraded mode per Section 11.4
            self.state = EngineState(error=f"model load failed: {e}")
            logger.error("model load failed: %s", e)

    def require_engine(self) -> SonarInferenceEngine:
        if not self.state.loaded:
            raise ModelUnavailableError(
                self.state.error or "no model loaded",
                details={"model_version": self.state.model_version},
            )
        return self.state.engine  # type: ignore[return-value]

    def check_model_available(self) -> None:
        """Fast-fail inference endpoints in degraded mode BEFORE other lookups
        (Section 11.4: detection endpoints return MODEL_UNAVAILABLE)."""
        if not self.state.loaded:
            raise ModelUnavailableError(
                self.state.error or "no model loaded",
                details={"model_version": self.state.model_version},
            )

    # -- single-image sync path ------------------------------------------------
    def run_for_image(
        self,
        image_id: str,
        model_version: str | None = None,
        confidence_override: float | None = None,
        save: bool = False,
    ) -> dict:
        """Load stored image -> run engine -> optionally persist run + detections."""
        img_doc = self.repo.get("sonar_images", image_id)
        if img_doc is None:
            raise NotFoundError(f"image '{image_id}' not found")

        engine = self.require_engine()
        if model_version and model_version != self.state.model_version:
            raise ModelUnavailableError(
                f"requested model '{model_version}' is not the loaded model "
                f"'{self.state.model_version}' (multi-model serving is out of MVP scope)"
            )

        pixels = self._load_image_array(img_doc["source_path"])
        params = None
        if confidence_override is not None:
            from mlpipeline.detection.base import PredictParams

            floor = self.settings.min_confidence_override
            params = PredictParams(confidence_threshold=max(confidence_override, floor))

        try:
            result = engine.run_array(pixels, image_id=image_id, params=params)
        except ModelNotLoadedError as e:
            raise ModelUnavailableError(str(e)) from e

        payload = result.model_dump()
        detection_run_id: str | None = None
        if save:
            from mlpipeline.datatypes.common import new_id

            run_doc = {
                "run_id": new_id("run"),
                "kind": "single_image",
                "image_id": image_id,
                "model_version": result.model_version,
                "preprocess_config_hash": result.preprocess_config_hash,
                "filter_config_hash": result.filter_config_hash,
                "overrides_applied": {"confidence_threshold": confidence_override} if confidence_override else {},
                "detection_ids": [d["detection_id"] for d in payload["detections"]],
                "image_ids": [image_id],
                "timings_ms": payload["timings_ms"].model_dump() if hasattr(payload["timings_ms"], "model_dump") else payload["timings_ms"],
                "warnings": result.warnings,
                "created_at": result.created_at,
            }
            run_doc = self.repo.insert("detection_runs", run_doc)
            detection_run_id = run_doc["run_id"]
            for d in payload["detections"]:
                d["run_id"] = run_doc["run_id"]
                self.repo.insert("detections", d)

        # processed-image artifact for the UI (best-effort; failure is a warning)
        try:
            pp_engine = _processed_preview_array(pixels, engine)
            artifact = self.storage.save_processed_image(pp_engine, image_id)
            payload["processed_image_ref"] = str(artifact)
        except Exception as e:  # noqa: BLE001
            logger.warning("processed-image artifact failed: %s", e)

        # Explicit (not a conditional read of `run_doc`, which is only bound when saving).
        payload["detection_run_id"] = detection_run_id
        return payload

    def _load_image_array(self, source_path: str) -> np.ndarray:
        p = resolve_within(self.settings.data_root, source_path) if not Path(source_path).is_absolute() else Path(source_path)
        data = np.fromfile(p, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            from backend.app.core.errors import InvalidImageError

            raise InvalidImageError(f"stored image undecodable: {img_doc_name(p)}")
        return img


def _processed_preview_array(pixels: np.ndarray, engine: SonarInferenceEngine) -> np.ndarray:
    """Re-run preprocessing for the preview artifact (same config, same hash)."""
    from mlpipeline.preprocessing.pipeline import PreprocessingEngine

    eng = PreprocessingEngine()
    _, arr = eng.run_with_array(
        _to_gray(pixels), engine.preprocess_config, engine.preprocess_hash, "preview"
    )
    return (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)


def _to_gray(pixels: np.ndarray) -> np.ndarray:
    arr = np.asarray(pixels)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def img_doc_name(p: Path) -> str:
    return p.name


def get_registry_at(path):
    from mlpipeline.registry.models import ModelRegistry

    return ModelRegistry(path)


_inference_service: InferenceService | None = None


def get_inference_service(settings: Settings) -> InferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService(settings)
    return _inference_service


def reset_inference_service() -> None:
    """Test helper: force re-creation with fresh settings."""
    global _inference_service
    _inference_service = None
    from backend.app.persistence.file_repository import reset_file_repository

    reset_file_repository()
